"""Команда eval-run:
прогон пайплайна по размеченному набору (evals/dataset.json), метрики
понимания сцены (шаг A) и поиска (шаг B + CLAP).

Переиспользует analyze_loop и search_tracks
как есть — не дублирует их логику и не пишет в renders (аудио не рендерит,
для этого есть blind-eval). Температуру задаёт вызывающий код (CLI фиксирует
0.0 для воспроизводимости).

use_rerank реранжирует только топ-RERANK_POOL_SIZE (см. rerank.py) —
hit@1 после этого отражает реальный выбор модели, а hit@5/best_rank
считаются по тому же полному списку (search_depth), что и в конфигурациях
без переранжирования: выбор модели просто поднимается на первое место,
остальные места 4+ не трогаются. Это специально сделано сравнимым с
конфигурацией без переранжирования, а не отдельной метрикой.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from pydantic import BaseModel

from sound_loops.agent_graph import build_eval_graph, initial_state
from sound_loops.analysis import analyze_loop
from sound_loops.config import Settings
from sound_loops.embeddings import Embedder
from sound_loops.eval_dataset import LoopAnnotation
from sound_loops.eval_metrics import best_rank, hit_at_k, mood_overlap, penalized_rank
from sound_loops.ffmpeg_utils import extract_frames
from sound_loops.motion import estimate_motion
from sound_loops.render import get_loop_by_path
from sound_loops.rerank import RERANK_POOL_SIZE, rerank_candidates
from sound_loops.search import search_tracks
from sound_loops.vlm import Motion, SceneAnalyzer, SceneDescription


class LoopEvalResult(BaseModel):
    loop: str
    setting_correct: bool
    predicted_setting: str
    true_setting: str
    mood_overlap: float
    predicted_mood: list[str]
    true_mood: list[str]
    music_query: str
    hit_at_1: bool
    hit_at_5: bool
    best_rank: int | None
    rerank_reasoning: str | None = None
    # Только для use_agent=True: queries — 3 запроса узла plan,
    # tool_calls_made/fallback_used — надёжность вызова инструмента на этом лупе.
    queries: list[str] = []
    tool_calls_made: int = 0
    fallback_used: int = 0


class EvalAggregates(BaseModel):
    setting_accuracy: float
    mean_mood_overlap: float
    hit_at_1_rate: float
    hit_at_5_rate: float
    mean_best_rank: float | None
    not_found_count: int
    # mean_penalized_rank считает не найденное как search_depth и годится для сравнения конфигураций
    # между собой — mean_best_rank оставлен для того, что реально нашлось.
    mean_penalized_rank: float
    # Только при use_agent=True: сколько раз узел plan вызвал инструмент
    # сам и сколько раз сработал резервный путь — суммарно по всем лупам,
    # честная характеристика надёжности tool calling на локальной 4B-модели.
    agent_tool_calls_total: int = 0
    agent_fallback_used_total: int = 0


class EvalRun(BaseModel):
    timestamp: str
    commit: str
    model: str
    prompt_version: str
    temperature: float
    search_depth: int
    use_rerank: bool = False
    use_agent: bool = False
    loops: list[LoopEvalResult]
    aggregates: EvalAggregates

    def print_summary(self) -> None:
        a = self.aggregates
        print(
            f"Модель: {self.model}  промпт: {self.prompt_version}  "
            f"температура: {self.temperature}  коммит: {self.commit}"
        )
        if self.use_agent:
            print("Конфигурация: агент (узел plan вызывает поиск сам, hit@k по объединению 3 query)")
        else:
            print(f"Конфигурация: переранжирование={'да' if self.use_rerank else 'нет'}")
        print(f"Лупов: {len(self.loops)}")
        print(f"setting accuracy:    {a.setting_accuracy:.2f}")
        print(f"mood overlap (mean): {a.mean_mood_overlap:.2f}")
        print(f"hit@1:               {a.hit_at_1_rate:.2f}")
        print(f"hit@5:               {a.hit_at_5_rate:.2f}")
        if a.mean_best_rank is not None:
            print(
                f"mean best rank:      {a.mean_best_rank:.1f}  "
                f"(не найдено в топ-{self.search_depth}: {a.not_found_count})"
            )
        else:
            print(f"mean best rank:      n/a (не найдено ни разу в топ-{self.search_depth})")
        print(
            f"mean penalized rank: {a.mean_penalized_rank:.1f}  "
            "(не найдено считается как search_depth — для сравнения конфигураций между собой)"
        )
        if self.use_agent:
            total_calls = a.agent_tool_calls_total + a.agent_fallback_used_total
            fallback_share = a.agent_fallback_used_total / total_calls if total_calls else 0.0
            print(
                f"вызовов инструмента моделью: {a.agent_tool_calls_total}  "
                f"резервных: {a.agent_fallback_used_total}  (доля резервных: {fallback_share:.0%})"
            )
        if self.use_rerank:
            print("\nОбъяснения переранжирования:")
            for r in self.loops:
                print(f"  {r.loop}: {r.rerank_reasoning}")


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _evaluate_loop(
    conn: psycopg.Connection,
    analyzer: SceneAnalyzer,
    embedder: Embedder,
    settings: Settings,
    entry: LoopAnnotation,
    use_rerank: bool,
) -> LoopEvalResult:
    loop = get_loop_by_path(conn, Path(entry.loop), settings)

    def get_frames() -> list[bytes]:
        return extract_frames(
            Path(loop.path), loop.duration_seconds, settings.vlm_frame_count, settings.vlm_frame_max_side
        )

    def get_motion() -> Motion:
        return estimate_motion(Path(loop.path), settings.motion_sample_fps, settings.motion_frame_size)

    record, _cached = analyze_loop(conn, analyzer, loop.id, get_frames, get_motion)
    scene = record.scene

    music_query = analyzer.compose_music_query(scene)

    candidates = search_tracks(
        conn, embedder, music_query.query, settings.eval_search_depth, min_duration_seconds=loop.duration_seconds
    )

    rerank_reasoning = None
    if use_rerank:
        pool = candidates[:RERANK_POOL_SIZE]
        rerank_result = rerank_candidates(analyzer, scene, pool)
        candidates = rerank_result.reordered + candidates[RERANK_POOL_SIZE:]
        rerank_reasoning = rerank_result.reasoning

    ranked_paths = [c.path for c in candidates]

    return LoopEvalResult(
        loop=entry.loop,
        setting_correct=scene.setting == entry.setting,
        predicted_setting=scene.setting,
        true_setting=entry.setting,
        mood_overlap=mood_overlap(scene.mood, entry.mood),
        predicted_mood=list(scene.mood),
        true_mood=list(entry.mood),
        music_query=music_query.query,
        hit_at_1=hit_at_k(ranked_paths, entry.good_tracks, 1),
        hit_at_5=hit_at_k(ranked_paths, entry.good_tracks, 5),
        best_rank=best_rank(ranked_paths, entry.good_tracks),
        rerank_reasoning=rerank_reasoning,
    )


def _merge_agent_pools(plan_pools: list[dict]) -> list[str]:
    """Объединить кандидатов всех slot'ов агента в один ранжированный список
    путей — дедуп по треку (побеждает лучшая позиция), сортировка по
    сходству. Основа для hit@k агента: трек считается найденным, если он
    попал в top-k хотя бы одного из 3 query."""
    best: dict[int, tuple[float, str]] = {}
    for pool_info in plan_pools:
        for candidate in pool_info["candidates"]:
            track_id, similarity, path = candidate["id"], candidate["similarity"], candidate["path"]
            if track_id not in best or similarity > best[track_id][0]:
                best[track_id] = (similarity, path)
    return [path for _similarity, path in sorted(best.values(), key=lambda pair: pair[0], reverse=True)]


def _evaluate_loop_agent(graph, settings: Settings, entry: LoopAnnotation) -> LoopEvalResult:
    """Первый проход графа-агента (analyze -> plan -> rerank) — не дублирует его логику."""
    state = graph.invoke(initial_state(entry.loop, settings))
    scene = SceneDescription.model_validate(state["scene"])
    queries = [pool["query"] for pool in state["plan_pools"]]
    ranked_paths = _merge_agent_pools(state["plan_pools"])

    return LoopEvalResult(
        loop=entry.loop,
        setting_correct=scene.setting == entry.setting,
        predicted_setting=scene.setting,
        true_setting=entry.setting,
        mood_overlap=mood_overlap(scene.mood, entry.mood),
        predicted_mood=list(scene.mood),
        true_mood=list(entry.mood),
        music_query="; ".join(queries),
        queries=queries,
        hit_at_1=hit_at_k(ranked_paths, entry.good_tracks, 1),
        hit_at_5=hit_at_k(ranked_paths, entry.good_tracks, 5),
        best_rank=best_rank(ranked_paths, entry.good_tracks),
        tool_calls_made=state["tool_calls_total"],
        fallback_used=state["fallback_used_total"],
    )


def _aggregate(loops: list[LoopEvalResult], search_depth: int) -> EvalAggregates:
    n = len(loops)
    ranks = [r.best_rank for r in loops if r.best_rank is not None]
    return EvalAggregates(
        setting_accuracy=sum(r.setting_correct for r in loops) / n,
        mean_mood_overlap=sum(r.mood_overlap for r in loops) / n,
        hit_at_1_rate=sum(r.hit_at_1 for r in loops) / n,
        hit_at_5_rate=sum(r.hit_at_5 for r in loops) / n,
        mean_best_rank=(sum(ranks) / len(ranks)) if ranks else None,
        not_found_count=n - len(ranks),
        mean_penalized_rank=sum(penalized_rank(r.best_rank, search_depth) for r in loops) / n,
        agent_tool_calls_total=sum(r.tool_calls_made for r in loops),
        agent_fallback_used_total=sum(r.fallback_used for r in loops),
    )


def run_eval(
    conn: psycopg.Connection,
    analyzer: SceneAnalyzer,
    embedder: Embedder,
    settings: Settings,
    dataset: list[LoopAnnotation],
    temperature: float,
    use_rerank: bool = False,
    use_agent: bool = False,
) -> EvalRun:
    if not dataset:
        raise ValueError("набор разметки пуст — нечего прогонять")

    if use_agent:
        eval_settings = settings.model_copy(update={"agent_search_pool_size": settings.eval_search_depth})
        graph = build_eval_graph(conn, embedder, analyzer, eval_settings)
        loops = [_evaluate_loop_agent(graph, eval_settings, entry) for entry in dataset]
    else:
        loops = [_evaluate_loop(conn, analyzer, embedder, settings, entry, use_rerank) for entry in dataset]
    return EvalRun(
        timestamp=datetime.now(UTC).isoformat(),
        use_rerank=use_rerank,
        use_agent=use_agent,
        commit=git_commit(),
        model=analyzer.model_id,
        prompt_version=analyzer.prompt_version,
        temperature=temperature,
        search_depth=settings.eval_search_depth,
        loops=loops,
        aggregates=_aggregate(loops, settings.eval_search_depth),
    )


def save_run(run: EvalRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = run.timestamp.replace(":", "").replace("-", "").split(".")[0]
    path = runs_dir / f"{ts}_{run.commit}.json"
    path.write_text(run.model_dump_json(indent=2) + "\n")
    return path


def load_run(path: Path) -> EvalRun:
    raw = json.loads(path.read_text())
    aggregates = raw.get("aggregates", {})
    if "mean_penalized_rank" not in aggregates:
        loops = raw["loops"]
        search_depth = raw["search_depth"]
        n = len(loops)
        aggregates["mean_penalized_rank"] = (
            sum(penalized_rank(loop.get("best_rank"), search_depth) for loop in loops) / n if n else 0.0
        )
    return EvalRun.model_validate(raw)
