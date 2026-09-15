"""Команда eval-run (итерация 3, docs/sound_loops-iteration-3.md): прогон
пайплайна по размеченному набору (evals/dataset.json), метрики понимания
сцены (шаг A) и поиска (шаг B + CLAP).

Переиспользует analyze_loop (кеш шага A) и search_tracks как есть — не
дублирует их логику и не пишет в renders (аудио не рендерит, для этого
есть blind-eval). Температуру задаёт вызывающий код (CLI фиксирует 0.0
для воспроизводимости, см. docs).
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from pydantic import BaseModel

from sound_loops.analysis import analyze_loop
from sound_loops.config import Settings
from sound_loops.embeddings import Embedder
from sound_loops.eval_dataset import LoopAnnotation
from sound_loops.eval_metrics import best_rank, hit_at_k, mood_overlap
from sound_loops.ffmpeg_utils import extract_frames
from sound_loops.motion import estimate_motion
from sound_loops.render import get_loop_by_path
from sound_loops.search import search_tracks
from sound_loops.vlm import Motion, SceneAnalyzer


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


class EvalAggregates(BaseModel):
    setting_accuracy: float
    mean_mood_overlap: float
    hit_at_1_rate: float
    hit_at_5_rate: float
    mean_best_rank: float | None
    not_found_count: int


class EvalRun(BaseModel):
    timestamp: str
    commit: str
    model: str
    prompt_version: str
    temperature: float
    search_depth: int
    loops: list[LoopEvalResult]
    aggregates: EvalAggregates

    def print_summary(self) -> None:
        a = self.aggregates
        print(
            f"Модель: {self.model}  промпт: {self.prompt_version}  "
            f"температура: {self.temperature}  коммит: {self.commit}"
        )
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

    music_query = analyzer.compose_music_query(scene).query
    candidates = search_tracks(
        conn, embedder, music_query, settings.eval_search_depth, min_duration_seconds=loop.duration_seconds
    )
    ranked_paths = [c.path for c in candidates]

    return LoopEvalResult(
        loop=entry.loop,
        setting_correct=scene.setting == entry.setting,
        predicted_setting=scene.setting,
        true_setting=entry.setting,
        mood_overlap=mood_overlap(scene.mood, entry.mood),
        predicted_mood=list(scene.mood),
        true_mood=list(entry.mood),
        music_query=music_query,
        hit_at_1=hit_at_k(ranked_paths, entry.good_tracks, 1),
        hit_at_5=hit_at_k(ranked_paths, entry.good_tracks, 5),
        best_rank=best_rank(ranked_paths, entry.good_tracks),
    )


def _aggregate(loops: list[LoopEvalResult]) -> EvalAggregates:
    n = len(loops)
    ranks = [r.best_rank for r in loops if r.best_rank is not None]
    return EvalAggregates(
        setting_accuracy=sum(r.setting_correct for r in loops) / n,
        mean_mood_overlap=sum(r.mood_overlap for r in loops) / n,
        hit_at_1_rate=sum(r.hit_at_1 for r in loops) / n,
        hit_at_5_rate=sum(r.hit_at_5 for r in loops) / n,
        mean_best_rank=(sum(ranks) / len(ranks)) if ranks else None,
        not_found_count=n - len(ranks),
    )


def run_eval(
    conn: psycopg.Connection,
    analyzer: SceneAnalyzer,
    embedder: Embedder,
    settings: Settings,
    dataset: list[LoopAnnotation],
    temperature: float,
) -> EvalRun:
    if not dataset:
        raise ValueError("набор разметки пуст — нечего прогонять")

    loops = [_evaluate_loop(conn, analyzer, embedder, settings, entry) for entry in dataset]
    return EvalRun(
        timestamp=datetime.now(UTC).isoformat(),
        commit=git_commit(),
        model=analyzer.model_id,
        prompt_version=analyzer.prompt_version,
        temperature=temperature,
        search_depth=settings.eval_search_depth,
        loops=loops,
        aggregates=_aggregate(loops),
    )


def save_run(run: EvalRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = run.timestamp.replace(":", "").replace("-", "").split(".")[0]
    path = runs_dir / f"{ts}_{run.commit}.json"
    path.write_text(run.model_dump_json(indent=2) + "\n")
    return path


def load_run(path: Path) -> EvalRun:
    return EvalRun.model_validate_json(path.read_text())
