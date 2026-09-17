"""Слепая оценка (метрика 3, docs/sound_loops-iteration-3.md): на каждый луп
из набора — превью от пайплайна (match_once) и от случайного baseline
(render_once), порядок A/B перемешан случайно, оценщик не знает, что есть что.

Переиспользует match_once/render_once как есть — они пишут в renders/
video_analyses обычным образом, отдельного бухгалтерского пути в обход БД
нет: слепой прогон — это просто ещё один способ вызвать существующий код.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import psycopg
from pydantic import BaseModel

from sound_loops.config import Settings
from sound_loops.embeddings import Embedder
from sound_loops.eval_dataset import LoopAnnotation
from sound_loops.eval_run import git_commit
from sound_loops.match import match_once
from sound_loops.render import render_once
from sound_loops.vlm import SceneAnalyzer

Side = Literal["pipeline", "baseline"]


class BlindPair(BaseModel):
    loop: str
    label_a: Side
    path_a: str
    path_b: str
    rerank_reasoning: str | None = None  # объяснение модели (use_rerank=True) — печатается после ответа, не до


class BlindAnswer(BaseModel):
    loop: str
    label_a: Side
    choice: Literal["A", "B", "tie"]
    winner: Literal["pipeline", "baseline", "tie"]


class BlindRun(BaseModel):
    timestamp: str
    commit: str
    answers: list[BlindAnswer]
    pipeline_win_rate: float  # доля побед pipeline среди пар без ничьей


def prepare_pairs(
    conn: psycopg.Connection,
    analyzer: SceneAnalyzer,
    embedder: Embedder,
    settings: Settings,
    dataset: list[LoopAnnotation],
    use_filters: bool = False,
    use_rerank: bool = False,
) -> list[BlindPair]:
    pairs = []
    for entry in dataset:
        loop_path = Path(entry.loop)
        pipeline_result = match_once(
            conn, analyzer, embedder, settings, loop_path, use_filters=use_filters, use_rerank=use_rerank
        )
        baseline_path = render_once(conn, settings, loop_path)

        label_a: Side = random.choice(["pipeline", "baseline"])
        pipeline_path, baseline_path_str = str(pipeline_result.output_path), str(baseline_path)
        path_a = pipeline_path if label_a == "pipeline" else baseline_path_str
        path_b = baseline_path_str if label_a == "pipeline" else pipeline_path

        pairs.append(
            BlindPair(
                loop=entry.loop,
                label_a=label_a,
                path_a=path_a,
                path_b=path_b,
                rerank_reasoning=pipeline_result.rerank_reasoning,
            )
        )
    return pairs


def _winner(label_a: Side, choice: str) -> Literal["pipeline", "baseline", "tie"]:
    if choice == "tie":
        return "tie"
    label_b: Side = "baseline" if label_a == "pipeline" else "pipeline"
    return label_a if choice == "A" else label_b


def score_pairs(pairs: list[BlindPair], ask: Callable[[BlindPair], str]) -> BlindRun:
    answers = []
    for pair in pairs:
        choice = ask(pair)
        winner = _winner(pair.label_a, choice)
        answers.append(BlindAnswer(loop=pair.loop, label_a=pair.label_a, choice=choice, winner=winner))

    decisive = [a for a in answers if a.winner != "tie"]
    pipeline_wins = sum(1 for a in decisive if a.winner == "pipeline")
    win_rate = pipeline_wins / len(decisive) if decisive else 0.0

    return BlindRun(
        timestamp=datetime.now(UTC).isoformat(), commit=git_commit(), answers=answers, pipeline_win_rate=win_rate
    )


def save_blind_run(run: BlindRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = run.timestamp.replace(":", "").replace("-", "").split(".")[0]
    path = runs_dir / f"{ts}_{run.commit}.json"
    path.write_text(run.model_dump_json(indent=2) + "\n")
    return path
