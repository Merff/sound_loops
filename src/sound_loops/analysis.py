"""Кеш шага A (video_analyses): анализ сцены лупа привязан к (loop_id,
model, prompt_version), повторный запуск с тем же ключом не гоняет VLM
заново — прогон по кадрам занимает десятки секунд, а формулировки шага B
на этом кеше можно крутить десятки раз за минуты (docs/sound_loops-iteration-2.md).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import psycopg

from sound_loops.vlm import SceneAnalyzer, SceneDescription


@dataclass(frozen=True)
class AnalysisRecord:
    id: int
    scene: SceneDescription


def get_cached_analysis(
    conn: psycopg.Connection, loop_id: int, model: str, prompt_version: str
) -> AnalysisRecord | None:
    row = conn.execute(
        """
        SELECT id, summary, motion, mood, is_comic FROM video_analyses
        WHERE loop_id = %s AND model = %s AND prompt_version = %s
        """,
        (loop_id, model, prompt_version),
    ).fetchone()
    if row is None:
        return None
    analysis_id, summary, motion, mood, is_comic = row
    return AnalysisRecord(analysis_id, SceneDescription(summary=summary, motion=motion, mood=mood, is_comic=is_comic))


def save_analysis(
    conn: psycopg.Connection,
    loop_id: int,
    model: str,
    prompt_version: str,
    scene: SceneDescription,
) -> AnalysisRecord:
    row = conn.execute(
        """
        INSERT INTO video_analyses (loop_id, model, prompt_version, summary, motion, mood, is_comic)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (loop_id, model, prompt_version, scene.summary, scene.motion, scene.mood, scene.is_comic),
    ).fetchone()
    conn.commit()
    return AnalysisRecord(row[0], scene)


def analyze_loop(
    conn: psycopg.Connection,
    analyzer: SceneAnalyzer,
    loop_id: int,
    get_frames: Callable[[], Sequence[bytes]],
) -> tuple[AnalysisRecord, bool]:
    """Вернуть (запись анализа, взята_ли_из_кеша).

    get_frames — кадры лупа, извлекаются лениво: при попадании в кеш ffmpeg
    вообще не запускается.
    """
    cached = get_cached_analysis(conn, loop_id, analyzer.model_id, analyzer.prompt_version)
    if cached is not None:
        return cached, True

    scene = analyzer.describe_scene(get_frames())
    return save_analysis(conn, loop_id, analyzer.model_id, analyzer.prompt_version, scene), False
