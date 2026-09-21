"""Анализ сцены лупа привязан к (loop_id, model, prompt_version), повторный запуск с тем же ключом
не гоняет VLM заново — прогон по кадрам занимает десятки секунд.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg

from sound_loops.config import Settings
from sound_loops.ffmpeg_utils import extract_frames
from sound_loops.motion import estimate_motion
from sound_loops.render import LoopRow, get_loop_by_path, get_random_loop
from sound_loops.vlm import Motion, SceneAnalyzer, SceneDescription


@dataclass(frozen=True)
class AnalysisRecord:
    id: int
    scene: SceneDescription


def get_cached_analysis(
    conn: psycopg.Connection, loop_id: int, model: str, prompt_version: str
) -> AnalysisRecord | None:
    row = conn.execute(
        """
        SELECT id, setting, motion, mood FROM video_analyses
        WHERE loop_id = %s AND model = %s AND prompt_version = %s
        """,
        (loop_id, model, prompt_version),
    ).fetchone()
    if row is None:
        return None
    analysis_id, setting, motion, mood = row
    return AnalysisRecord(analysis_id, SceneDescription(setting=setting, motion=motion, mood=mood))


def save_analysis(
    conn: psycopg.Connection,
    loop_id: int,
    model: str,
    prompt_version: str,
    scene: SceneDescription,
) -> AnalysisRecord:
    row = conn.execute(
        """
        INSERT INTO video_analyses (loop_id, model, prompt_version, setting, motion, mood)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (loop_id, model, prompt_version, scene.setting, scene.motion, scene.mood),
    ).fetchone()
    conn.commit()
    return AnalysisRecord(row[0], scene)


def analyze_loop(
    conn: psycopg.Connection,
    analyzer: SceneAnalyzer,
    loop_id: int,
    get_frames: Callable[[], Sequence[bytes]],
    get_motion: Callable[[], Motion],
) -> tuple[AnalysisRecord, bool]:
    """get_frames/get_motion — лениво: при попадании в кеш ни VLM, ни разница
    кадров вообще не считаются."""
    cached = get_cached_analysis(conn, loop_id, analyzer.model_id, analyzer.prompt_version)
    if cached is not None:
        return cached, True

    observation = analyzer.describe_scene(get_frames())
    scene = SceneDescription(setting=observation.setting, motion=get_motion(), mood=observation.mood)
    return save_analysis(conn, loop_id, analyzer.model_id, analyzer.prompt_version, scene), False


def analyze_loop_by_path(
    conn: psycopg.Connection,
    analyzer: SceneAnalyzer,
    settings: Settings,
    loop_path: Path | None = None,
) -> tuple[LoopRow, AnalysisRecord, bool]:
    """Разрешить луп (по пути или случайный) и прогнать analyze_loop с реальным
    извлечением кадров/оценкой motion. Точка входа команды `analyze`."""
    loop = get_loop_by_path(conn, loop_path, settings) if loop_path else get_random_loop(conn)

    def get_frames() -> list[bytes]:
        return extract_frames(
            Path(loop.path), loop.duration_seconds, settings.vlm_frame_count, settings.vlm_frame_max_side
        )

    def get_motion() -> Motion:
        return estimate_motion(Path(loop.path), settings.motion_sample_fps, settings.motion_frame_size)

    record, cached = analyze_loop(conn, analyzer, loop.id, get_frames, get_motion)
    return loop, record, cached
