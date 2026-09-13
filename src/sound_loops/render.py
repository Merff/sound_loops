"""Сборка превью: луп + случайный отрезок трека, подрезанный под его длительность.

Кандидат на отрезок выбирается из tracks и вырезается на лету — заранее
никакая сетка не считается. Координаты использованного куска (track_id,
start_seconds) сохраняются прямо в renders — отдельной таблицы под них
не заводим, так как каждый отрезок используется ровно в одном рендере.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import psycopg

from sound_loops.config import Settings
from sound_loops.ffmpeg_utils import extract_audio_segment, mux_loop_with_audio
from sound_loops.ingest import ingest_loop_file
from sound_loops.segments import pick_random_start


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoopRow:
    id: int
    path: str
    duration_seconds: float


@dataclass(frozen=True)
class TrackRow:
    id: int
    path: str
    duration_seconds: float


def get_loop_by_path(conn: psycopg.Connection, path: Path, settings: Settings) -> LoopRow:
    """Найти луп в базе по пути; если его там ещё нет — провалидировать и заапсертить."""
    row = conn.execute(
        "SELECT id, path, duration_seconds FROM loops WHERE path = %s", (str(path),)
    ).fetchone()
    if row is not None:
        return LoopRow(*row)

    loop_id, note = ingest_loop_file(conn, path, settings)
    if loop_id is None:
        raise RenderError(f"луп {path} не прошёл проверку: {note}")

    row = conn.execute(
        "SELECT id, path, duration_seconds FROM loops WHERE id = %s", (loop_id,)
    ).fetchone()
    return LoopRow(*row)


def get_random_loop(conn: psycopg.Connection) -> LoopRow:
    row = conn.execute(
        "SELECT id, path, duration_seconds FROM loops ORDER BY random() LIMIT 1"
    ).fetchone()
    if row is None:
        raise RenderError("в базе нет лупов — сначала запустите ingest")
    return LoopRow(*row)


def get_random_track(conn: psycopg.Connection, min_duration_seconds: float) -> TrackRow:
    """Взять случайный трек, которого хватит на всю длительность лупа."""
    row = conn.execute(
        """
        SELECT id, path, duration_seconds
        FROM tracks
        WHERE duration_seconds >= %s
        ORDER BY random()
        LIMIT 1
        """,
        (min_duration_seconds,),
    ).fetchone()
    if row is None:
        raise RenderError(
            "в базе нет треков достаточной длины — сначала запустите ingest"
        )
    return TrackRow(*row)


def render_once(
    conn: psycopg.Connection,
    settings: Settings,
    loop_path: Path | None = None,
) -> Path:
    """Собрать одно превью: луп + случайный отрезок трека под его длительность."""
    loop = get_loop_by_path(conn, loop_path, settings) if loop_path else get_random_loop(conn)
    track = get_random_track(conn, loop.duration_seconds)
    start_seconds = pick_random_start(track.duration_seconds, loop.duration_seconds)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{Path(loop.path).stem}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = settings.output_dir / output_name

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp_audio_path = Path(tmp.name)
    try:
        extract_audio_segment(
            track_path=Path(track.path),
            start_seconds=start_seconds,
            duration_seconds=loop.duration_seconds,
            fade_seconds=settings.fade_seconds,
            output_path=tmp_audio_path,
        )
        mux_loop_with_audio(Path(loop.path), tmp_audio_path, output_path)
    finally:
        tmp_audio_path.unlink(missing_ok=True)

    conn.execute(
        """
        INSERT INTO renders (loop_id, track_id, start_seconds, output_path, duration_seconds)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (loop.id, track.id, start_seconds, str(output_path), loop.duration_seconds),
    )
    conn.commit()

    return output_path
