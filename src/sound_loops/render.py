"""Сборка превью: трек проигрывается с начала (почти) целиком, под его
длительность видео-луп повторяется целое число раз.

Трек — естественная единица длительности превью (FMA-отрывки и кураторские
треки ~30с, длиннее любого лупа): луп повторяется, пока не заполнит трек,
а не наоборот. Если трек не делится на длительность лупа без остатка,
лишний хвост (короче одного повтора лупа) отбрасывается с конца — луп либо
крутится целиком, либо не крутится вовсе, обрывать его на середине цикла не
хотим. Кандидат на трек выбирается из tracks — заранее никакая сетка не
считается. track_id/start_seconds (всегда 0.0 — трек начинается с начала)
сохраняются прямо в renders — отдельной таблицы под них не заводим, так как
каждый отрезок используется ровно в одном рендере.
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


class RenderError(RuntimeError):
    pass


def compute_repeat_count(loop_duration_seconds: float, track_duration_seconds: float) -> int:
    """Сколько раз луп целиком помещается в трек — заполняем длительность
    трека повторами лупа, остаток короче одного повтора отбрасывается.
    Минимум 1 (не бывает нулевой длительности рендера), хотя на практике
    трек всегда не короче лупа (см. get_random_track/search)."""
    return max(1, int(track_duration_seconds // loop_duration_seconds))


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


def render_preview(
    conn: psycopg.Connection,
    settings: Settings,
    loop: LoopRow,
    track_id: int,
    track_path: str,
    track_duration_seconds: float,
    analysis_id: int | None = None,
    music_query: str | None = None,
) -> tuple[int, Path]:
    """Взять трек (почти) целиком, повторить луп нужное число раз, склеить и
    записать renders. Общий хвост render_once/match_once/агентного узла
    render (итерация 5) — каждый по-своему выбирает трек, дальше всё
    одинаково. analysis_id/music_query — NULL для случайного baseline'а.
    Возвращает (id рендера, путь) — id нужен агентному UI, чтобы потом
    привязать к этому конкретному превью оценку пользователя (rating)."""
    repeat_count = compute_repeat_count(loop.duration_seconds, track_duration_seconds)
    final_duration = repeat_count * loop.duration_seconds
    start_seconds = 0.0  # всегда с начала трека, остаток короче лупа отбрасывается с конца

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{Path(loop.path).stem}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = settings.output_dir / output_name

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp_audio_path = Path(tmp.name)
    try:
        extract_audio_segment(
            track_path=Path(track_path),
            start_seconds=start_seconds,
            duration_seconds=final_duration,
            fade_seconds=settings.fade_seconds,
            output_path=tmp_audio_path,
        )
        mux_loop_with_audio(Path(loop.path), tmp_audio_path, output_path, repeat_count=repeat_count)
    finally:
        tmp_audio_path.unlink(missing_ok=True)

    row = conn.execute(
        """
        INSERT INTO renders
            (loop_id, track_id, start_seconds, output_path, duration_seconds, analysis_id, music_query)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (loop.id, track_id, start_seconds, str(output_path), final_duration, analysis_id, music_query),
    ).fetchone()
    conn.commit()

    return row[0], output_path


def render_once(
    conn: psycopg.Connection,
    settings: Settings,
    loop_path: Path | None = None,
) -> Path:
    loop = get_loop_by_path(conn, loop_path, settings) if loop_path else get_random_loop(conn)
    track = get_random_track(conn, loop.duration_seconds)
    _render_id, output_path = render_preview(conn, settings, loop, track.id, track.path, track.duration_seconds)
    return output_path


def set_render_rating(conn: psycopg.Connection, render_id: int, rating: str) -> None:
    """Пользовательская оценка превью в UI (итерация 5) — good/neutral/bad,
    см. migrations/0007. rating валиден по CHECK в схеме, здесь не дублируем
    проверку — некорректное значение просто упадёт на INSERT/UPDATE."""
    conn.execute("UPDATE renders SET rating = %s WHERE id = %s", (rating, render_id))
    conn.commit()


def get_bad_rated_track_ids(conn: psycopg.Connection, loop_id: int) -> list[int]:
    """Треки, отмеченные "плохо" именно для этого лупа — узел plan (агент,
    итерация 5) исключает их из поиска для этого лупа во всех будущих
    сессиях, не только в пределах текущей (см. docs/sound_loops-iteration-5.md).
    Намеренно не глобально: трек, не подошедший одной сцене, может подойти
    другой."""
    rows = conn.execute(
        "SELECT DISTINCT track_id FROM renders WHERE loop_id = %s AND rating = 'bad'", (loop_id,)
    ).fetchall()
    return [row[0] for row in rows]
