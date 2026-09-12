"""Обход директорий с лупами и музыкой, заполнение таблиц.

Путь — естественный ключ уникальности для лупов и треков: повторный
запуск обновляет существующие строки, а не создаёт дубликаты.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from sound_loops.config import Settings
from sound_loops.ffmpeg_utils import FfmpegError, probe
from sound_loops.metadata import load_metadata
from sound_loops.segments import compute_segment_grid

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    loops_added: int = 0
    loops_updated: int = 0
    loops_skipped: list[tuple[Path, str]] = field(default_factory=list)
    tracks_added: int = 0
    tracks_updated: int = 0
    tracks_skipped: list[tuple[Path, str]] = field(default_factory=list)
    segments_created: int = 0

    def print_summary(self) -> None:
        print(
            f"Лупы: добавлено {self.loops_added}, обновлено {self.loops_updated}, "
            f"пропущено {len(self.loops_skipped)}"
        )
        for path, reason in self.loops_skipped:
            print(f"  пропущен {path}: {reason}")

        print(
            f"Треки: добавлено {self.tracks_added}, обновлено {self.tracks_updated}, "
            f"пропущено {len(self.tracks_skipped)}"
        )
        for path, reason in self.tracks_skipped:
            print(f"  пропущен {path}: {reason}")

        print(f"Отрезков создано: {self.segments_created}")


def ingest_loop_file(
    conn: psycopg.Connection, path: Path, settings: Settings
) -> tuple[int | None, str | None]:
    """Провалидировать и заапсертить один луп. Вернуть (loop_id, причина_пропуска)."""
    try:
        result = probe(path)
    except FfmpegError as exc:
        return None, f"ffprobe не смог прочитать файл: {exc}"

    if result.has_audio:
        return None, "у лупа есть аудиодорожка, ожидался немой файл"
    if not (settings.min_loop_seconds <= result.duration_seconds <= settings.max_loop_seconds):
        return None, (
            f"длительность {result.duration_seconds:.2f}с вне диапазона "
            f"[{settings.min_loop_seconds}, {settings.max_loop_seconds}]"
        )

    row = conn.execute(
        """
        INSERT INTO loops (path, duration_seconds, width, height)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (path) DO UPDATE SET
            duration_seconds = EXCLUDED.duration_seconds,
            width = EXCLUDED.width,
            height = EXCLUDED.height
        RETURNING id, (xmax = 0) AS inserted
        """,
        (str(path), result.duration_seconds, result.width, result.height),
    ).fetchone()
    conn.commit()
    loop_id, inserted = row
    return loop_id, None if inserted else "updated"


def scan_loops(conn: psycopg.Connection, settings: Settings, report: IngestReport) -> None:
    if not settings.loops_dir.exists():
        logger.warning("директория с лупами не найдена: %s", settings.loops_dir)
        return

    for path in sorted(settings.loops_dir.rglob("*.mp4")):
        loop_id, note = ingest_loop_file(conn, path, settings)
        if loop_id is None:
            report.loops_skipped.append((path, note or "неизвестная причина"))
        elif note == "updated":
            report.loops_updated += 1
        else:
            report.loops_added += 1


def ingest_track_file(
    conn: psycopg.Connection,
    path: Path,
    settings: Settings,
    metadata: dict,
    report: IngestReport,
) -> None:
    try:
        fma_track_id = int(path.stem)
    except ValueError:
        report.tracks_skipped.append((path, "имя файла не похоже на ID трека FMA"))
        return

    try:
        result = probe(path)
    except FfmpegError as exc:
        report.tracks_skipped.append((path, f"ffprobe не смог прочитать файл: {exc}"))
        return

    if not result.has_audio:
        report.tracks_skipped.append((path, "нет аудиодорожки"))
        return

    meta = metadata.get(fma_track_id)
    title = meta.title if meta else ""
    artist = meta.artist if meta else ""
    genre = meta.genre if meta else ""

    row = conn.execute(
        """
        INSERT INTO tracks (path, duration_seconds, fma_track_id, title, artist, genre)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (path) DO UPDATE SET
            duration_seconds = EXCLUDED.duration_seconds,
            fma_track_id = EXCLUDED.fma_track_id,
            title = EXCLUDED.title,
            artist = EXCLUDED.artist,
            genre = EXCLUDED.genre
        RETURNING id, (xmax = 0) AS inserted
        """,
        (str(path), result.duration_seconds, fma_track_id, title, artist, genre),
    ).fetchone()
    track_id, inserted = row
    if inserted:
        report.tracks_added += 1
    else:
        report.tracks_updated += 1

    grid = compute_segment_grid(
        result.duration_seconds, settings.segment_seconds, settings.min_segment_seconds
    )
    for start, duration in grid:
        seg_row = conn.execute(
            """
            INSERT INTO track_segments (track_id, start_seconds, duration_seconds)
            VALUES (%s, %s, %s)
            ON CONFLICT (track_id, start_seconds) DO NOTHING
            RETURNING id
            """,
            (track_id, start, duration),
        ).fetchone()
        if seg_row is not None:
            report.segments_created += 1

    conn.commit()


def scan_tracks(conn: psycopg.Connection, settings: Settings, report: IngestReport) -> None:
    if not settings.music_dir.exists():
        logger.warning("директория с музыкой не найдена: %s", settings.music_dir)
        return

    metadata = load_metadata(settings.metadata_csv)

    for path in sorted(settings.music_dir.rglob("*.mp3")):
        ingest_track_file(conn, path, settings, metadata, report)


def ingest(conn: psycopg.Connection, settings: Settings) -> IngestReport:
    report = IngestReport()
    scan_loops(conn, settings, report)
    scan_tracks(conn, settings, report)
    return report
