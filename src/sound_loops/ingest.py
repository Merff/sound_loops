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
from sound_loops.ffmpeg_utils import FfmpegError, ProbeResult, probe
from sound_loops.metadata import load_metadata

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    loops_added: int = 0
    loops_updated: int = 0
    loops_skipped: list[tuple[Path, str]] = field(default_factory=list)
    tracks_added: int = 0
    tracks_updated: int = 0
    tracks_skipped: list[tuple[Path, str]] = field(default_factory=list)

    def print_loops_summary(self) -> None:
        print(
            f"Лупы: добавлено {self.loops_added}, обновлено {self.loops_updated}, "
            f"пропущено {len(self.loops_skipped)}"
        )
        for path, reason in self.loops_skipped:
            print(f"  пропущен {path}: {reason}")

    def print_tracks_summary(self) -> None:
        print(
            f"Треки: добавлено {self.tracks_added}, обновлено {self.tracks_updated}, "
            f"пропущено {len(self.tracks_skipped)}"
        )
        for path, reason in self.tracks_skipped:
            print(f"  пропущен {path}: {reason}")

    def print_summary(self) -> None:
        self.print_loops_summary()
        self.print_tracks_summary()


def loop_skip_reason(result: ProbeResult, settings: Settings) -> str | None:
    """Проверить, годится ли пробированный файл в лупы"""
    if result.has_audio:
        return "у лупа есть аудиодорожка, ожидался немой файл"
    if not (settings.min_loop_seconds <= result.duration_seconds <= settings.max_loop_seconds):
        return (
            f"длительность {result.duration_seconds:.2f}с вне диапазона "
            f"[{settings.min_loop_seconds}, {settings.max_loop_seconds}]"
        )
    return None


def ingest_loop_file(
    conn: psycopg.Connection, path: Path, settings: Settings
) -> tuple[int | None, str | None]:
    """Провалидировать и заапсертить один луп. Вернуть (loop_id, причина_пропуска)."""
    try:
        result = probe(path)
    except FfmpegError as exc:
        return None, f"ffprobe не смог прочитать файл: {exc}"

    reason = loop_skip_reason(result, settings)
    if reason is not None:
        return None, reason

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


def parse_fma_track_id(path: Path) -> int | None:
    """Достать ID трека FMA из имени файла (например, 000002.mp3 -> 2)."""
    try:
        return int(path.stem)
    except ValueError:
        return None


def track_skip_reason(result: ProbeResult) -> str | None:
    """Проверить, годится ли пробированный файл в треки. None — годится."""
    if not result.has_audio:
        return "нет аудиодорожки"
    return None


def ingest_track_file(
    conn: psycopg.Connection,
    path: Path,
    metadata: dict,
    report: IngestReport,
) -> None:
    fma_track_id = parse_fma_track_id(path)
    if fma_track_id is None:
        report.tracks_skipped.append((path, "имя файла не похоже на ID трека FMA"))
        return

    try:
        result = probe(path)
    except FfmpegError as exc:
        report.tracks_skipped.append((path, f"ffprobe не смог прочитать файл: {exc}"))
        return

    reason = track_skip_reason(result)
    if reason is not None:
        report.tracks_skipped.append((path, reason))
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
    _track_id, inserted = row
    if inserted:
        report.tracks_added += 1
    else:
        report.tracks_updated += 1

    conn.commit()


def scan_tracks(conn: psycopg.Connection, settings: Settings, report: IngestReport) -> None:
    if not settings.music_dir.exists():
        logger.warning("директория с музыкой не найдена: %s", settings.music_dir)
        return

    metadata = load_metadata(settings.metadata_csv)

    for path in sorted(settings.music_dir.rglob("*.mp3")):
        ingest_track_file(conn, path, metadata, report)


def ingest(conn: psycopg.Connection, settings: Settings) -> IngestReport:
    report = IngestReport()
    scan_loops(conn, settings, report)
    scan_tracks(conn, settings, report)
    return report
