"""Индексация: посчитать эмбеддинги треков, которых ещё нет (или которые
посчитаны другой моделью), и записать в базу.

Коммит после каждого батча — если процесс прервать, уже записанные
эмбеддинги не теряются, а следующий запуск сам подберёт только то, что
осталось (WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM ...).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from sound_loops.embeddings import Embedder, normalize
from sound_loops.ffmpeg_utils import FfmpegError, decode_audio_mono

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    embedded: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"Эмбеддинги посчитаны для {self.embedded} треков, пропущено {len(self.skipped)}")
        for path, reason in self.skipped:
            print(f"  пропущен {path}: {reason}")


def _chunks(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def tracks_needing_embedding(
    conn: psycopg.Connection, model_id: str, limit: int | None = None
) -> list[tuple[int, str]]:
    """Треки без эмбеддинга или посчитанные другой моделью."""
    query = """
        SELECT id, path FROM tracks
        WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %s
        ORDER BY id
    """
    params: tuple = (model_id,)
    if limit is not None:
        query += " LIMIT %s"
        params = (model_id, limit)
    return conn.execute(query, params).fetchall()


def index_tracks(
    conn: psycopg.Connection,
    embedder: Embedder,
    batch_size: int,
    limit: int | None = None,
) -> IndexReport:
    register_vector(conn)
    report = IndexReport()
    rows = tracks_needing_embedding(conn, embedder.model_id, limit)
    total = len(rows)
    logger.info("нужно посчитать эмбеддинги для %d треков", total)

    for batch in _chunks(rows, batch_size):
        waveforms = []
        ok_ids = []
        for track_id, path in batch:
            try:
                waveforms.append(decode_audio_mono(Path(path), embedder.sample_rate))
                ok_ids.append(track_id)
            except FfmpegError as exc:
                report.skipped.append((path, f"ffmpeg не смог декодировать: {exc}"))

        if not ok_ids:
            continue

        vectors = normalize(embedder.embed_audio(waveforms))
        for track_id, vector in zip(ok_ids, vectors, strict=True):
            conn.execute(
                "UPDATE tracks SET embedding = %s, embedding_model = %s WHERE id = %s",
                (vector, embedder.model_id, track_id),
            )
        conn.commit()
        report.embedded += len(ok_ids)
        logger.info("посчитано %d/%d", report.embedded + len(report.skipped), total)

    return report
