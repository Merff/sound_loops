"""Команда tag-tracks (итерация 4, docs/sound_loops-iteration-4.md): темп +
zero-shot теги для треков без атрибутов текущей версии. Возобновляемая, как
index.py — коммит после каждого трека, прерванный запуск не теряет уже
посчитанное. Теги берутся из уже посчитанного tracks.embedding (без похода
в CLAP за аудио), темп требует отдельного decode через ffmpeg — CLAP тут
не участвует вообще.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from sound_loops.embeddings import Embedder
from sound_loops.ffmpeg_utils import FfmpegError, decode_audio_mono
from sound_loops.tags import TagPhraseBank
from sound_loops.tempo import estimate_tempo_bpm, normalize_tempo_octave

logger = logging.getLogger(__name__)

# Версия набора фраз/логики темпа — меняется, если правятся VOCAL_PHRASES/
# MOOD_PHRASES/GENRE_PHRASES в tags.py или границы в tempo.py, иначе в базе
# смешаются числа, посчитанные разными фразами.
ATTRS_VERSION = "v1"


@dataclass
class AttrsReport:
    computed: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"Атрибуты посчитаны для {self.computed} треков, пропущено {len(self.skipped)}")
        for path, reason in self.skipped:
            print(f"  пропущен {path}: {reason}")


def tracks_needing_attrs(conn: psycopg.Connection, version: str, limit: int | None = None) -> list[tuple]:
    """Треки с эмбеддингом (нужен для тегов), у которых атрибуты не текущей версии."""
    query = """
        SELECT id, path, embedding FROM tracks
        WHERE embedding IS NOT NULL AND (attrs_version IS NULL OR attrs_version IS DISTINCT FROM %s)
        ORDER BY id
    """
    params: tuple = (version,)
    if limit is not None:
        query += " LIMIT %s"
        params = (version, limit)
    return conn.execute(query, params).fetchall()


def compute_attrs(
    conn: psycopg.Connection,
    embedder: Embedder,
    tempo_sample_rate: int,
    tempo_max_seconds: float | None,
    limit: int | None = None,
) -> AttrsReport:
    register_vector(conn)
    report = AttrsReport()
    bank = TagPhraseBank(embedder)

    rows = tracks_needing_attrs(conn, ATTRS_VERSION, limit)
    total = len(rows)
    logger.info("нужно посчитать атрибуты для %d треков", total)

    for track_id, path, embedding in rows:
        try:
            waveform = decode_audio_mono(Path(path), tempo_sample_rate, tempo_max_seconds)
        except FfmpegError as exc:
            report.skipped.append((path, f"ffmpeg не смог декодировать: {exc}"))
            continue

        tempo_bpm = normalize_tempo_octave(estimate_tempo_bpm(waveform, tempo_sample_rate))
        tags = bank.tags_for(embedding.to_numpy().astype(np.float32))

        conn.execute(
            "UPDATE tracks SET tempo_bpm = %s, tags = %s, attrs_version = %s WHERE id = %s",
            (tempo_bpm, json.dumps(tags), ATTRS_VERSION, track_id),
        )
        conn.commit()
        report.computed += 1
        if report.computed % 50 == 0:
            logger.info("посчитано %d/%d", report.computed, total)

    return report
