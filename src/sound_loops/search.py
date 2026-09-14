"""Поиск треков по текстовому описанию через эмбеддинг CLAP + pgvector.

Оценка похожести — косинусная (1 - оператор <=> из vector_cosine_ops,
тот же индекс, что построен в миграции 0002). Прослушивание — отдельным
шагом: экспорт топ-N в папку и, опционально, открытие системным плеером
(macOS `open`) — числа сами по себе не говорят, что модель считает
«грустным» или «эпичным», это можно только услышать.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from sound_loops.embeddings import Embedder, normalize


class SearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchResult:
    id: int
    path: str
    title: str | None
    artist: str | None
    genre: str | None
    similarity: float


def search_tracks(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    top_n: int,
) -> list[SearchResult]:
    register_vector(conn)
    vector = normalize(embedder.embed_texts([query]))[0]

    rows = conn.execute(
        """
        SELECT id, path, title, artist, genre, 1 - (embedding <=> %s) AS similarity
        FROM tracks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (vector, vector, top_n),
    ).fetchall()

    if not rows:
        raise SearchError("в базе нет проиндексированных треков — сначала запустите index")

    return [SearchResult(*row) for row in rows]


def export_results(results: list[SearchResult], export_dir: Path) -> list[Path]:
    """Скопировать найденные треки в отдельную папку, чтобы их можно было послушать."""
    export_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for rank, result in enumerate(results, start=1):
        src = Path(result.path)
        dest = export_dir / f"{rank:02d}_{result.similarity:.3f}_{src.name}"
        shutil.copyfile(src, dest)
        exported.append(dest)
    return exported


def open_with_player(paths: list[Path]) -> None:
    """Открыть файлы системным плеером через macOS `open`."""
    subprocess.run(["open", *[str(p) for p in paths]], check=True)
