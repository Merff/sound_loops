"""Поиск треков по текстовому описанию через эмбеддинг CLAP + pgvector.

Оценка похожести — косинусная (1 - оператор <=> из vector_cosine_ops,
тот же индекс, что построен в миграции 0002). Прослушивание — отдельным
шагом: экспорт топ-N в папку — числа сами по себе не говорят, что модель
считает «грустным» или «эпичным», это можно только услышать.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from sound_loops.embeddings import Embedder, normalize
from sound_loops.filters import (
    VOCALS_CONFIDENCE_MARGIN,
    FilterLevel,
    TempoRange,
    relaxation_ladder,
    run_relaxation_ladder,
)
from sound_loops.vlm import Vocals


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
    tempo_bpm: float | None = None
    tags: dict | None = None


def search_tracks(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    top_n: int,
    min_duration_seconds: float | None = None,
) -> list[SearchResult]:
    register_vector(conn)
    vector = normalize(embedder.embed_texts([query]))[0]

    duration_clause = "AND duration_seconds >= %s" if min_duration_seconds is not None else ""
    params = [vector]
    if min_duration_seconds is not None:
        params.append(min_duration_seconds)
    params += [vector, top_n]

    rows = conn.execute(
        f"""
        SELECT id, path, title, artist, genre, 1 - (embedding <=> %s) AS similarity
        FROM tracks
        WHERE embedding IS NOT NULL {duration_clause}
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        params,
    ).fetchall()

    if not rows:
        if min_duration_seconds is not None:
            raise SearchError(
                f"нет проиндексированных треков длиной от {min_duration_seconds:.1f}с"
            )
        raise SearchError("в базе нет проиндексированных треков — сначала запустите index")

    return [SearchResult(*row) for row in rows]


def _hybrid_where(
    level: FilterLevel, min_duration_seconds: float | None
) -> tuple[str, list]:
    clauses = ["embedding IS NOT NULL"]
    params: list = []
    if min_duration_seconds is not None:
        clauses.append("duration_seconds >= %s")
        params.append(min_duration_seconds)
    if level.tempo_range is not None:
        clauses.append("tempo_bpm BETWEEN %s AND %s")
        params += [level.tempo_range[0], level.tempo_range[1]]
    if level.vocals is not None:
        # Трек проходит, если его метка вокала уверенно совпадает с искомой,
        # ИЛИ если у него самого разница между метками мала (шумная
        # классификация — не исключаем по ней, см. VOCALS_CONFIDENCE_MARGIN).
        other = "with_vocals" if level.vocals == "instrumental" else "instrumental"
        clauses.append(
            "("
            "(tags -> 'vocals' ->> %s)::float >= (tags -> 'vocals' ->> %s)::float"
            " OR ABS((tags -> 'vocals' ->> 'with_vocals')::float - (tags -> 'vocals' ->> 'instrumental')::float) < %s"
            ")"
        )
        params += [level.vocals, other, VOCALS_CONFIDENCE_MARGIN]
    return " AND ".join(clauses), params


def search_tracks_hybrid(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    tempo_range: TempoRange,
    vocals: Vocals,
    top_n: int,
    min_duration_seconds: float | None = None,
) -> tuple[list[SearchResult], list[str]]:
    """SQL-фильтр по темпу/вокалу -> векторное ранжирование остатка, с
    лестницей послаблений (filters.py) — библиотека маленькая, жёсткие
    условия регулярно дают пустую выдачу. На нескольких сотнях треков
    точный перебор мгновенный, HNSW-индекс тут ничего не решает — он и не
    используется целиком (WHERE-фильтр по атрибутам не покрыт им никак,
    Postgres сам выбирает план).

    Возвращает (результат, применённые послабления — для диагностики).
    """
    register_vector(conn)
    vector = normalize(embedder.embed_texts([query]))[0]

    def attempt(level: FilterLevel) -> list[SearchResult]:
        where_sql, filter_params = _hybrid_where(level, min_duration_seconds)
        rows = conn.execute(
            f"""
            SELECT id, path, title, artist, genre, 1 - (embedding <=> %s) AS similarity, tempo_bpm, tags
            FROM tracks
            WHERE {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            [vector, *filter_params, vector, top_n],
        ).fetchall()
        return [SearchResult(*row) for row in rows]

    levels = relaxation_ladder(tempo_range, vocals)
    results, relaxed = run_relaxation_ladder(levels, attempt)

    if not results:
        raise SearchError("в базе нет проиндексированных треков — сначала запустите index")

    return results, relaxed


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
