"""Поиск треков по текстовому описанию через эмбеддинг CLAP + pgvector.

Оценка похожести — косинусная. Прослушивание — отдельным
шагом: экспорт топ-N в папку — числа сами по себе не говорят, что модель
считает «грустным» или «эпичным».
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from sound_loops.embeddings import Embedder, normalize
from sound_loops.vlm import Vocals

TempoRange = tuple[float, float]

# Порог "уверенности" тега вокала: трек с
# |with_vocals - instrumental| меньше этого числа фильтр по вокалу не
# трогает, значение — медиана этого зазора по всей библиотеке.
VOCALS_CONFIDENCE_MARGIN = 0.11

_COLUMNS = "id, path, title, artist, genre, 1 - (embedding <=> %s) AS similarity, tempo_bpm, tags"


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
        SELECT {_COLUMNS}
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


def _where(
    min_duration_seconds: float | None,
    tempo_range: TempoRange | None,
    vocals: Vocals | None,
    exclude_ids: Sequence[int] | None,
) -> tuple[str, list]:
    clauses = ["embedding IS NOT NULL"]
    params: list = []
    if min_duration_seconds is not None:
        clauses.append("duration_seconds >= %s")
        params.append(min_duration_seconds)
    if exclude_ids:
        clauses.append("id != ALL(%s)")
        params.append(list(exclude_ids))
    if tempo_range is not None:
        clauses.append("tempo_bpm BETWEEN %s AND %s")
        params += [tempo_range[0], tempo_range[1]]
    if vocals is not None:
        # Трек проходит, если его метка вокала уверенно совпадает с искомой,
        # ИЛИ если у него самого разница между метками мала (шумная
        # классификация — не исключаем по ней, см. VOCALS_CONFIDENCE_MARGIN).
        other = "with_vocals" if vocals == "instrumental" else "instrumental"
        clauses.append(
            "("
            "(tags -> 'vocals' ->> %s)::float >= (tags -> 'vocals' ->> %s)::float"
            " OR ABS((tags -> 'vocals' ->> 'with_vocals')::float - (tags -> 'vocals' ->> 'instrumental')::float) < %s"
            ")"
        )
        params += [vocals, other, VOCALS_CONFIDENCE_MARGIN]
    return " AND ".join(clauses), params


def search_tracks_filtered(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    top_n: int,
    min_duration_seconds: float | None = None,
    tempo_range: TempoRange | None = None,
    vocals: Vocals | None = None,
    exclude_ids: Sequence[int] | None = None,
) -> list[SearchResult]:
    """Один SQL-запрос с необязательными фильтрами по темпу/вокалу.
    Используется инструментом поиска узла plan (agent_planner.py): если
    фильтры дали пусто, это решает сам агент следующим вызовом
    инструмента, а не код автоматически. exclude_ids — треки,
    уже показанные в предыдущих кругах обратной связи (см. agent_graph.py).
    """
    register_vector(conn)
    vector = normalize(embedder.embed_texts([query]))[0]

    where_sql, filter_params = _where(min_duration_seconds, tempo_range, vocals, exclude_ids)
    rows = conn.execute(
        f"""
        SELECT {_COLUMNS}
        FROM tracks
        WHERE {where_sql}
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        [vector, *filter_params, vector, top_n],
    ).fetchall()
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
