import json
from pathlib import Path

import numpy as np
import pytest
from pgvector.psycopg import register_vector

from sound_loops.search import SearchError, SearchResult, export_results, search_tracks, search_tracks_hybrid


class FixedTextEmbedder:
    """embed_texts всегда возвращает один и тот же заранее заданный вектор."""

    model_id = "fixed-text-embedder"
    sample_rate = 8000

    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector.astype(np.float32)

    def embed_texts(self, texts):
        return np.tile(self._vector, (len(texts), 1))

    def embed_audio(self, waveforms):
        raise NotImplementedError


DIM = 512


def _basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _insert_track_with_embedding(conn, path: str, embedding: np.ndarray, model_id: str = "fixed-text-embedder"):
    register_vector(conn)
    conn.execute(
        """
        INSERT INTO tracks (path, duration_seconds, embedding, embedding_model)
        VALUES (%s, %s, %s, %s)
        """,
        (path, 5.0, embedding, model_id),
    )
    conn.commit()


def test_search_tracks_orders_results_by_similarity(db_conn):
    query = _basis(0)
    close = _basis(0)
    medium = _basis(0) + _basis(1)
    medium = medium / np.linalg.norm(medium)
    far = _basis(1)

    _insert_track_with_embedding(db_conn, "far.mp3", far)
    _insert_track_with_embedding(db_conn, "medium.mp3", medium)
    _insert_track_with_embedding(db_conn, "close.mp3", close)

    results = search_tracks(db_conn, FixedTextEmbedder(query), "irrelevant query text", top_n=10)

    assert [r.path for r in results] == ["close.mp3", "medium.mp3", "far.mp3"]
    assert results[0].similarity == pytest.approx(1.0, abs=1e-4)
    assert results[-1].similarity == pytest.approx(0.0, abs=1e-4)


def test_search_tracks_respects_top_n(db_conn):
    query = _basis(0)
    for i in range(5):
        _insert_track_with_embedding(db_conn, f"track_{i}.mp3", _basis(0))

    results = search_tracks(db_conn, FixedTextEmbedder(query), "query", top_n=2)

    assert len(results) == 2


def test_search_tracks_ignores_tracks_without_embedding(db_conn):
    db_conn.execute("INSERT INTO tracks (path, duration_seconds) VALUES ('no_embedding.mp3', 5.0)")
    db_conn.commit()

    with pytest.raises(SearchError):
        search_tracks(db_conn, FixedTextEmbedder(_basis(0)), "query", top_n=5)


def _insert_track_with_attrs(
    conn,
    path: str,
    embedding: np.ndarray,
    tempo_bpm: float | None,
    vocals_scores: dict[str, float] | None,
    model_id: str = "fixed-text-embedder",
):
    register_vector(conn)
    tags = {"vocals": vocals_scores} if vocals_scores is not None else None
    conn.execute(
        """
        INSERT INTO tracks (path, duration_seconds, embedding, embedding_model, tempo_bpm, tags, attrs_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (path, 5.0, embedding, model_id, tempo_bpm, json.dumps(tags) if tags else None, "v1" if tags else None),
    )
    conn.commit()


def test_search_tracks_hybrid_matches_with_no_relaxation(db_conn):
    query = _basis(0)
    _insert_track_with_attrs(db_conn, "in_range.mp3", query, 100.0, {"instrumental": 0.9, "with_vocals": 0.1})
    _insert_track_with_attrs(db_conn, "out_of_range.mp3", query, 300.0, {"instrumental": 0.9, "with_vocals": 0.1})

    results, applied = search_tracks_hybrid(
        db_conn, FixedTextEmbedder(query), "query", tempo_range=(90.0, 110.0), vocals="instrumental", top_n=10
    )

    assert [r.path for r in results] == ["in_range.mp3"]
    assert applied == []


def test_search_tracks_hybrid_widens_tempo_range_when_strict_is_empty(db_conn):
    query = _basis(0)
    # (90, 110) расширяется в 1.5 раза до (85, 115) — 112 снаружи строгого диапазона, внутри расширенного.
    _insert_track_with_attrs(db_conn, "just_outside.mp3", query, 112.0, {"instrumental": 0.9, "with_vocals": 0.1})

    results, applied = search_tracks_hybrid(
        db_conn, FixedTextEmbedder(query), "query", tempo_range=(90.0, 110.0), vocals="instrumental", top_n=10
    )

    assert [r.path for r in results] == ["just_outside.mp3"]
    assert applied == ["расширен диапазон темпа"]


def test_search_tracks_hybrid_drops_vocals_requirement_when_still_empty(db_conn):
    query = _basis(0)
    _insert_track_with_attrs(db_conn, "wrong_vocals.mp3", query, 100.0, {"instrumental": 0.1, "with_vocals": 0.9})

    results, applied = search_tracks_hybrid(
        db_conn, FixedTextEmbedder(query), "query", tempo_range=(90.0, 110.0), vocals="instrumental", top_n=10
    )

    assert [r.path for r in results] == ["wrong_vocals.mp3"]
    assert applied == ["расширен диапазон темпа", "снято требование по вокалу"]


def test_search_tracks_hybrid_falls_back_to_unfiltered_vector_search(db_conn):
    query = _basis(0)
    # Темп вне даже расширенного диапазона и не соответствует вокалу — найдётся
    # только на последней ступени, когда фильтры сняты полностью.
    _insert_track_with_attrs(db_conn, "no_attrs.mp3", query, None, None)

    results, applied = search_tracks_hybrid(
        db_conn, FixedTextEmbedder(query), "query", tempo_range=(90.0, 110.0), vocals="instrumental", top_n=10
    )

    assert [r.path for r in results] == ["no_attrs.mp3"]
    assert applied == ["расширен диапазон темпа", "снято требование по вокалу", "фильтры сняты полностью"]


def test_search_tracks_hybrid_raises_when_catalog_empty(db_conn):
    with pytest.raises(SearchError):
        search_tracks_hybrid(
            db_conn, FixedTextEmbedder(_basis(0)), "query", tempo_range=(90.0, 110.0), vocals="instrumental", top_n=10
        )


def test_export_results_copies_files_with_rank_prefix(tmp_path: Path):
    src_a = tmp_path / "a.mp3"
    src_a.write_bytes(b"fake mp3 a")
    src_b = tmp_path / "b.mp3"
    src_b.write_bytes(b"fake mp3 b")

    results = [
        SearchResult(id=1, path=str(src_a), title="A", artist=None, genre=None, similarity=0.9),
        SearchResult(id=2, path=str(src_b), title="B", artist=None, genre=None, similarity=0.5),
    ]

    export_dir = tmp_path / "export"
    exported = export_results(results, export_dir)

    assert len(exported) == 2
    assert exported[0].name == "01_0.900_a.mp3"
    assert exported[0].read_bytes() == b"fake mp3 a"
    assert exported[1].name == "02_0.500_b.mp3"
