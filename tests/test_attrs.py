from pathlib import Path

from sound_loops.attrs import ATTRS_VERSION, compute_attrs, tracks_needing_attrs
from sound_loops.index import index_tracks


def _insert_track(conn, path: Path, duration_seconds: float = 5.0) -> int:
    row = conn.execute(
        "INSERT INTO tracks (path, duration_seconds) VALUES (%s, %s) RETURNING id",
        (str(path), duration_seconds),
    ).fetchone()
    conn.commit()
    return row[0]


def test_compute_attrs_needs_embedding_first(db_conn, get_tone_track, tmp_path, fake_embedder):
    """Без index трек не попадает в выборку — не "пропущен", а просто не взят в работу."""
    _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 5.0))

    rows = tracks_needing_attrs(db_conn, ATTRS_VERSION)

    assert rows == []


def test_compute_attrs_writes_tempo_and_tags(db_conn, get_tone_track, tmp_path, fake_embedder):
    track_id = _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 5.0))
    index_tracks(db_conn, fake_embedder, batch_size=8)

    report = compute_attrs(db_conn, fake_embedder, tempo_sample_rate=22050, tempo_max_seconds=5.0)

    assert report.computed == 1
    assert report.skipped == []
    tempo_bpm, tags, attrs_version = db_conn.execute(
        "SELECT tempo_bpm, tags, attrs_version FROM tracks WHERE id = %s", (track_id,)
    ).fetchone()
    assert tempo_bpm is not None
    assert 60.0 <= tempo_bpm <= 200.0
    assert set(tags) == {"vocals", "mood", "genre"}
    assert attrs_version == ATTRS_VERSION


def test_compute_attrs_is_idempotent(db_conn, get_tone_track, tmp_path, fake_embedder):
    _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 5.0))
    index_tracks(db_conn, fake_embedder, batch_size=8)

    first = compute_attrs(db_conn, fake_embedder, tempo_sample_rate=22050, tempo_max_seconds=5.0)
    second = compute_attrs(db_conn, fake_embedder, tempo_sample_rate=22050, tempo_max_seconds=5.0)

    assert first.computed == 1
    assert second.computed == 0


def test_compute_attrs_recomputes_when_version_changes(db_conn, get_tone_track, tmp_path, fake_embedder):
    track_id = _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 5.0))
    index_tracks(db_conn, fake_embedder, batch_size=8)
    compute_attrs(db_conn, fake_embedder, tempo_sample_rate=22050, tempo_max_seconds=5.0)

    db_conn.execute("UPDATE tracks SET attrs_version = 'old-version' WHERE id = %s", (track_id,))
    db_conn.commit()

    report = compute_attrs(db_conn, fake_embedder, tempo_sample_rate=22050, tempo_max_seconds=5.0)

    assert report.computed == 1


def test_compute_attrs_skips_file_that_fails_to_decode(db_conn, get_tone_track, tmp_path, fake_embedder):
    ok_id = _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 5.0))
    broken_path = tmp_path / "missing.mp3"
    broken_id = _insert_track(db_conn, broken_path)
    # index_tracks сам бы пропустил broken_path как недекодируемый — подставим
    # эмбеддинг руками, чтобы он попал именно в выборку compute_attrs.
    index_tracks(db_conn, fake_embedder, batch_size=8)
    db_conn.execute(
        "UPDATE tracks SET embedding = (SELECT embedding FROM tracks WHERE id = %s), embedding_model = %s "
        "WHERE id = %s",
        (ok_id, fake_embedder.model_id, broken_id),
    )
    db_conn.commit()

    report = compute_attrs(db_conn, fake_embedder, tempo_sample_rate=22050, tempo_max_seconds=5.0)

    assert report.computed == 1
    assert len(report.skipped) == 1
    assert report.skipped[0][0] == str(broken_path)


def test_tracks_needing_attrs_respects_limit(db_conn, get_tone_track, tmp_path, fake_embedder):
    _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 5.0))
    _insert_track(db_conn, get_tone_track(tmp_path / "b.mp3", 5.0))
    index_tracks(db_conn, fake_embedder, batch_size=8)

    rows = tracks_needing_attrs(db_conn, ATTRS_VERSION, limit=1)

    assert len(rows) == 1
