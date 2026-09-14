from pathlib import Path

from sound_loops.index import index_tracks, tracks_needing_embedding


def _insert_track(conn, path: Path, duration_seconds: float = 5.0) -> int:
    row = conn.execute(
        "INSERT INTO tracks (path, duration_seconds) VALUES (%s, %s) RETURNING id",
        (str(path), duration_seconds),
    ).fetchone()
    conn.commit()
    return row[0]


def test_index_tracks_embeds_tracks_without_embedding(db_conn, get_tone_track, tmp_path, fake_embedder):
    a = _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 3.0))
    b = _insert_track(db_conn, get_tone_track(tmp_path / "b.mp3", 3.0))

    report = index_tracks(db_conn, fake_embedder, batch_size=2)

    assert report.embedded == 2
    assert report.skipped == []
    rows = db_conn.execute(
        "SELECT id, embedding IS NOT NULL, embedding_model FROM tracks ORDER BY id"
    ).fetchall()
    assert rows == [(a, True, fake_embedder.model_id), (b, True, fake_embedder.model_id)]


def test_index_tracks_is_idempotent(db_conn, get_tone_track, tmp_path, fake_embedder):
    _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 3.0))

    first = index_tracks(db_conn, fake_embedder, batch_size=8)
    second = index_tracks(db_conn, fake_embedder, batch_size=8)

    assert first.embedded == 1
    assert second.embedded == 0


def test_index_tracks_reembeds_when_model_changes(db_conn, get_tone_track, tmp_path, fake_embedder):
    track_id = _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 3.0))
    index_tracks(db_conn, fake_embedder, batch_size=8)

    db_conn.execute("UPDATE tracks SET embedding_model = 'old-checkpoint' WHERE id = %s", (track_id,))
    db_conn.commit()

    report = index_tracks(db_conn, fake_embedder, batch_size=8)

    assert report.embedded == 1
    model = db_conn.execute("SELECT embedding_model FROM tracks WHERE id = %s", (track_id,)).fetchone()[0]
    assert model == fake_embedder.model_id


def test_index_tracks_skips_file_that_fails_to_decode(db_conn, get_tone_track, tmp_path, fake_embedder):
    ok_id = _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 3.0))
    broken_path = tmp_path / "missing.mp3"
    _insert_track(db_conn, broken_path)

    report = index_tracks(db_conn, fake_embedder, batch_size=8)

    assert report.embedded == 1
    assert len(report.skipped) == 1
    assert report.skipped[0][0] == str(broken_path)
    embedded = db_conn.execute("SELECT embedding IS NOT NULL FROM tracks WHERE id = %s", (ok_id,)).fetchone()[0]
    assert embedded


def test_tracks_needing_embedding_respects_limit(db_conn, get_tone_track, tmp_path, fake_embedder):
    _insert_track(db_conn, get_tone_track(tmp_path / "a.mp3", 3.0))
    _insert_track(db_conn, get_tone_track(tmp_path / "b.mp3", 3.0))

    rows = tracks_needing_embedding(db_conn, fake_embedder.model_id, limit=1)

    assert len(rows) == 1
