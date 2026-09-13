from pathlib import Path

from sound_loops.config import Settings
from sound_loops.ffmpeg_utils import ProbeResult
from sound_loops.ingest import (
    IngestReport,
    ingest,
    ingest_loop_file,
    ingest_track_file,
    loop_skip_reason,
    parse_fma_track_id,
    track_skip_reason,
)
from sound_loops.metadata import TrackMetadata

SETTINGS = Settings(database_url="postgresql://localhost/unused", min_loop_seconds=3.0,
                     max_loop_seconds=10.0)


def _probe(duration=5.0, has_audio=False, has_video=True, width=640, height=360):
    return ProbeResult(
        duration_seconds=duration, width=width, height=height,
        has_video=has_video, has_audio=has_audio,
    )


def test_loop_with_audio_is_rejected():
    reason = loop_skip_reason(_probe(has_audio=True), SETTINGS)
    assert reason is not None
    assert "аудиодорожка" in reason


def test_loop_too_short_is_rejected():
    reason = loop_skip_reason(_probe(duration=2.9), SETTINGS)
    assert reason is not None
    assert "диапазона" in reason


def test_loop_too_long_is_rejected():
    reason = loop_skip_reason(_probe(duration=10.1), SETTINGS)
    assert reason is not None
    assert "диапазона" in reason


def test_loop_at_exact_bounds_is_accepted():
    assert loop_skip_reason(_probe(duration=3.0), SETTINGS) is None
    assert loop_skip_reason(_probe(duration=10.0), SETTINGS) is None


def test_valid_silent_loop_is_accepted():
    assert loop_skip_reason(_probe(duration=5.0, has_audio=False), SETTINGS) is None


def test_parse_fma_track_id_from_zero_padded_name():
    assert parse_fma_track_id(Path("data/raw/fma_small/000/000002.mp3")) == 2


def test_parse_fma_track_id_rejects_non_numeric_name():
    assert parse_fma_track_id(Path("data/raw/fma_small/000/not_a_track.mp3")) is None


def test_track_without_audio_is_rejected():
    reason = track_skip_reason(_probe(has_audio=False))
    assert reason is not None
    assert "аудиодорожки" in reason


def test_track_with_audio_is_accepted():
    assert track_skip_reason(_probe(has_audio=True)) is None


# --- Интеграционные тесты с настоящей базой (db_conn/db_settings из conftest.py) ---


def test_ingest_loop_file_inserts_new_row(db_conn, db_settings, get_silent_loop, tmp_path):
    path = get_silent_loop(tmp_path / "loop.mp4", 5.0)

    loop_id, note = ingest_loop_file(db_conn, path, db_settings)

    assert loop_id is not None
    assert note is None
    row = db_conn.execute("SELECT path FROM loops WHERE id = %s", (loop_id,)).fetchone()
    assert row == (str(path),)


def test_ingest_loop_file_is_idempotent_on_path(db_conn, db_settings, get_silent_loop, tmp_path):
    path = get_silent_loop(tmp_path / "loop.mp4", 5.0)

    first_id, first_note = ingest_loop_file(db_conn, path, db_settings)
    second_id, second_note = ingest_loop_file(db_conn, path, db_settings)

    assert first_id == second_id
    assert first_note is None
    assert second_note == "updated"
    count = db_conn.execute("SELECT count(*) FROM loops").fetchone()[0]
    assert count == 1


def test_ingest_loop_file_rejects_invalid_loop_without_inserting(
    db_conn, db_settings, get_silent_loop, tmp_path
):
    path = get_silent_loop(tmp_path / "too_long.mp4", 15.0)

    loop_id, note = ingest_loop_file(db_conn, path, db_settings)

    assert loop_id is None
    assert "диапазона" in note
    count = db_conn.execute("SELECT count(*) FROM loops").fetchone()[0]
    assert count == 0


def test_ingest_track_file_inserts_new_row_with_metadata(db_conn, get_tone_track, tmp_path):
    path = get_tone_track(tmp_path / "000042.mp3", 20.0)
    report = IngestReport()

    ingest_track_file(db_conn, path, metadata={42: TrackMetadata("T", "A", "G")}, report=report)

    assert report.tracks_added == 1
    row = db_conn.execute(
        "SELECT title, artist, genre FROM tracks WHERE fma_track_id = 42"
    ).fetchone()
    assert row == ("T", "A", "G")


def test_scan_and_ingest_is_idempotent_end_to_end(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path
):
    db_settings.loops_dir.mkdir(parents=True)
    db_settings.music_dir.mkdir(parents=True)
    get_silent_loop(db_settings.loops_dir / "a.mp4", 4.0)
    get_silent_loop(db_settings.loops_dir / "b.mp4", 6.0)
    get_tone_track(db_settings.music_dir / "000001.mp3", 20.0)

    first = ingest(db_conn, db_settings)
    assert first.loops_added == 2
    assert first.tracks_added == 1
    assert first.loops_skipped == []
    assert first.tracks_skipped == []

    second = ingest(db_conn, db_settings)
    assert second.loops_added == 0
    assert second.loops_updated == 2
    assert second.tracks_added == 0
    assert second.tracks_updated == 1

    loops_count = db_conn.execute("SELECT count(*) FROM loops").fetchone()[0]
    tracks_count = db_conn.execute("SELECT count(*) FROM tracks").fetchone()[0]
    assert loops_count == 2
    assert tracks_count == 1
