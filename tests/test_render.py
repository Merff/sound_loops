"""Тесты render.py на настоящей тестовой базе (см. db_conn/db_settings в conftest.py)."""

from pathlib import Path

import pytest
from conftest import make_silent_loop, make_tone_track

from sound_loops.ffmpeg_utils import probe
from sound_loops.ingest import IngestReport, ingest_loop_file, ingest_track_file
from sound_loops.render import (
    RenderError,
    get_loop_by_path,
    get_random_loop,
    get_random_track,
    render_once,
)


def _insert_loop(conn, settings, tmp_path: Path, name: str, duration: float) -> int:
    path = make_silent_loop(tmp_path / name, duration_seconds=duration)
    loop_id, _note = ingest_loop_file(conn, path, settings)
    assert loop_id is not None
    return loop_id


def _insert_track(conn, tmp_path: Path, fma_track_id: int, duration: float) -> int:
    # Имя файла должно быть числовым — так ingest_track_file достаёт ID трека FMA.
    path = make_tone_track(tmp_path / f"{fma_track_id:06d}.mp3", duration_seconds=duration)
    report = IngestReport()
    ingest_track_file(conn, path, metadata={}, report=report)
    row = conn.execute("SELECT id FROM tracks WHERE path = %s", (str(path),)).fetchone()
    assert row is not None
    return row[0]


def test_get_random_loop_raises_when_table_empty(db_conn):
    with pytest.raises(RenderError, match="лупов"):
        get_random_loop(db_conn)


def test_get_random_track_raises_when_no_track_long_enough(db_conn, db_settings, tmp_path):
    _insert_track(db_conn, tmp_path, 1, duration=4.0)
    db_conn.commit()

    with pytest.raises(RenderError, match="треков"):
        get_random_track(db_conn, min_duration_seconds=10.0)


def test_get_random_track_filters_by_min_duration(db_conn, db_settings, tmp_path):
    short_id = _insert_track(db_conn, tmp_path, 1, duration=4.0)
    long_id = _insert_track(db_conn, tmp_path, 2, duration=20.0)
    db_conn.commit()

    for _ in range(10):
        track = get_random_track(db_conn, min_duration_seconds=10.0)
        assert track.id == long_id
        assert track.id != short_id


def test_get_random_loop_returns_inserted_loop(db_conn, db_settings, tmp_path):
    loop_id = _insert_loop(db_conn, db_settings, tmp_path, "loop.mp4", duration=5.0)
    db_conn.commit()

    loop = get_random_loop(db_conn)
    assert loop.id == loop_id
    assert loop.duration_seconds == pytest.approx(5.0, abs=0.2)


def test_get_loop_by_path_reuses_existing_row(db_conn, db_settings, tmp_path):
    path = make_silent_loop(tmp_path / "loop.mp4", duration_seconds=5.0)
    loop_id, _ = ingest_loop_file(db_conn, path, db_settings)
    db_conn.commit()

    loop = get_loop_by_path(db_conn, path, db_settings)
    assert loop.id == loop_id


def test_get_loop_by_path_ingests_unknown_loop_on_the_fly(db_conn, db_settings, tmp_path):
    path = make_silent_loop(tmp_path / "fresh.mp4", duration_seconds=6.0)

    loop = get_loop_by_path(db_conn, path, db_settings)

    assert loop.path == str(path)
    row = db_conn.execute("SELECT id FROM loops WHERE path = %s", (str(path),)).fetchone()
    assert row is not None and row[0] == loop.id


def test_get_loop_by_path_raises_for_invalid_loop(db_conn, db_settings, tmp_path):
    too_long = make_silent_loop(tmp_path / "too_long.mp4", duration_seconds=15.0)

    with pytest.raises(RenderError):
        get_loop_by_path(db_conn, too_long, db_settings)


def test_render_once_produces_valid_mp4_and_db_row(db_conn, db_settings, tmp_path):
    loop_id = _insert_loop(db_conn, db_settings, tmp_path, "loop.mp4", duration=5.0)
    track_id = _insert_track(db_conn, tmp_path, 3, duration=20.0)
    db_conn.commit()

    output_path = render_once(db_conn, db_settings)

    assert output_path.exists()
    result = probe(output_path)
    assert result.has_video
    assert result.has_audio
    assert result.duration_seconds == pytest.approx(5.0, abs=0.2)

    row = db_conn.execute(
        "SELECT loop_id, track_id, start_seconds, output_path, duration_seconds FROM renders"
    ).fetchone()
    assert row is not None
    got_loop_id, got_track_id, start_seconds, output_path_in_db, duration_seconds = row
    assert got_loop_id == loop_id
    assert got_track_id == track_id
    assert 0.0 <= start_seconds <= 20.0 - 5.0
    assert output_path_in_db == str(output_path)
    assert duration_seconds == pytest.approx(5.0, abs=0.2)


def test_render_ten_times_yields_one_render_row_each_time(db_conn, db_settings, tmp_path):
    """То же, что раньше проверялось вручную: 10 рендеров подряд, каждый — новая строка."""
    _insert_loop(db_conn, db_settings, tmp_path, "loop.mp4", duration=4.0)
    _insert_track(db_conn, tmp_path, 4, duration=25.0)
    db_conn.commit()

    for _ in range(10):
        output_path = render_once(db_conn, db_settings)
        result = probe(output_path)
        assert result.has_video and result.has_audio
        assert result.duration_seconds == pytest.approx(4.0, abs=0.2)

    renders = db_conn.execute("SELECT count(*) FROM renders").fetchone()[0]
    assert renders == 10
