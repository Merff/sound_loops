"""Тесты render.py.

pick_random_start — чистая функция, без базы. Остальное — на настоящей
тестовой базе (см. db_conn/db_settings в conftest.py).
"""

from pathlib import Path

import psycopg
import pytest

from sound_loops.ffmpeg_utils import probe
from sound_loops.ingest import IngestReport, ingest_loop_file, ingest_track_file
from sound_loops.render import (
    RenderError,
    get_bad_rated_track_ids,
    get_loop_by_path,
    get_random_loop,
    get_random_track,
    pick_random_start,
    render_once,
    render_preview,
    set_render_rating,
)


def test_start_within_bounds_for_various_lengths():
    for _ in range(200):
        start = pick_random_start(track_duration_seconds=30.0, needed_seconds=7.0)
        assert 0.0 <= start <= 30.0 - 7.0


def test_track_exactly_as_long_as_needed_always_starts_at_zero():
    for _ in range(20):
        assert pick_random_start(track_duration_seconds=5.0, needed_seconds=5.0) == 0.0


def test_track_shorter_than_needed_raises():
    with pytest.raises(ValueError):
        pick_random_start(track_duration_seconds=4.0, needed_seconds=5.0)


def test_non_positive_needed_raises():
    with pytest.raises(ValueError):
        pick_random_start(track_duration_seconds=30.0, needed_seconds=0.0)
    with pytest.raises(ValueError):
        pick_random_start(track_duration_seconds=30.0, needed_seconds=-1.0)


def _insert_loop(
    conn, settings, get_silent_loop, tmp_path: Path, name: str, duration: float
) -> int:
    path = get_silent_loop(tmp_path / name, duration)
    loop_id, _note = ingest_loop_file(conn, path, settings)
    assert loop_id is not None
    return loop_id


def _insert_track(conn, get_tone_track, tmp_path: Path, fma_track_id: int, duration: float) -> int:
    # Имя файла должно быть числовым — так ingest_track_file достаёт ID трека FMA.
    path = get_tone_track(tmp_path / f"{fma_track_id:06d}.mp3", duration)
    report = IngestReport()
    ingest_track_file(conn, path, metadata={}, report=report)
    row = conn.execute("SELECT id FROM tracks WHERE path = %s", (str(path),)).fetchone()
    assert row is not None
    return row[0]


def test_get_random_loop_raises_when_table_empty(db_conn):
    with pytest.raises(RenderError, match="лупов"):
        get_random_loop(db_conn)


def test_get_random_track_raises_when_no_track_long_enough(
    db_conn, db_settings, get_tone_track, tmp_path
):
    _insert_track(db_conn, get_tone_track, tmp_path, 1, duration=4.0)
    db_conn.commit()

    with pytest.raises(RenderError, match="треков"):
        get_random_track(db_conn, min_duration_seconds=10.0)


def test_get_random_track_filters_by_min_duration(db_conn, db_settings, get_tone_track, tmp_path):
    short_id = _insert_track(db_conn, get_tone_track, tmp_path, 1, duration=4.0)
    long_id = _insert_track(db_conn, get_tone_track, tmp_path, 2, duration=20.0)
    db_conn.commit()

    for _ in range(10):
        track = get_random_track(db_conn, min_duration_seconds=10.0)
        assert track.id == long_id
        assert track.id != short_id


def test_get_random_loop_returns_inserted_loop(db_conn, db_settings, get_silent_loop, tmp_path):
    loop_id = _insert_loop(
        db_conn, db_settings, get_silent_loop, tmp_path, "loop.mp4", duration=5.0
    )
    db_conn.commit()

    loop = get_random_loop(db_conn)
    assert loop.id == loop_id
    assert loop.duration_seconds == pytest.approx(5.0, abs=0.2)


def test_get_loop_by_path_reuses_existing_row(db_conn, db_settings, get_silent_loop, tmp_path):
    path = get_silent_loop(tmp_path / "loop.mp4", 5.0)
    loop_id, _ = ingest_loop_file(db_conn, path, db_settings)
    db_conn.commit()

    loop = get_loop_by_path(db_conn, path, db_settings)
    assert loop.id == loop_id


def test_get_loop_by_path_ingests_unknown_loop_on_the_fly(
    db_conn, db_settings, get_silent_loop, tmp_path
):
    path = get_silent_loop(tmp_path / "fresh.mp4", 6.0)

    loop = get_loop_by_path(db_conn, path, db_settings)

    assert loop.path == str(path)
    row = db_conn.execute("SELECT id FROM loops WHERE path = %s", (str(path),)).fetchone()
    assert row is not None and row[0] == loop.id


def test_get_loop_by_path_raises_for_invalid_loop(db_conn, db_settings, get_silent_loop, tmp_path):
    too_long = get_silent_loop(tmp_path / "too_long.mp4", 15.0)

    with pytest.raises(RenderError):
        get_loop_by_path(db_conn, too_long, db_settings)


def test_render_once_produces_valid_mp4_and_db_row(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path
):
    loop_id = _insert_loop(
        db_conn, db_settings, get_silent_loop, tmp_path, "loop.mp4", duration=5.0
    )
    track_id = _insert_track(db_conn, get_tone_track, tmp_path, 3, duration=20.0)
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


def test_render_repeatedly_yields_one_render_row_each_time(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path
):
    """То же, что раньше проверялось вручную (там — 10 раз подряд): несколько
    рендеров подряд, каждый — новая строка, без дублей и сирот."""
    render_count = 3
    _insert_loop(db_conn, db_settings, get_silent_loop, tmp_path, "loop.mp4", duration=4.0)
    _insert_track(db_conn, get_tone_track, tmp_path, 4, duration=25.0)
    db_conn.commit()

    for _ in range(render_count):
        output_path = render_once(db_conn, db_settings)
        result = probe(output_path)
        assert result.has_video and result.has_audio
        assert result.duration_seconds == pytest.approx(4.0, abs=0.2)

    renders = db_conn.execute("SELECT count(*) FROM renders").fetchone()[0]
    assert renders == render_count


def _insert_loop_and_get_row(conn, settings, get_silent_loop, tmp_path, name, duration):
    loop_id = _insert_loop(conn, settings, get_silent_loop, tmp_path, name, duration)
    row = conn.execute("SELECT id, path, duration_seconds FROM loops WHERE id = %s", (loop_id,)).fetchone()
    from sound_loops.render import LoopRow

    return LoopRow(*row)


def _track_path(conn, track_id: int) -> str:
    return conn.execute("SELECT path FROM tracks WHERE id = %s", (track_id,)).fetchone()[0]


def test_render_preview_returns_render_id_alongside_path(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path
):
    loop = _insert_loop_and_get_row(db_conn, db_settings, get_silent_loop, tmp_path, "loop.mp4", 4.0)
    track_id = _insert_track(db_conn, get_tone_track, tmp_path, 5, duration=20.0)
    db_conn.commit()

    render_id, output_path = render_preview(db_conn, db_settings, loop, track_id, _track_path(db_conn, track_id), 20.0)

    assert output_path.exists()
    row = db_conn.execute("SELECT id FROM renders WHERE output_path = %s", (str(output_path),)).fetchone()
    assert row is not None and row[0] == render_id


def test_set_render_rating_updates_row(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path):
    _insert_loop(db_conn, db_settings, get_silent_loop, tmp_path, "loop.mp4", duration=4.0)
    _insert_track(db_conn, get_tone_track, tmp_path, 6, duration=20.0)
    db_conn.commit()
    output_path = render_once(db_conn, db_settings)
    render_id = db_conn.execute("SELECT id FROM renders WHERE output_path = %s", (str(output_path),)).fetchone()[0]

    set_render_rating(db_conn, render_id, "bad")

    rating = db_conn.execute("SELECT rating FROM renders WHERE id = %s", (render_id,)).fetchone()[0]
    assert rating == "bad"


def test_set_render_rating_rejects_invalid_value(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path):
    _insert_loop(db_conn, db_settings, get_silent_loop, tmp_path, "loop.mp4", duration=4.0)
    _insert_track(db_conn, get_tone_track, tmp_path, 7, duration=20.0)
    db_conn.commit()
    output_path = render_once(db_conn, db_settings)
    render_id = db_conn.execute("SELECT id FROM renders WHERE output_path = %s", (str(output_path),)).fetchone()[0]

    with pytest.raises(psycopg.errors.CheckViolation):
        set_render_rating(db_conn, render_id, "amazing")
    db_conn.rollback()


def test_get_bad_rated_track_ids_returns_only_bad_for_that_loop(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path
):
    loop_a = _insert_loop_and_get_row(db_conn, db_settings, get_silent_loop, tmp_path, "loop_a.mp4", 4.0)
    loop_b = _insert_loop_and_get_row(db_conn, db_settings, get_silent_loop, tmp_path, "loop_b.mp4", 4.0)
    track_bad_for_a = _insert_track(db_conn, get_tone_track, tmp_path, 8, duration=20.0)
    track_good = _insert_track(db_conn, get_tone_track, tmp_path, 9, duration=20.0)
    db_conn.commit()

    render_id_a_bad, _ = render_preview(
        db_conn, db_settings, loop_a, track_bad_for_a, _track_path(db_conn, track_bad_for_a), 20.0
    )
    render_id_a_good, _ = render_preview(
        db_conn, db_settings, loop_a, track_good, _track_path(db_conn, track_good), 20.0
    )
    render_id_b, _ = render_preview(
        db_conn, db_settings, loop_b, track_bad_for_a, _track_path(db_conn, track_bad_for_a), 20.0
    )

    set_render_rating(db_conn, render_id_a_bad, "bad")
    set_render_rating(db_conn, render_id_a_good, "good")
    set_render_rating(db_conn, render_id_b, "good")  # тот же трек, но для другого лупа — не должен исключаться там

    assert get_bad_rated_track_ids(db_conn, loop_a.id) == [track_bad_for_a]
    assert get_bad_rated_track_ids(db_conn, loop_b.id) == []
