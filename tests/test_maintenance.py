from pathlib import Path

from sound_loops.maintenance import cleanup_session_renders, clear_analyses, clear_renders


def _insert_loop(conn, path: str = "loop.mp4", duration_seconds: float = 5.0) -> int:
    row = conn.execute(
        "INSERT INTO loops (path, duration_seconds) VALUES (%s, %s) RETURNING id",
        (path, duration_seconds),
    ).fetchone()
    conn.commit()
    return row[0]


def _insert_track(conn, path: str = "track.mp3", duration_seconds: float = 5.0) -> int:
    row = conn.execute(
        "INSERT INTO tracks (path, duration_seconds) VALUES (%s, %s) RETURNING id",
        (path, duration_seconds),
    ).fetchone()
    conn.commit()
    return row[0]


def _insert_analysis(conn, loop_id: int, model: str = "model-a", prompt_version: str = "v1") -> int:
    row = conn.execute(
        """
        INSERT INTO video_analyses (loop_id, model, prompt_version, setting, motion, mood)
        VALUES (%s, %s, %s, 'domestic', 'slow', %s)
        RETURNING id
        """,
        (loop_id, model, prompt_version, ["calm"]),
    ).fetchone()
    conn.commit()
    return row[0]


def _insert_render(conn, loop_id: int, track_id: int, output_path: Path, analysis_id: int | None = None) -> int:
    row = conn.execute(
        """
        INSERT INTO renders (loop_id, track_id, start_seconds, output_path, duration_seconds, analysis_id)
        VALUES (%s, %s, 0, %s, 5.0, %s)
        RETURNING id
        """,
        (loop_id, track_id, str(output_path), analysis_id),
    ).fetchone()
    conn.commit()
    return row[0]


def test_clear_renders_deletes_rows_and_files(db_conn, tmp_path):
    loop_id = _insert_loop(db_conn)
    track_id = _insert_track(db_conn)
    output = tmp_path / "render.mp4"
    output.write_bytes(b"fake mp4")
    _insert_render(db_conn, loop_id, track_id, output)

    report = clear_renders(db_conn)

    assert report.renders_deleted == 1
    assert report.files_deleted == 1
    assert report.files_missing == 0
    assert not output.exists()
    assert db_conn.execute("SELECT count(*) FROM renders").fetchone()[0] == 0


def test_clear_renders_counts_missing_files_without_raising(db_conn, tmp_path):
    loop_id = _insert_loop(db_conn)
    track_id = _insert_track(db_conn)
    missing_output = tmp_path / "already_gone.mp4"
    _insert_render(db_conn, loop_id, track_id, missing_output)

    report = clear_renders(db_conn)

    assert report.renders_deleted == 1
    assert report.files_deleted == 0
    assert report.files_missing == 1


def test_clear_renders_does_not_touch_video_analyses(db_conn, tmp_path):
    loop_id = _insert_loop(db_conn)
    track_id = _insert_track(db_conn)
    analysis_id = _insert_analysis(db_conn, loop_id)
    output = tmp_path / "render.mp4"
    output.write_bytes(b"fake mp4")
    _insert_render(db_conn, loop_id, track_id, output, analysis_id=analysis_id)

    clear_renders(db_conn)

    assert db_conn.execute("SELECT count(*) FROM video_analyses").fetchone()[0] == 1


def test_clear_analyses_deletes_analyses_and_dependent_renders_only(db_conn, tmp_path):
    loop_id = _insert_loop(db_conn)
    track_id = _insert_track(db_conn)
    analysis_id = _insert_analysis(db_conn, loop_id)

    match_output = tmp_path / "match_render.mp4"
    match_output.write_bytes(b"fake mp4")
    _insert_render(db_conn, loop_id, track_id, match_output, analysis_id=analysis_id)

    baseline_output = tmp_path / "baseline_render.mp4"
    baseline_output.write_bytes(b"fake mp4")
    _insert_render(db_conn, loop_id, track_id, baseline_output, analysis_id=None)

    report = clear_analyses(db_conn)

    assert report.analyses_deleted == 1
    assert report.dependent_renders_deleted == 1
    assert report.dependent_files_deleted == 1
    assert not match_output.exists()

    # baseline-рендер (без analysis_id) не трогаем
    assert baseline_output.exists()
    remaining = db_conn.execute("SELECT output_path FROM renders").fetchall()
    assert remaining == [(str(baseline_output),)]
    assert db_conn.execute("SELECT count(*) FROM video_analyses").fetchone()[0] == 0


def test_clear_analyses_with_no_data_is_a_noop(db_conn):
    report = clear_analyses(db_conn)

    assert report.analyses_deleted == 0
    assert report.dependent_renders_deleted == 0


def _insert_rated_render(conn, loop_id: int, track_id: int, output_path: Path, rating: str | None) -> int:
    render_id = _insert_render(conn, loop_id, track_id, output_path)
    if rating is not None:
        conn.execute("UPDATE renders SET rating = %s WHERE id = %s", (rating, render_id))
        conn.commit()
    return render_id


def test_cleanup_session_renders_keeps_good_deletes_unrated_and_neutral(db_conn, tmp_path):
    loop_id = _insert_loop(db_conn)
    track_id = _insert_track(db_conn)

    good_output = tmp_path / "good.mp4"
    good_output.write_bytes(b"fake mp4")
    good_id = _insert_rated_render(db_conn, loop_id, track_id, good_output, "good")

    neutral_output = tmp_path / "neutral.mp4"
    neutral_output.write_bytes(b"fake mp4")
    neutral_id = _insert_rated_render(db_conn, loop_id, track_id, neutral_output, "neutral")

    unrated_output = tmp_path / "unrated.mp4"
    unrated_output.write_bytes(b"fake mp4")
    unrated_id = _insert_rated_render(db_conn, loop_id, track_id, unrated_output, None)

    report = cleanup_session_renders(db_conn, [good_id, neutral_id, unrated_id])

    assert report.good_kept == 1
    assert report.other_deleted == 2
    assert good_output.exists()
    assert not neutral_output.exists()
    assert not unrated_output.exists()
    remaining_ids = {row[0] for row in db_conn.execute("SELECT id FROM renders").fetchall()}
    assert remaining_ids == {good_id}


def test_cleanup_session_renders_deletes_bad_file_but_keeps_row(db_conn, tmp_path):
    loop_id = _insert_loop(db_conn)
    track_id = _insert_track(db_conn)

    bad_output = tmp_path / "bad.mp4"
    bad_output.write_bytes(b"fake mp4")
    bad_id = _insert_rated_render(db_conn, loop_id, track_id, bad_output, "bad")

    report = cleanup_session_renders(db_conn, [bad_id])

    assert report.bad_kept_files_deleted == 1
    assert not bad_output.exists()
    row = db_conn.execute("SELECT rating FROM renders WHERE id = %s", (bad_id,)).fetchone()
    assert row == ("bad",)


def test_cleanup_session_renders_with_empty_ids_is_a_noop(db_conn):
    report = cleanup_session_renders(db_conn, [])

    assert report.good_kept == 0
    assert report.other_deleted == 0
