from pathlib import Path

from sound_loops.maintenance import clear_analyses, clear_renders


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
        INSERT INTO video_analyses (loop_id, model, prompt_version, summary, motion, mood, is_comic)
        VALUES (%s, %s, %s, 'a test scene', 'slow', %s, false)
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
