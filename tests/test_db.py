import pytest

from sound_loops.db import _server_url_and_dbname, ensure_database_exists, init_schema


def test_splits_dbname_from_server_url():
    server_url, dbname = _server_url_and_dbname("postgresql://user@localhost:5432/sound_loops")
    assert dbname == "sound_loops"
    assert server_url == "postgresql://user@localhost:5432/postgres"


def test_works_without_explicit_user():
    server_url, dbname = _server_url_and_dbname("postgresql://localhost:5432/mydb")
    assert dbname == "mydb"
    assert server_url == "postgresql://localhost:5432/postgres"


def test_missing_dbname_raises():
    with pytest.raises(ValueError):
        _server_url_and_dbname("postgresql://localhost:5432/")


def test_ensure_database_exists_is_idempotent(test_database_url):
    # test_database_url уже создана фикстурой — повторный вызов не должен падать.
    ensure_database_exists(test_database_url)
    ensure_database_exists(test_database_url)


def test_init_schema_creates_all_tables(test_database_url, db_conn):
    init_schema(test_database_url)  # повторный запуск — не должен падать (DoD итерации 0)

    tables = {
        row[0]
        for row in db_conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    }
    assert {"loops", "tracks", "renders"} <= tables
    assert "track_segments" not in tables


def test_init_schema_migrates_old_track_segments_layout(test_database_url, db_conn):
    """До этой схемы координаты отрезка лежали в отдельной таблице
    track_segments, на которую renders ссылался через track_segment_id.
    init_schema должен перенести данные и убрать старые объекты, ничего
    не потеряв."""
    loop_id = db_conn.execute(
        "INSERT INTO loops (path, duration_seconds) VALUES ('l.mp4', 5) RETURNING id"
    ).fetchone()[0]
    track_id = db_conn.execute(
        "INSERT INTO tracks (path, duration_seconds) VALUES ('t.mp3', 30) RETURNING id"
    ).fetchone()[0]

    db_conn.execute("DROP TABLE renders")
    db_conn.execute(
        """
        CREATE TABLE track_segments (
            id SERIAL PRIMARY KEY,
            track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
            start_seconds DOUBLE PRECISION NOT NULL,
            duration_seconds DOUBLE PRECISION NOT NULL
        )
        """
    )
    segment_id = db_conn.execute(
        """
        INSERT INTO track_segments (track_id, start_seconds, duration_seconds)
        VALUES (%s, 12.5, 5) RETURNING id
        """,
        (track_id,),
    ).fetchone()[0]
    db_conn.execute(
        """
        CREATE TABLE renders (
            id SERIAL PRIMARY KEY,
            loop_id INTEGER NOT NULL REFERENCES loops(id),
            track_segment_id INTEGER NOT NULL REFERENCES track_segments(id),
            output_path TEXT NOT NULL,
            duration_seconds DOUBLE PRECISION NOT NULL
        )
        """
    )
    db_conn.execute(
        """
        INSERT INTO renders (loop_id, track_segment_id, output_path, duration_seconds)
        VALUES (%s, %s, 'out.mp4', 5)
        """,
        (loop_id, segment_id),
    )
    db_conn.commit()

    init_schema(test_database_url)

    row = db_conn.execute(
        "SELECT loop_id, track_id, start_seconds, duration_seconds FROM renders"
    ).fetchone()
    assert row == (loop_id, track_id, 12.5, 5.0)

    tables = {
        r[0]
        for r in db_conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    }
    assert "track_segments" not in tables
