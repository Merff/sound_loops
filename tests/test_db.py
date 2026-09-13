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
    tables = {
        row[0]
        for row in db_conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    }
    assert {"loops", "tracks", "renders"} <= tables


def test_init_schema_is_idempotent(test_database_url):
    # test_database_url уже накатана фикстурой — повторный запуск не должен
    # падать и не должен повторно применять уже применённую миграцию.
    init_schema(test_database_url)
    init_schema(test_database_url)


def test_init_schema_tracks_applied_migrations(test_database_url, db_conn):
    applied = db_conn.execute("SELECT count(*) FROM _yoyo_migration").fetchone()[0]
    assert applied >= 1
