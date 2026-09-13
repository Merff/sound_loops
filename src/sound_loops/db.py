"""Подключение к Postgres, создание базы и схемы."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import psycopg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS loops (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    duration_seconds DOUBLE PRECISION NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tracks (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    duration_seconds DOUBLE PRECISION NOT NULL,
    fma_track_id INTEGER,
    title TEXT,
    artist TEXT,
    genre TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Координаты использованного отрезка (track_id, start_seconds) хранятся
-- прямо здесь, а не в отдельной таблице: каждый отрезок вырезается на
-- лету и используется ровно в одном рендере, отдельная таблица под
-- него была бы join на пустом месте.
CREATE TABLE IF NOT EXISTS renders (
    id SERIAL PRIMARY KEY,
    loop_id INTEGER NOT NULL REFERENCES loops(id),
    track_id INTEGER NOT NULL REFERENCES tracks(id),
    start_seconds DOUBLE PRECISION NOT NULL,
    output_path TEXT NOT NULL,
    duration_seconds DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Миграция со старой схемы, где у renders была ссылка на отдельную
-- таблицу track_segments. Идемпотентно: на новых базах renders уже
-- создаётся без track_segment_id, и блок ничего не делает.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'renders' AND column_name = 'track_segment_id'
    ) THEN
        ALTER TABLE renders ADD COLUMN IF NOT EXISTS track_id INTEGER REFERENCES tracks(id);
        ALTER TABLE renders ADD COLUMN IF NOT EXISTS start_seconds DOUBLE PRECISION;
        UPDATE renders r
        SET track_id = ts.track_id, start_seconds = ts.start_seconds
        FROM track_segments ts
        WHERE ts.id = r.track_segment_id AND r.track_id IS NULL;
        ALTER TABLE renders ALTER COLUMN track_id SET NOT NULL;
        ALTER TABLE renders ALTER COLUMN start_seconds SET NOT NULL;
        ALTER TABLE renders DROP COLUMN track_segment_id;
        DROP TABLE IF EXISTS track_segments;
    END IF;
END $$;
"""


def _server_url_and_dbname(database_url: str) -> tuple[str, str]:
    """Разбить URL на (строка подключения к серверу без конкретной базы, имя базы)."""
    parts = urlsplit(database_url)
    dbname = parts.path.lstrip("/")
    if not dbname:
        raise ValueError(f"в DATABASE_URL не указано имя базы: {database_url}")
    server_parts = parts._replace(path="/postgres")
    return urlunsplit(server_parts), dbname


def ensure_database_exists(database_url: str) -> None:
    """Создать базу, если её ещё нет. Подключается к обслуживающей базе postgres."""
    server_url, dbname = _server_url_and_dbname(database_url)
    with psycopg.connect(server_url, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{dbname}"')


def init_schema(database_url: str) -> None:
    """Создать схему в целевой базе. Идемпотентно — повторный запуск не падает."""
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(SCHEMA_SQL)


@contextmanager
def connect(database_url: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(database_url) as conn:
        yield conn
