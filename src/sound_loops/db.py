"""Подключение к Postgres: создание базы и применение миграций (yoyo)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from yoyo import get_backend, read_migrations

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _server_url_and_dbname(database_url: str) -> tuple[str, str]:
    parts = urlsplit(database_url)
    dbname = parts.path.lstrip("/")
    if not dbname:
        raise ValueError(f"в DATABASE_URL не указано имя базы: {database_url}")
    server_parts = parts._replace(path="/postgres")
    return urlunsplit(server_parts), dbname


def ensure_database_exists(database_url: str) -> None:
    server_url, dbname = _server_url_and_dbname(database_url)
    with psycopg.connect(server_url, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{dbname}"')


def _yoyo_url(database_url: str) -> str:
    """yoyo с psycopg3 требует схему postgresql+psycopg://, а не postgresql://."""
    parts = urlsplit(database_url)
    return urlunsplit(parts._replace(scheme="postgresql+psycopg"))


def init_schema(database_url: str) -> None:
    backend = get_backend(_yoyo_url(database_url))
    migrations = read_migrations(str(MIGRATIONS_DIR))
    with backend.lock():
        backend.apply_migrations(backend.to_apply(migrations))


@contextmanager
def connect(database_url: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(database_url) as conn:
        yield conn
