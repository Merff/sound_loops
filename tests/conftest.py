"""Общие фикстуры: синтетические видео-луп/трек и тестовая база.

Синтетические медиа генерируются на лету через lavfi-источники ffmpeg —
тесты не зависят от скачанного датасета FMA или реальных лупов.

Тесты работают с настоящим Postgres: используется отдельная база
`<основная_база>_test` в том же локальном инстансе, что и рабочая
(`DATABASE_URL` из `.env`). База и схема создаются автоматически при
первом запуске; перед каждым тестом, использующим db_conn, все таблицы
очищаются.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from sound_loops.config import Settings
from sound_loops.db import ensure_database_exists, init_schema
from sound_loops.ffmpeg_utils import run


def make_silent_loop(path: Path, duration_seconds: float) -> Path:
    """Немой mp4: testsrc, без аудиодорожки — как настоящий видео-луп."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration_seconds}:size=320x240:rate=25",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ]
    run(cmd)
    return path


def make_tone_track(path: Path, duration_seconds: float) -> Path:
    """mp3 с синусоидой — как отрезок настоящего музыкального трека."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_seconds}",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(path),
    ]
    run(cmd)
    return path


@pytest.fixture
def silent_loop(tmp_path: Path) -> Path:
    return make_silent_loop(tmp_path / "loop.mp4", duration_seconds=4.0)


@pytest.fixture
def tone_track(tmp_path: Path) -> Path:
    return make_tone_track(tmp_path / "track.mp3", duration_seconds=12.0)


def _derive_test_database_url() -> str:
    """Взять DATABASE_URL из .env / окружения и приписать _test к имени базы."""
    base_url = Settings().database_url
    parts = urlsplit(base_url)
    dbname = parts.path.lstrip("/")
    test_dbname = dbname if dbname.endswith("_test") else f"{dbname}_test"
    return urlunsplit(parts._replace(path=f"/{test_dbname}"))


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Отдельная тестовая база в том же Postgres. Создаётся и накатывается один раз за сессию."""
    url = _derive_test_database_url()
    ensure_database_exists(url)
    init_schema(url)
    return url


@pytest.fixture
def db_conn(test_database_url: str):
    """Соединение с чистой тестовой базой — таблицы очищены перед каждым тестом."""
    with psycopg.connect(test_database_url, autocommit=True) as setup_conn:
        setup_conn.execute(
            "TRUNCATE loops, tracks, track_segments, renders RESTART IDENTITY CASCADE"
        )

    with psycopg.connect(test_database_url) as conn:
        yield conn


@pytest.fixture
def db_settings(test_database_url: str, tmp_path: Path) -> Settings:
    """Settings, указывающие на тестовую базу и временные директории данных."""
    return Settings(
        database_url=test_database_url,
        loops_dir=tmp_path / "loops",
        music_dir=tmp_path / "raw" / "fma_small",
        metadata_csv=tmp_path / "raw" / "fma_metadata" / "tracks.csv",
        output_dir=tmp_path / "renders",
    )
