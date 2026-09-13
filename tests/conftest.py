"""Общие фикстуры: синтетические видео-луп/трек и тестовая база.

Синтетические медиа генерируются через lavfi-источники ffmpeg — тесты не
зависят от скачанного датасета FMA или реальных лупов. Каждая уникальная
пара (тип, длительность) кодируется только один раз за сессию pytest и
кэшируется — тестам, которым нужен файл под конкретным именем/путём,
отдаётся дешёвая копия закэшированного файла, а не свежий ffmpeg-энкод
(см. get_silent_loop/get_tone_track).

Тесты работают с настоящим Postgres: используется отдельная база
`<основная_база>_test` в том же локальном инстансе, что и рабочая
(`DATABASE_URL` из `.env`). База и схема создаются автоматически при
первом запуске; перед каждым тестом, использующим db_conn, все таблицы
очищаются.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from sound_loops.config import Settings
from sound_loops.db import ensure_database_exists, init_schema
from sound_loops.ffmpeg_utils import run


def make_silent_loop(path: Path, duration_seconds: float) -> Path:
    """Немой mp4: testsrc, без аудиодорожки — как настоящий видео-луп.

    -preset ultrafast — это тестовая фикстура, не production-код: сам
    mux_loop_with_audio делает -c:v copy, так что кодек и скорость его
    энкода на итоговую логику не влияют, а на генерацию — заметно.
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration_seconds}:size=320x240:rate=25",
        "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
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


_MEDIA_CACHE: dict[tuple[str, float], Path] = {}


def _cached_media(kind: str, duration_seconds: float, cache_dir: Path, maker) -> Path:
    """Сгенерировать (kind, duration) один раз за сессию и вернуть путь к кэшу."""
    key = (kind, duration_seconds)
    cached = _MEDIA_CACHE.get(key)
    if cached is None:
        suffix = ".mp4" if kind == "loop" else ".mp3"
        cached = cache_dir / f"{kind}_{duration_seconds}{suffix}"
        maker(cached, duration_seconds)
        _MEDIA_CACHE[key] = cached
    return cached


@pytest.fixture(scope="session")
def _media_cache_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("media_cache")


@pytest.fixture
def get_silent_loop(_media_cache_dir: Path):
    """get_silent_loop(dest, duration) -> Path: копия закэшированного немого лупа."""

    def _get(dest: Path, duration_seconds: float) -> Path:
        cached = _cached_media("loop", duration_seconds, _media_cache_dir, make_silent_loop)
        shutil.copyfile(cached, dest)
        return dest

    return _get


@pytest.fixture
def get_tone_track(_media_cache_dir: Path):
    """get_tone_track(dest, duration) -> Path: копия закэшированного трека с тоном."""

    def _get(dest: Path, duration_seconds: float) -> Path:
        cached = _cached_media("track", duration_seconds, _media_cache_dir, make_tone_track)
        shutil.copyfile(cached, dest)
        return dest

    return _get


@pytest.fixture
def silent_loop(tmp_path: Path, get_silent_loop) -> Path:
    return get_silent_loop(tmp_path / "loop.mp4", 4.0)


@pytest.fixture
def tone_track(tmp_path: Path, get_tone_track) -> Path:
    return get_tone_track(tmp_path / "track.mp3", 12.0)


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
        setup_conn.execute("TRUNCATE loops, tracks, renders RESTART IDENTITY CASCADE")

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
