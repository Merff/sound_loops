import pytest
from pydantic import ValidationError

from sound_loops.config import Settings


def test_defaults_are_valid():
    settings = Settings(database_url="postgresql://localhost/sound_loops")
    assert settings.min_loop_seconds == 3.0
    assert settings.max_loop_seconds == 10.0
    assert settings.fade_seconds == 0.3


def test_fade_seconds_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(database_url="postgresql://localhost/x", fade_seconds=0)


def test_max_loop_seconds_must_exceed_min():
    with pytest.raises(ValidationError):
        Settings(database_url="postgresql://localhost/x", min_loop_seconds=10, max_loop_seconds=5)


def test_min_loop_seconds_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(database_url="postgresql://localhost/x", min_loop_seconds=0)


def test_database_url_is_required(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
