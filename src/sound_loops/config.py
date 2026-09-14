"""Конфигурация приложения. Загружается из .env и валидируется pydantic."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(
        ...,
        description="Строка подключения к Postgres, например postgresql://user@localhost:5432/sound_loops",
    )

    loops_dir: Path = Path("data/loops")
    music_dir: Path = Path("data/raw/fma_small")
    metadata_csv: Path = Path("data/raw/fma_metadata/tracks.csv")
    output_dir: Path = Path("data/renders")

    min_loop_seconds: float = 3.0
    max_loop_seconds: float = 10.0

    fade_seconds: float = 0.3

    clap_checkpoint: str = "laion/larger_clap_general"
    clap_device: str = "mps"
    embedding_batch_size: int = 8

    @field_validator("fade_seconds")
    @classmethod
    def _must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("должно быть больше нуля")
        return v

    @model_validator(mode="after")
    def _check_loop_bounds(self) -> Settings:
        if self.min_loop_seconds <= 0:
            raise ValueError("min_loop_seconds должно быть больше нуля")
        if self.max_loop_seconds <= self.min_loop_seconds:
            raise ValueError("max_loop_seconds должно быть больше min_loop_seconds")
        return self


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
