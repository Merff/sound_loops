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
    max_loop_seconds: float = 11.0

    fade_seconds: float = 0.3

    clap_checkpoint: str = "laion/larger_clap_general"
    clap_device: str = "mps"
    embedding_batch_size: int = 8

    vlm_model: str = "qwen3-vl:4b-instruct"
    vlm_base_url: str = "http://localhost:11434"
    vlm_frame_count: int = 5
    vlm_frame_max_side: int = 448
    # Ollama по умолчанию режет контекст до 4096 токенов — с 5 кадрами по
    # 448px этого не хватает, поднимаем явно.
    vlm_context_length: int = 16384
    # None = температура Ollama по умолчанию. Команды эвала (eval-run,
    # blind-eval) всегда используют 0.0 сами, это только для analyze/match.
    vlm_temperature: float | None = None

    eval_dataset_path: Path = Path("evals/dataset.json")
    eval_runs_dir: Path = Path("evals/runs")
    eval_blind_runs_dir: Path = Path("evals/blind_runs")
    eval_search_depth: int = 500

    # Motion считается алгоритмически (motion.py), не через VLM.
    motion_sample_fps: float = 8.0
    motion_frame_size: int = 64

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
