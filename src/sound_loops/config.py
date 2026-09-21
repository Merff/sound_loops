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
    # ClapFeatureExtractor по умолчанию берёт случайные max_length_s секунд
    # из файла длиннее (truncation="rand_trunc") — эмбеддинг одного и того
    # же трека меняется от запуска к запуску. Чтобы этого не происходило,
    # сами не декодируем дальше этой границы, всегда с начала файла.
    clap_max_audio_seconds: float = 10.0

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

    # Атрибуты треков: темп через librosa. 22050 — стандартная
    # частота для beat-tracking (librosa сама ресемплит при необходимости).
    # max_seconds ограничивает decode/beat-tracking одного трека, а не
    # точность темпа — глобальный BPM устойчиво виден на минуте записи.
    tempo_sample_rate: int = 22050
    tempo_max_seconds: float = 60.0

    # Агент: узел plan вызывает инструмент поиска сам, пока не
    # соберёт agent_slot_count разных треков или не упрётся в потолок
    # ходов модели (agent_max_plan_iterations) — после него включается
    # резервный путь (см. agent_planner.py). agent_max_rounds — предел
    # кругов обратной связи, обязателен.
    agent_slot_count: int = 3
    agent_max_plan_iterations: int = 6
    agent_max_rounds: int = 3
    # Пол для top_n инструмента поиска (agent_planner.py) — независимо от
    # того, что запросила модель. Запас кандидатов для дедупа между слотами.
    # eval-run временно подменяет это значение на eval_search_depth (см. eval_run.py) —
    # иначе hit@k агента считался бы по пулу в разы меньше, чем у
    # остальных конфигураций, и был бы с ними несравним.
    agent_search_pool_size: int = 25

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
