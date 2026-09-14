"""Решить, включать ли HF_HUB_OFFLINE, не импортируя transformers/huggingface_hub.

Переменную нужно выставить до импорта transformers — если сделать это
позже (например, прямо перед from_pretrained), часть внутренних клиентов
huggingface_hub (замечено на версии с hf-xet) её всё равно не подхватывает
и продолжает бить по сети за файлами, которые уже лежат в кэше. Поэтому
проверка кэша сделана как отдельный модуль без тяжёлых зависимостей — CLI
вызывает её раньше, чем `from sound_loops.clap import ClapEmbedder`.
"""

from __future__ import annotations

import os
from pathlib import Path


def _hub_cache_dir() -> Path:
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if value := os.environ.get(var):
            return Path(value)
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def ensure_offline_if_cached(checkpoint: str) -> None:
    """Если чекпоинт уже скачан, выставить HF_HUB_OFFLINE=1 (не трогать явный выбор пользователя)."""
    if "HF_HUB_OFFLINE" in os.environ:
        return
    repo_dir = _hub_cache_dir() / f"models--{checkpoint.replace('/', '--')}"
    if any(repo_dir.glob("snapshots/*/config.json")):
        os.environ["HF_HUB_OFFLINE"] = "1"
