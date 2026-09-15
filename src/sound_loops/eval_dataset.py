"""Разметка набора лупов для эвала (итерация 3, docs/sound_loops-iteration-3.md).

Хранится одним файлом в git (evals/dataset.json), не в базе — правится
вместе с кодом. Разметка делается руками (прослушивание/просмотр), этот
модуль только валидирует формат и даёт понятную ошибку на битую метку.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

from sound_loops.vlm import Mood, Setting

Tempo = Literal["slow", "mid", "fast"]
Vocals = Literal["instrumental", "vocals_ok", "any"]


class LoopAnnotation(BaseModel):
    loop: str = Field(description="Путь к лупу, как в таблице loops (например data/loops/loop-1.mp4).")
    setting: Setting
    tempo: Tempo
    mood: list[Mood] = Field(min_length=1, max_length=3)
    vocals: Vocals
    good_tracks: list[str] = Field(
        min_length=2,
        max_length=3,
        description="Пути к трекам (как в tracks.path), найденные вручную через `search`.",
    )


_DatasetAdapter = TypeAdapter(list[LoopAnnotation])


def load_dataset(path: Path) -> list[LoopAnnotation]:
    if not path.exists():
        raise FileNotFoundError(f"файл разметки не найден: {path}")
    return _DatasetAdapter.validate_json(path.read_text())


def save_dataset(path: Path, entries: list[LoopAnnotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _DatasetAdapter.dump_python(entries, mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
