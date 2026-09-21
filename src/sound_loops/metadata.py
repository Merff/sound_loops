"""Чтение метаданных треков из tracks.csv датасета FMA.
У tracks.csv двухуровневый заголовок (как если бы его сохранили из
pandas.DataFrame с MultiIndex-колонками): первая строка — верхний уровень
("track", "artist", ...), вторая — нижний ("title", "name", "genre_top", ...).
Первая колонка — track_id, у неё оба уровня заголовка пустые.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_WANTED = {
    ("track", "title"): "title",
    ("artist", "name"): "artist",
    ("track", "genre_top"): "genre",
}


@dataclass(frozen=True)
class TrackMetadata:
    title: str = ""
    artist: str = ""
    genre: str = ""


def load_metadata(csv_path: Path) -> dict[int, TrackMetadata]:
    if not csv_path.exists():
        logger.warning("метаданные не найдены (%s) — поля треков будут пустыми", csv_path)
        return {}

    try:
        return _parse(csv_path)
    except Exception:
        logger.warning(
            "не удалось разобрать метаданные (%s) — поля треков будут пустыми", csv_path,
            exc_info=True,
        )
        return {}


def _parse(csv_path: Path) -> dict[int, TrackMetadata]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        top = next(reader)
        sub = next(reader)

        field_to_column: dict[str, int] = {}
        for i, (top_cell, sub_cell) in enumerate(zip(top, sub, strict=False)):
            key = (top_cell.strip().lower(), sub_cell.strip().lower())
            if key in _WANTED:
                field_to_column[_WANTED[key]] = i

        result: dict[int, TrackMetadata] = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                track_id = int(row[0])
            except ValueError:
                continue

            result[track_id] = TrackMetadata(
                title=_cell(row, field_to_column, "title"),
                artist=_cell(row, field_to_column, "artist"),
                genre=_cell(row, field_to_column, "genre"),
            )

    return result


def _cell(row: list[str], field_to_column: dict[str, int], field: str) -> str:
    col = field_to_column.get(field)
    if col is None or col >= len(row):
        return ""
    return row[col].strip()
