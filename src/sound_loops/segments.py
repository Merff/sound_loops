"""Выбор случайной позиции отрезка внутри трека нужной длины."""

from __future__ import annotations

import random


def pick_random_start(track_duration_seconds: float, needed_seconds: float) -> float:
    """Случайная точка начала внутри трека, чтобы после неё хватило needed_seconds.

    Трек должен быть не короче needed_seconds — это проверяется заранее
    при выборе кандидата (render.get_random_track), здесь только защита
    от неверного использования.
    """
    if needed_seconds <= 0:
        raise ValueError("needed_seconds должно быть больше нуля")
    if track_duration_seconds < needed_seconds:
        raise ValueError(
            f"трек короче нужного отрезка: {track_duration_seconds} < {needed_seconds}"
        )

    max_start = track_duration_seconds - needed_seconds
    if max_start <= 0:
        return 0.0
    return random.uniform(0, max_start)
