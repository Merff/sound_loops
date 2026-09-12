"""Разбиение трека на отрезки по сетке фиксированной длины."""

from __future__ import annotations


def compute_segment_grid(
    duration_seconds: float,
    segment_seconds: float,
    min_segment_seconds: float = 1.0,
) -> list[tuple[float, float]]:
    """Построить сетку отрезков (start, duration) для трека заданной длины.

    - Трек короче одного отрезка сетки целиком становится одним отрезком.
    - Трек режется на отрезки длиной segment_seconds; хвост короче
      min_segment_seconds отбрасывается, иначе добавляется отдельным
      более коротким отрезком.
    """
    if duration_seconds <= 0 or segment_seconds <= 0:
        return []

    if duration_seconds <= segment_seconds:
        return [(0.0, duration_seconds)]

    segments: list[tuple[float, float]] = []
    start = 0.0
    while start + segment_seconds <= duration_seconds:
        segments.append((start, segment_seconds))
        start += segment_seconds

    remainder = duration_seconds - start
    if remainder >= min_segment_seconds:
        segments.append((start, remainder))

    return segments
