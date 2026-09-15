"""Чистые функции метрик эвала (итерация 3, docs/sound_loops-iteration-3.md) —
без базы, без VLM, без ffmpeg. Тесты со заранее известным ответом в
tests/test_eval_metrics.py.
"""

from __future__ import annotations

from collections.abc import Sequence


def mood_overlap(predicted: Sequence[str], truth: Sequence[str]) -> float:
    """Пересечение множеств настроений (Jaccard): |A∩B| / |A∪B|.

    Оба множества пустыми не бывают (схема требует 1-3 значения), но пустое
    пересечение пустых множеств естественно читать как полное совпадение.
    """
    predicted_set, truth_set = set(predicted), set(truth)
    union = predicted_set | truth_set
    if not union:
        return 1.0
    return len(predicted_set & truth_set) / len(union)


def best_rank(ranked_paths: Sequence[str], good_paths: Sequence[str]) -> int | None:
    """1-based позиция первого из good_paths в ranked_paths, или None, если ни один не встретился."""
    good_set = set(good_paths)
    for rank, path in enumerate(ranked_paths, start=1):
        if path in good_set:
            return rank
    return None


def hit_at_k(ranked_paths: Sequence[str], good_paths: Sequence[str], k: int) -> bool:
    """Есть ли хотя бы один из good_paths среди первых k ranked_paths."""
    rank = best_rank(ranked_paths[:k], good_paths)
    return rank is not None
