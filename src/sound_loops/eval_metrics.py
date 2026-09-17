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


def penalized_rank(rank: int | None, search_depth: int) -> int:
    """rank, или search_depth, если трек не найден.

    Просто усреднять best_rank по найденным лупам (как делает
    mean_best_rank) нельзя сравнивать между конфигурациями с разным
    числом "не найдено" — выброс ненайденных из среднего искажает
    сравнение в пользу той конфигурации, что больше отбросила (найдено
    в итерации 4, docs/sound_loops-iteration-4.md, при сравнении фильтров
    до/после калибровки промпта шага B). search_depth как штраф — то же
    приближение, что eval_compare.py::_rank_component уже использует для
    попарных дельт.
    """
    return rank if rank is not None else search_depth
