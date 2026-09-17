"""Целевые атрибуты для гибридного поиска (итерация 4,
docs/sound_loops-iteration-4.md): диапазон темпа из motion лупа и лестница
послаблений фильтров.

Границы MOTION_TEMPO_RANGES откалиброваны по tempo_bpm настоящих
good_tracks (evals/dataset.json), не взяты на глаз — ручная разметка
tempo=slow/mid/fast сама почти не разделяется по измеренному BPM, так
что диапазоны широкие и заметно перекрываются (см. README, раздел
«Гибридный поиск и переранжирование», за историей калибровки).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from sound_loops.vlm import Motion, Vocals

TempoRange = tuple[float, float]

MOTION_TEMPO_RANGES: dict[Motion, TempoRange] = {
    "static": (60.0, 130.0),
    "slow": (60.0, 140.0),
    "moderate": (70.0, 160.0),
    "fast": (80.0, 180.0),
    "chaotic": (90.0, 200.0),
}

# Множитель расширения диапазона темпа на первой ступени лестницы послаблений.
TEMPO_RELAXATION_FACTOR = 1.5

# Порог "уверенности" тега вокала (search.py::_hybrid_where): трек с
# |with_vocals - instrumental| меньше этого числа фильтр по вокалу не
# трогает, даже если формально "не та" метка выше — такой зазор шумный,
# не решение. Значение — медиана этого зазора по всей библиотеке.
VOCALS_CONFIDENCE_MARGIN = 0.11


def tempo_range_for_motion(motion: Motion) -> TempoRange:
    return MOTION_TEMPO_RANGES[motion]


def widen_tempo_range(tempo_range: TempoRange, factor: float = TEMPO_RELAXATION_FACTOR) -> TempoRange:
    """Раздвинуть диапазон темпа вокруг его середины в factor раз."""
    low, high = tempo_range
    center = (low + high) / 2
    half_span = (high - low) / 2 * factor
    return (max(0.0, center - half_span), center + half_span)


class FilterLevel:
    """Один уровень лестницы послаблений: диапазон темпа + требование по
    вокалу (None — фильтр снят). action — что было ослаблено по сравнению
    с предыдущим уровнем (None у самого строгого — это не послабление)."""

    __slots__ = ("tempo_range", "vocals", "action")

    def __init__(self, tempo_range: TempoRange | None, vocals: Vocals | None, action: str | None) -> None:
        self.tempo_range = tempo_range
        self.vocals = vocals
        self.action = action


def relaxation_ladder(tempo_range: TempoRange, vocals: Vocals) -> list[FilterLevel]:
    """Уровни от самого строгого к полному отсутствию фильтров — в этом
    порядке пробуются фильтры гибридного поиска (search.py)."""
    return [
        FilterLevel(tempo_range, vocals, None),
        FilterLevel(widen_tempo_range(tempo_range), vocals, "расширен диапазон темпа"),
        FilterLevel(widen_tempo_range(tempo_range), None, "снято требование по вокалу"),
        FilterLevel(None, None, "фильтры сняты полностью"),
    ]


def run_relaxation_ladder[T](
    levels: Sequence[FilterLevel], attempt: Callable[[FilterLevel], list[T]]
) -> tuple[list[T], list[str]]:
    """Пробовать уровни по порядку, пока attempt не вернёт непустой список.

    Возвращает (результат, применённые_послабления) — action всех уровней
    вплоть до успешного включительно. Последний уровень — без фильтров —
    гарантирует непустую выдачу, если она вообще есть в базе (это
    проверяет сам attempt/вызывающий код).
    """
    applied: list[str] = []
    for level in levels:
        if level.action is not None:
            applied.append(level.action)
        result = attempt(level)
        if result:
            return result, applied
    return [], applied
