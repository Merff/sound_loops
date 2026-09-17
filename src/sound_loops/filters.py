"""Целевые атрибуты для гибридного поиска (итерация 4,
docs/sound_loops-iteration-4.md): диапазон темпа из motion лупа и лестница
послаблений фильтров.

Границы MOTION_TEMPO_RANGES изначально были взяты на глаз — прогон
eval-run показал, что это резко ухудшило retrieval (не найдено в топ-500
выросло 7->18/30): у 12 из 30 лупов собственный tempo_bpm размеченного
трека не попадал в диапазон, выведенный из motion его лупа. Перепроверка
по факту (сопоставление tempo_bpm good_tracks с ручной разметкой
tempo=slow/mid/fast в evals/dataset.json) показала, что человеческая
разметка темпа почти не разделяется по измеренному BPM (медианы
83-172/86-185/86-129 для slow/mid/fast — сильно перекрываются), так что
tempo — слабый сигнал на этой библиотеке в принципе, не только вопрос
неверных чисел. Границы ниже расширены до наименьших, что покрывают
27 из 30 реальных good_tracks (было 18/30) — три оставшихся выброса
(record BPM 172/185/86 у лупов с motion static/slow/chaotic) не покрыты
осознанно: расширять дальше означало бы фактически убрать фильтр.
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

# Порог "уверенности" тега вокала (search.py::_hybrid_where): у трека с
# |with_vocals - instrumental| меньше этого числа фильтр по вокалу его не
# трогает, даже если формально "не та" метка выше. Взято не на глаз — это
# примерно медиана |with_vocals - instrumental| по всей библиотеке (830
# треков, 0.112 на момент подбора): у половины треков зазор меньше, то
# есть эти теги про вокал у них по сути шумные, а не решение. Фильтр по
# вокалу без этого порога резал больше лупов, чем сам темп — из 10 лупов,
# ещё пропавших после расширения темпа, 7 оказались именно из-за вокала.
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
