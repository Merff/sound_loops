"""Команда eval-compare:
разница между двумя прогонами эвала — по агрегатам и по каждому лупу,
ухудшившиеся лупы первыми. Луп, которого нет в одном из прогонов
(новый/удалённый), не участвует в диффе метрик, а перечисляется отдельно.
"""

from __future__ import annotations

from pydantic import BaseModel

from sound_loops.eval_run import EvalRun, LoopEvalResult


class LoopDiff(BaseModel):
    loop: str
    mood_overlap_delta: float
    hit_at_1_delta: int  # -1/0/+1: было -> стало
    hit_at_5_delta: int
    best_rank_a: int | None
    best_rank_b: int | None
    score: float  # < 0 — стало хуже, > 0 — стало лучше


class CompareResult(BaseModel):
    only_in_a: list[str]
    only_in_b: list[str]
    diffs: list[LoopDiff]  # отсортированы: худшие изменения первыми
    aggregate_delta: dict[str, float]


def _rank_component(a_rank: int | None, b_rank: int | None, search_depth: int) -> float:
    """Вклад ранга в score: ниже ранг — лучше. Появление/исчезновение из выдачи — сильный сигнал."""
    if a_rank is None and b_rank is None:
        return 0.0
    if a_rank is None:  # не находили -> нашли
        return float(search_depth)
    if b_rank is None:  # находили -> перестали
        return -float(search_depth)
    return float(a_rank - b_rank)  # ранг уменьшился = стало лучше = положительный вклад


def _diff_loop(a: LoopEvalResult, b: LoopEvalResult, search_depth: int) -> LoopDiff:
    mood_delta = b.mood_overlap - a.mood_overlap
    hit1_delta = int(b.hit_at_1) - int(a.hit_at_1)
    hit5_delta = int(b.hit_at_5) - int(a.hit_at_5)
    rank_component = _rank_component(a.best_rank, b.best_rank, search_depth)
    score = hit1_delta * 3 + hit5_delta * 2 + mood_delta + rank_component * 0.02
    return LoopDiff(
        loop=a.loop,
        mood_overlap_delta=mood_delta,
        hit_at_1_delta=hit1_delta,
        hit_at_5_delta=hit5_delta,
        best_rank_a=a.best_rank,
        best_rank_b=b.best_rank,
        score=score,
    )


def compare_runs(a: EvalRun, b: EvalRun) -> CompareResult:
    a_by_loop = {r.loop: r for r in a.loops}
    b_by_loop = {r.loop: r for r in b.loops}

    common = sorted(a_by_loop.keys() & b_by_loop.keys())
    only_in_a = sorted(a_by_loop.keys() - b_by_loop.keys())
    only_in_b = sorted(b_by_loop.keys() - a_by_loop.keys())

    search_depth = max(a.search_depth, b.search_depth)
    diffs = [_diff_loop(a_by_loop[loop], b_by_loop[loop], search_depth) for loop in common]
    diffs.sort(key=lambda d: d.score)

    aggregate_delta = {
        "setting_accuracy": b.aggregates.setting_accuracy - a.aggregates.setting_accuracy,
        "mean_mood_overlap": b.aggregates.mean_mood_overlap - a.aggregates.mean_mood_overlap,
        "hit_at_1_rate": b.aggregates.hit_at_1_rate - a.aggregates.hit_at_1_rate,
        "hit_at_5_rate": b.aggregates.hit_at_5_rate - a.aggregates.hit_at_5_rate,
        "mean_penalized_rank": b.aggregates.mean_penalized_rank - a.aggregates.mean_penalized_rank,
    }

    return CompareResult(only_in_a=only_in_a, only_in_b=only_in_b, diffs=diffs, aggregate_delta=aggregate_delta)


def print_compare(result: CompareResult) -> None:
    print("Изменение агрегатов (b - a):")
    for name, delta in result.aggregate_delta.items():
        print(f"  {name}: {delta:+.2f}")

    if result.only_in_a:
        print(f"\nТолько в прогоне a (пропущены в сравнении): {', '.join(result.only_in_a)}")
    if result.only_in_b:
        print(f"Только в прогоне b (пропущены в сравнении): {', '.join(result.only_in_b)}")

    print("\nПо лупам (худшие изменения первыми):")
    for d in result.diffs:
        print(
            f"  {d.loop}: mood {d.mood_overlap_delta:+.2f}  hit@1 {d.hit_at_1_delta:+d}  "
            f"hit@5 {d.hit_at_5_delta:+d}  rank {d.best_rank_a} -> {d.best_rank_b}"
        )
