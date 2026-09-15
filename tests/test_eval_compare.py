from sound_loops.eval_compare import compare_runs
from sound_loops.eval_run import EvalAggregates, EvalRun, LoopEvalResult


def _loop_result(loop: str, *, mood: float, hit1: bool, hit5: bool, rank: int | None) -> LoopEvalResult:
    return LoopEvalResult(
        loop=loop,
        setting_correct=True,
        predicted_setting="nature",
        true_setting="nature",
        mood_overlap=mood,
        predicted_mood=["calm"],
        true_mood=["calm"],
        music_query="slow calm ambient instrumental",
        hit_at_1=hit1,
        hit_at_5=hit5,
        best_rank=rank,
    )


def _run(loops: list[LoopEvalResult], search_depth: int = 100) -> EvalRun:
    n = len(loops)
    ranks = [r.best_rank for r in loops if r.best_rank is not None]
    aggregates = EvalAggregates(
        setting_accuracy=1.0,
        mean_mood_overlap=sum(r.mood_overlap for r in loops) / n if n else 0.0,
        hit_at_1_rate=sum(r.hit_at_1 for r in loops) / n if n else 0.0,
        hit_at_5_rate=sum(r.hit_at_5 for r in loops) / n if n else 0.0,
        mean_best_rank=(sum(ranks) / len(ranks)) if ranks else None,
        not_found_count=n - len(ranks),
    )
    return EvalRun(
        timestamp="2026-01-01T00:00:00+00:00",
        commit="abc123",
        model="m",
        prompt_version="v1",
        temperature=0.0,
        search_depth=search_depth,
        loops=loops,
        aggregates=aggregates,
    )


def test_regressed_loop_sorted_first():
    a = _run(
        [
            _loop_result("improved.mp4", mood=0.5, hit1=False, hit5=True, rank=4),
            _loop_result("regressed.mp4", mood=0.8, hit1=True, hit5=True, rank=1),
        ]
    )
    b = _run(
        [
            _loop_result("improved.mp4", mood=0.8, hit1=True, hit5=True, rank=1),
            _loop_result("regressed.mp4", mood=0.2, hit1=False, hit5=False, rank=None),
        ]
    )

    result = compare_runs(a, b)

    assert [d.loop for d in result.diffs] == ["regressed.mp4", "improved.mp4"]
    assert result.diffs[0].hit_at_1_delta == -1
    assert result.diffs[0].best_rank_b is None


def test_loop_present_in_only_one_run_is_listed_separately():
    a = _run([_loop_result("only_a.mp4", mood=0.5, hit1=True, hit5=True, rank=1)])
    b = _run([_loop_result("only_b.mp4", mood=0.5, hit1=True, hit5=True, rank=1)])

    result = compare_runs(a, b)

    assert result.diffs == []
    assert result.only_in_a == ["only_a.mp4"]
    assert result.only_in_b == ["only_b.mp4"]


def test_aggregate_delta_is_b_minus_a():
    a = _run([_loop_result("x.mp4", mood=0.4, hit1=False, hit5=False, rank=None)])
    b = _run([_loop_result("x.mp4", mood=0.9, hit1=True, hit5=True, rank=1)])

    result = compare_runs(a, b)

    assert result.aggregate_delta["hit_at_1_rate"] == 1.0
    assert result.aggregate_delta["mean_mood_overlap"] == 0.5
