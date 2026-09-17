"""Целевой диапазон темпа и лестница послаблений — чистые функции, без базы."""

from sound_loops.filters import (
    MOTION_TEMPO_RANGES,
    relaxation_ladder,
    run_relaxation_ladder,
    tempo_range_for_motion,
    widen_tempo_range,
)


def test_tempo_range_for_motion_covers_every_motion_category():
    for motion in ("static", "slow", "moderate", "fast", "chaotic"):
        low, high = tempo_range_for_motion(motion)
        assert low < high

    assert set(MOTION_TEMPO_RANGES) == {"static", "slow", "moderate", "fast", "chaotic"}


def test_tempo_ranges_shift_upward_from_static_to_chaotic():
    ranges = [tempo_range_for_motion(m) for m in ("static", "slow", "moderate", "fast", "chaotic")]
    midpoints = [(low + high) / 2 for low, high in ranges]
    assert midpoints == sorted(midpoints)


def test_widen_tempo_range_grows_the_span_around_the_same_center():
    low, high = (90.0, 110.0)
    widened_low, widened_high = widen_tempo_range((low, high), factor=2.0)
    assert widened_high - widened_low == (high - low) * 2.0
    assert (widened_low + widened_high) / 2 == (low + high) / 2


def test_widen_tempo_range_does_not_go_below_zero():
    assert widen_tempo_range((1.0, 3.0), factor=10.0)[0] == 0.0


def test_relaxation_ladder_order():
    levels = relaxation_ladder((90.0, 110.0), "instrumental")
    actions = [level.action for level in levels]
    assert actions == [None, "расширен диапазон темпа", "снято требование по вокалу", "фильтры сняты полностью"]


def test_run_relaxation_ladder_returns_first_non_empty_without_relaxing():
    levels = relaxation_ladder((90.0, 110.0), "instrumental")

    result, applied = run_relaxation_ladder(levels, lambda level: [1, 2, 3])

    assert result == [1, 2, 3]
    assert applied == []


def test_run_relaxation_ladder_tries_levels_in_order_and_reports_relaxations():
    levels = relaxation_ladder((90.0, 110.0), "instrumental")
    calls = []

    def attempt(level):
        calls.append(level.action)
        return [42] if level.action == "снято требование по вокалу" else []

    result, applied = run_relaxation_ladder(levels, attempt)

    assert result == [42]
    assert applied == ["расширен диапазон темпа", "снято требование по вокалу"]
    assert calls == [None, "расширен диапазон темпа", "снято требование по вокалу"]


def test_run_relaxation_ladder_returns_empty_when_even_unfiltered_is_empty():
    levels = relaxation_ladder((90.0, 110.0), "instrumental")

    result, applied = run_relaxation_ladder(levels, lambda level: [])

    assert result == []
    assert applied == ["расширен диапазон темпа", "снято требование по вокалу", "фильтры сняты полностью"]
