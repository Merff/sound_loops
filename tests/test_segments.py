import pytest

from sound_loops.segments import pick_random_start


def test_start_within_bounds_for_various_lengths():
    for _ in range(200):
        start = pick_random_start(track_duration_seconds=30.0, needed_seconds=7.0)
        assert 0.0 <= start <= 30.0 - 7.0


def test_track_exactly_as_long_as_needed_always_starts_at_zero():
    for _ in range(20):
        assert pick_random_start(track_duration_seconds=5.0, needed_seconds=5.0) == 0.0


def test_track_shorter_than_needed_raises():
    with pytest.raises(ValueError):
        pick_random_start(track_duration_seconds=4.0, needed_seconds=5.0)


def test_non_positive_needed_raises():
    with pytest.raises(ValueError):
        pick_random_start(track_duration_seconds=30.0, needed_seconds=0.0)
    with pytest.raises(ValueError):
        pick_random_start(track_duration_seconds=30.0, needed_seconds=-1.0)
