from sound_loops.segments import compute_segment_grid


def test_exact_multiple_of_grid():
    segments = compute_segment_grid(duration_seconds=30.0, segment_seconds=10.0)
    assert segments == [(0.0, 10.0), (10.0, 10.0), (20.0, 10.0)]


def test_remainder_kept_when_above_threshold():
    segments = compute_segment_grid(
        duration_seconds=25.0, segment_seconds=10.0, min_segment_seconds=1.0
    )
    assert segments == [(0.0, 10.0), (10.0, 10.0), (20.0, 5.0)]


def test_remainder_dropped_when_below_threshold():
    segments = compute_segment_grid(
        duration_seconds=21.0, segment_seconds=10.0, min_segment_seconds=2.0
    )
    assert segments == [(0.0, 10.0), (10.0, 10.0)]


def test_track_shorter_than_grid_is_single_segment():
    segments = compute_segment_grid(duration_seconds=7.0, segment_seconds=10.0)
    assert segments == [(0.0, 7.0)]


def test_track_equal_to_grid_is_single_segment():
    segments = compute_segment_grid(duration_seconds=10.0, segment_seconds=10.0)
    assert segments == [(0.0, 10.0)]


def test_non_positive_duration_yields_no_segments():
    assert compute_segment_grid(duration_seconds=0.0, segment_seconds=10.0) == []
    assert compute_segment_grid(duration_seconds=-5.0, segment_seconds=10.0) == []
