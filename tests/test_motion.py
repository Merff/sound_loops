"""frame_diff_scores/classify_motion — чистые функции, без ffmpeg и без базы."""

import numpy as np
import pytest

from sound_loops.motion import (
    CHAOTIC_CV,
    CHAOTIC_MEAN,
    MODERATE_MEAN,
    SLOW_MEAN,
    STATIC_MEAN,
    classify_motion,
    frame_diff_scores,
)


def test_frame_diff_scores_identical_frames_is_zero():
    frames = np.zeros((5, 8, 8), dtype=np.uint8)
    assert frame_diff_scores(frames) == [0.0, 0.0, 0.0, 0.0]


def test_frame_diff_scores_known_constant_difference():
    frames = np.zeros((3, 4, 4), dtype=np.uint8)
    frames[1] = 255
    frames[2] = 0
    assert frame_diff_scores(frames) == pytest.approx([1.0, 1.0])


def test_frame_diff_scores_single_frame_is_empty():
    frames = np.zeros((1, 4, 4), dtype=np.uint8)
    assert frame_diff_scores(frames) == []


def test_classify_motion_empty_is_static():
    assert classify_motion([]) == "static"


def test_classify_motion_low_mean_is_static():
    assert classify_motion([STATIC_MEAN / 2] * 10) == "static"


def test_classify_motion_mid_range_is_slow():
    assert classify_motion([(STATIC_MEAN + SLOW_MEAN) / 2] * 10) == "slow"


def test_classify_motion_higher_range_is_moderate():
    assert classify_motion([(SLOW_MEAN + MODERATE_MEAN) / 2] * 10) == "moderate"


def test_classify_motion_high_uniform_is_fast():
    assert classify_motion([(MODERATE_MEAN + CHAOTIC_MEAN) / 2] * 10) == "fast"


def test_classify_motion_very_high_mean_is_chaotic():
    assert classify_motion([CHAOTIC_MEAN * 2] * 10) == "chaotic"


def test_classify_motion_spiky_low_mean_is_chaotic():
    # Невысокий средний уровень, но один резкий всплеск (например, склейка
    # между непохожими кадрами) — высокий коэффициент вариации должен
    # переклассифицировать в chaotic, а не moderate/slow по одному mean.
    diffs = [0.01] * 9 + [0.5]
    assert np.std(diffs) / np.mean(diffs) >= CHAOTIC_CV  # предпосылка теста
    assert classify_motion(diffs) == "chaotic"
