"""normalize_tempo_octave — чистая функция, без librosa/ffmpeg/базы."""

from sound_loops.tempo import normalize_tempo_octave


def test_within_bounds_unchanged():
    assert normalize_tempo_octave(120.0, low=60.0, high=200.0) == 120.0


def test_at_high_bound_unchanged():
    assert normalize_tempo_octave(200.0, low=60.0, high=200.0) == 200.0


def test_at_low_bound_unchanged():
    assert normalize_tempo_octave(60.0, low=60.0, high=200.0) == 60.0


def test_doubles_when_too_low():
    assert normalize_tempo_octave(30.0, low=60.0, high=200.0) == 60.0


def test_doubles_repeatedly_when_far_too_low():
    assert normalize_tempo_octave(15.0, low=60.0, high=200.0) == 60.0


def test_halves_when_too_high():
    assert normalize_tempo_octave(250.0, low=60.0, high=200.0) == 125.0


def test_halves_repeatedly_when_far_too_high():
    assert normalize_tempo_octave(410.0, low=60.0, high=200.0) == 102.5


def test_zero_returns_low_without_looping_forever():
    assert normalize_tempo_octave(0.0, low=60.0, high=200.0) == 60.0


def test_negative_returns_low_without_looping_forever():
    assert normalize_tempo_octave(-5.0, low=60.0, high=200.0) == 60.0
