from pathlib import Path

import numpy as np

from sound_loops.ffmpeg_utils import (
    decode_audio_mono,
    decode_frames_gray,
    extract_audio_segment,
    extract_frames,
    mux_loop_with_audio,
    probe,
)

DURATION_TOLERANCE = 0.2


def test_extract_audio_segment_has_expected_duration(tone_track: Path, tmp_path: Path):
    output = tmp_path / "segment.m4a"
    extract_audio_segment(
        track_path=tone_track,
        start_seconds=2.0,
        duration_seconds=4.0,
        fade_seconds=0.3,
        output_path=output,
    )

    result = probe(output)
    assert result.has_audio
    assert abs(result.duration_seconds - 4.0) < DURATION_TOLERANCE


def test_mux_contains_both_streams_and_matches_loop_duration(
    silent_loop: Path, tone_track: Path, tmp_path: Path
):
    loop_info = probe(silent_loop)

    audio_segment = tmp_path / "segment.m4a"
    extract_audio_segment(
        track_path=tone_track,
        start_seconds=1.0,
        duration_seconds=loop_info.duration_seconds,
        fade_seconds=0.3,
        output_path=audio_segment,
    )

    output = tmp_path / "render.mp4"
    mux_loop_with_audio(silent_loop, audio_segment, output)

    result = probe(output)
    assert result.has_video
    assert result.has_audio
    assert abs(result.duration_seconds - loop_info.duration_seconds) < DURATION_TOLERANCE


def test_mux_trims_audio_longer_than_loop(silent_loop: Path, tone_track: Path, tmp_path: Path):
    loop_info = probe(silent_loop)

    # Отрезок аудио заведомо длиннее лупа — итоговая длительность должна
    # определяться видео, лишнее аудио обрезается.
    audio_segment = tmp_path / "long_segment.m4a"
    extract_audio_segment(
        track_path=tone_track,
        start_seconds=0.0,
        duration_seconds=loop_info.duration_seconds + 5.0,
        fade_seconds=0.3,
        output_path=audio_segment,
    )

    output = tmp_path / "render_trimmed.mp4"
    mux_loop_with_audio(silent_loop, audio_segment, output)

    result = probe(output)
    assert abs(result.duration_seconds - loop_info.duration_seconds) < DURATION_TOLERANCE


def test_mux_repeats_loop_to_match_repeat_count(silent_loop: Path, tone_track: Path, tmp_path: Path):
    loop_info = probe(silent_loop)
    repeat_count = 2
    final_duration = loop_info.duration_seconds * repeat_count

    audio_segment = tmp_path / "long_segment.m4a"
    extract_audio_segment(
        track_path=tone_track,
        start_seconds=0.0,
        duration_seconds=final_duration,
        fade_seconds=0.3,
        output_path=audio_segment,
    )

    output = tmp_path / "render_looped.mp4"
    mux_loop_with_audio(silent_loop, audio_segment, output, repeat_count=repeat_count)

    result = probe(output)
    assert result.has_video
    assert result.has_audio
    assert abs(result.duration_seconds - final_duration) < DURATION_TOLERANCE


def test_extract_frames_returns_requested_count_of_valid_jpegs(silent_loop: Path):
    loop_info = probe(silent_loop)

    frames = extract_frames(silent_loop, loop_info.duration_seconds, count=4, max_side=448)

    assert len(frames) == 4
    for frame in frames:
        assert frame.startswith(b"\xff\xd8")  # JPEG SOI marker


def test_extract_frames_downscales_to_max_side(silent_loop: Path, tmp_path: Path):
    import subprocess

    loop_info = probe(silent_loop)
    frames = extract_frames(silent_loop, loop_info.duration_seconds, count=1, max_side=100)

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(frames[0])
    identify = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(frame_path)],
        capture_output=True,
        text=True,
    )
    width, height = (int(v) for v in identify.stdout.strip().split(","))
    assert max(width, height) <= 100


def test_decode_frames_gray_returns_expected_shape(silent_loop: Path):
    loop_info = probe(silent_loop)

    frames = decode_frames_gray(silent_loop, sample_fps=8.0, size=32)

    expected_count = round(loop_info.duration_seconds * 8.0)
    assert abs(len(frames) - expected_count) <= 1
    assert frames.shape[1:] == (32, 32)
    assert frames.dtype == np.uint8


def test_decode_audio_mono_returns_expected_sample_count(tone_track: Path):
    sample_rate = 16000
    waveform = decode_audio_mono(tone_track, sample_rate)

    track_duration = probe(tone_track).duration_seconds
    expected_samples = track_duration * sample_rate
    assert abs(len(waveform) - expected_samples) < sample_rate * DURATION_TOLERANCE
    assert waveform.dtype.name == "float32"


def test_decode_audio_mono_respects_max_seconds(tone_track: Path):
    """tone_track — 12с (conftest.py). С max_seconds=5 должно вернуться ~5с, не 12."""
    sample_rate = 16000
    waveform = decode_audio_mono(tone_track, sample_rate, max_seconds=5.0)

    expected_samples = 5.0 * sample_rate
    assert abs(len(waveform) - expected_samples) < sample_rate * DURATION_TOLERANCE
