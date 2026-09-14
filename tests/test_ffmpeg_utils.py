from pathlib import Path

from sound_loops.ffmpeg_utils import decode_audio_mono, extract_audio_segment, mux_loop_with_audio, probe

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


def test_decode_audio_mono_returns_expected_sample_count(tone_track: Path):
    sample_rate = 16000
    waveform = decode_audio_mono(tone_track, sample_rate)

    track_duration = probe(tone_track).duration_seconds
    expected_samples = track_duration * sample_rate
    assert abs(len(waveform) - expected_samples) < sample_rate * DURATION_TOLERANCE
    assert waveform.dtype.name == "float32"
