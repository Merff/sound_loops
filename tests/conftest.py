"""Общие фикстуры: синтетические видео-луп и трек, сгенерированные ffmpeg.

Тесты не зависят от скачанного датасета FMA или реальных лупов — медиа
генерируется на лету через lavfi-источники ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sound_loops.ffmpeg_utils import run


def make_silent_loop(path: Path, duration_seconds: float) -> Path:
    """Немой mp4: testsrc, без аудиодорожки — как настоящий видео-луп."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration_seconds}:size=320x240:rate=25",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ]
    run(cmd)
    return path


def make_tone_track(path: Path, duration_seconds: float) -> Path:
    """mp3 с синусоидой — как отрезок настоящего музыкального трека."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_seconds}",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(path),
    ]
    run(cmd)
    return path


@pytest.fixture
def silent_loop(tmp_path: Path) -> Path:
    return make_silent_loop(tmp_path / "loop.mp4", duration_seconds=4.0)


@pytest.fixture
def tone_track(tmp_path: Path) -> Path:
    return make_tone_track(tmp_path / "track.mp3", duration_seconds=12.0)
