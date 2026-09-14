"""Тонкие обёртки над ffmpeg/ffprobe через subprocess.

Команды собираются как плоские списки строк, которые остаются читаемыми и
копируемыми в терминал как есть — никаких питоновских DSL поверх ffmpeg.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FfmpegError(RuntimeError):
    """Бинарь ffmpeg/ffprobe завершился с ошибкой."""


def run(cmd: list[str]) -> str:
    """Выполнить команду, вернуть stdout. Бросить FfmpegError с текстом stderr при неудаче."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pretty_cmd = " ".join(cmd)
        stderr = result.stderr.strip()
        raise FfmpegError(
            f"команда завершилась с кодом {result.returncode}: {pretty_cmd}\n{stderr}"
        )
    return result.stdout


@dataclass(frozen=True)
class ProbeResult:
    duration_seconds: float
    width: int | None
    height: int | None
    has_video: bool
    has_audio: bool


def probe(path: Path) -> ProbeResult:
    """Прочитать метаданные медиафайла через ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        stdout = run(cmd)
    except FfmpegError as exc:
        raise FfmpegError(f"ffprobe не смог прочитать {path}: {exc}") from exc

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise FfmpegError(f"ffprobe вернул невалидный JSON для {path}: {exc}") from exc

    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    duration_raw = data.get("format", {}).get("duration")
    if duration_raw is None and video_streams:
        duration_raw = video_streams[0].get("duration")
    if duration_raw is None:
        raise FfmpegError(f"не удалось определить длительность {path}")
    duration = float(duration_raw)

    width = video_streams[0].get("width") if video_streams else None
    height = video_streams[0].get("height") if video_streams else None

    return ProbeResult(
        duration_seconds=duration,
        width=width,
        height=height,
        has_video=bool(video_streams),
        has_audio=bool(audio_streams),
    )


def extract_audio_segment(
    track_path: Path,
    start_seconds: float,
    duration_seconds: float,
    fade_seconds: float,
    output_path: Path,
) -> None:
    """Вырезать отрезок аудио из трека с короткими фейдами на краях.

    Фейды нужны, иначе на срезе слышен щелчок независимо от того, насколько
    удачно выбрана музыка.
    """
    fade = min(fade_seconds, duration_seconds / 2)
    fade_out_start = max(duration_seconds - fade, 0.0)
    audio_filter = (
        f"afade=t=in:st=0:d={fade:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade:.3f}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_seconds:.3f}",
        "-t", f"{duration_seconds:.3f}",
        "-i", str(track_path),
        "-af", audio_filter,
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ]
    run(cmd)


def decode_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Декодировать аудиодорожку в моно float32 PCM заданной частоты дискретизации."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        pretty_cmd = " ".join(cmd)
        stderr = result.stderr.decode(errors="replace").strip()
        raise FfmpegError(
            f"команда завершилась с кодом {result.returncode}: {pretty_cmd}\n{stderr}"
        )
    return np.frombuffer(result.stdout, dtype=np.float32)


def mux_loop_with_audio(loop_path: Path, audio_path: Path, output_path: Path) -> None:
    """Склеить немой видео-луп с готовым аудио без перекодирования видео.

    Итоговую длительность определяет видео: -shortest обрезает лишнее аудио,
    если оно почему-то оказалось длиннее.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(loop_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        str(output_path),
    ]
    run(cmd)
