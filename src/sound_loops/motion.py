"""Алгоритмическая оценка количества движения в лупе — через разницу
соседних кадров, а не через VLM: маленькая модель ненадёжно сравнивает
между собой отдельные картинки, а разница кадров через ffmpeg дёшева и
детерминирована. VLM (vlm.py) отвечает только за то, с чем справляется —
смысл и настроение.

Пороги ниже — первое приближение, не строгая теория. Если на новом наборе
лупов категории ощущаются неправильно — крутить эти числа, а не архитектуру.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sound_loops.ffmpeg_utils import decode_frames_gray
from sound_loops.vlm import Motion

STATIC_MEAN = 0.015
SLOW_MEAN = 0.035
MODERATE_MEAN = 0.07
CHAOTIC_MEAN = 0.15
CHAOTIC_CV = 1.5


def frame_diff_scores(frames: np.ndarray) -> list[float]:
    """Нормализованная (0..1) средняя абсолютная разница соседних кадров."""
    if len(frames) < 2:
        return []
    diffs = frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)
    return [float(np.abs(d).mean() / 255.0) for d in diffs]


def classify_motion(diffs: Sequence[float]) -> Motion:
    """Разница кадров -> одна из пяти категорий motion.

    Помимо среднего уровня разницы (mean) учитывается её неравномерность
    (cv — коэффициент вариации): резкие склейки между непохожими кадрами
    дают редкие всплески на фоне низкой разницы внутри кадра — это тоже
    "chaotic", а не "static", даже если средний уровень невысок.
    """
    if not diffs:
        return "static"

    mean = float(np.mean(diffs))
    if mean < STATIC_MEAN:
        return "static"

    std = float(np.std(diffs))
    cv = std / mean if mean > 0 else 0.0
    if mean >= CHAOTIC_MEAN or cv >= CHAOTIC_CV:
        return "chaotic"
    if mean < SLOW_MEAN:
        return "slow"
    if mean < MODERATE_MEAN:
        return "moderate"
    return "fast"


def estimate_motion(loop_path: Path, sample_fps: float, frame_size: int) -> Motion:
    frames = decode_frames_gray(loop_path, sample_fps, frame_size)
    return classify_motion(frame_diff_scores(frames))
