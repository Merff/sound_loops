"""Оценка темпа трека (BPM) через librosa — единственная новая тяжёлая
зависимость проекта (итерация 4, docs/sound_loops-iteration-4.md).
Декодирование по-прежнему через ffmpeg (decode_audio_mono), librosa только
считает биты по готовому PCM.
"""

from __future__ import annotations

import librosa
import numpy as np

# Алгоритмы beat-tracking регулярно путают темп с половинным/двойным
# (октавная ошибка) — приводим к этому диапазону, а не отбрасываем
# значение. Границы взяты на глаз (разумный диапазон реальной музыки),
# калибровать по метрикам, не по теории.
TEMPO_MIN_BPM = 60.0
TEMPO_MAX_BPM = 200.0


def estimate_tempo_bpm(waveform: np.ndarray, sample_rate: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
    # librosa возвращает 1-элементный ndarray, а не скаляр.
    return float(np.asarray(tempo).reshape(-1)[0])


def normalize_tempo_octave(bpm: float, low: float = TEMPO_MIN_BPM, high: float = TEMPO_MAX_BPM) -> float:
    """Удвоить/раздвоить bpm, пока оно не попадёт в [low, high]."""
    if bpm <= 0:
        return low
    while bpm < low:
        bpm *= 2
    while bpm > high:
        bpm /= 2
    return bpm
