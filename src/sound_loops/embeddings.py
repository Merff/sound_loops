"""Интерфейс эмбеддера и общие утилиты — независимо от конкретной модели.

Вычисление эмбеддинга спрятано за Embedder: index.py/search.py работают
с любым объектом, у которого есть эти методы, не зная про transformers/torch.
В тестах это позволяет подставлять детерминированный fake-эмбеддер вместо
настоящей CLAP — саму CLAP проверяют sanity-check команды и ручное прослушивание.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_id: str
    sample_rate: int

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Текст -> эмбеддинги, форма (len(texts), dim)."""

    def embed_audio(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        """Моно PCM (sample_rate каждой волны) -> эмбеддинги, форма (len(waveforms), dim)."""


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Привести строки к единичной длине. Нулевые вектора оставить как есть."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
