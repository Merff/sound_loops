"""Эмбеддер на CLAP (transformers.ClapModel/ClapProcessor). Чекпоинт по умолчанию — laion/larger_clap_general."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from sound_loops.embeddings import normalize

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "laion/larger_clap_general"

# Заведомо непохожие фразы для проверки, что текстовая башня не схлопнута:
# если все попарные косинусные близости около 0.99 — чекпоинт для поиска
# по тексту непригоден, дальше идти бессмысленно (см. docs/sound_loops-iteration-1.md).
COLLAPSE_CHECK_PHRASES = [
    "heavy distorted guitar riff",
    "slow sad piano ballad",
    "female operatic vocal",
    "fast electronic dance music",
]


class ClapEmbedder:
    """Реальная CLAP: пробует запрошенное устройство, при ошибке — откат на CPU."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "mps") -> None:
        self.model_id = checkpoint
        self._device = self._resolve_device(device)
        logger.info("загружаю CLAP %s на %s", checkpoint, self._device)
        self.processor = ClapProcessor.from_pretrained(checkpoint)
        self.model = ClapModel.from_pretrained(checkpoint).to(self._device)
        self.model.eval()
        self.sample_rate = self.processor.feature_extractor.sampling_rate

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS недоступен, использую CPU")
            return torch.device("cpu")
        return torch.device(device)

    def _fallback_to_cpu(self, exc: Exception) -> None:
        logger.warning("ошибка на %s (%s), откатываюсь на CPU", self._device, exc)
        self._device = torch.device("cpu")
        self.model = self.model.to(self._device)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
        return self._run(
            lambda dev_inputs: self.model.get_text_features(**dev_inputs).pooler_output, inputs
        )

    def embed_audio(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        inputs = self.processor(
            audio=[np.asarray(w, dtype=np.float32) for w in waveforms],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        return self._run(
            lambda dev_inputs: self.model.get_audio_features(**dev_inputs).pooler_output, inputs
        )

    def _run(self, compute, inputs: dict) -> np.ndarray:
        try:
            with torch.no_grad():
                dev_inputs = {k: v.to(self._device) for k, v in inputs.items()}
                features = compute(dev_inputs)
        except RuntimeError as exc:
            if self._device.type == "cpu":
                raise
            self._fallback_to_cpu(exc)
            with torch.no_grad():
                dev_inputs = {k: v.to(self._device) for k, v in inputs.items()}
                features = compute(dev_inputs)
        return features.cpu().numpy()


def check_text_tower(embedder) -> np.ndarray:
    """Матрица попарных косинусных близостей COLLAPSE_CHECK_PHRASES."""
    vectors = normalize(embedder.embed_texts(COLLAPSE_CHECK_PHRASES))
    return vectors @ vectors.T
