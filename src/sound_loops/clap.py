"""Эмбеддер на CLAP (transformers.ClapModel/ClapProcessor). Чекпоинт по умолчанию — laion/larger_clap_general."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from sound_loops.embeddings import normalize

logger = logging.getLogger(__name__)


def _from_pretrained(cls, checkpoint: str):
    """from_pretrained с разовым запасным путём на случай HF_HUB_OFFLINE=1 без кэша.

    Обычно офлайн-режим уже включён до импорта transformers (см.
    hf_cache.ensure_offline_if_cached — выставлять HF_HUB_OFFLINE после
    импорта ненадёжно, часть внутренних клиентов huggingface_hub его не
    подхватывает). Если чекпоинта всё же нет локально, на один раз снимаем
    офлайн-режим и качаем.
    """
    try:
        return cls.from_pretrained(checkpoint)
    except OSError:
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise
        logger.info("%s не найден в локальном кэше при HF_HUB_OFFLINE=1, скачиваю...", checkpoint)
        del os.environ["HF_HUB_OFFLINE"]
        try:
            return cls.from_pretrained(checkpoint)
        finally:
            os.environ["HF_HUB_OFFLINE"] = "1"

DEFAULT_CHECKPOINT = "laion/larger_clap_general"

# Заведомо непохожие фразы для проверки, что текстовая башня не схлопнута
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
        self.processor = _from_pretrained(ClapProcessor, checkpoint)
        self.model = _from_pretrained(ClapModel, checkpoint).to(self._device)
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
    vectors = normalize(embedder.embed_texts(COLLAPSE_CHECK_PHRASES))
    return vectors @ vectors.T
