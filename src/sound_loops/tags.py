"""Zero-shot теги треков через CLAP (итерация 4, docs/sound_loops-iteration-4.md):
текстовый эмбеддинг контрастных фраз против уже посчитанного аудио-эмбеддинга
трека (tracks.embedding, см. index.py) — без новой модели и без повторного
декодирования аудио. Хранятся числа близости, не победившая метка — порог
можно двигать без пересчёта. Для mood/genre берётся топ-3, вокал — обе фразы.
"""

from __future__ import annotations

import numpy as np

from sound_loops.embeddings import Embedder, normalize

# Ключи совпадают со значениями vlm.Vocals — гибридный поиск (search.py)
# сравнивает эти два числа напрямую как SQL-фильтр по вокалу.
VOCAL_PHRASES = {
    "instrumental": "instrumental music with no vocals",
    "with_vocals": "a song with singing voice",
}

# Те же 14 категорий, что в схеме шага A (vlm.py::Mood).
MOOD_PHRASES = {
    "calm": "calm, peaceful music",
    "tense": "tense, suspenseful music",
    "joyful": "joyful, upbeat music",
    "melancholic": "melancholic, sad music",
    "epic": "epic, heroic music",
    "comic": "comic, playful music",
    "eerie": "eerie, unsettling music",
    "romantic": "romantic music",
    "aggressive": "aggressive, intense music",
    "nostalgic": "nostalgic music",
    "dreamy": "dreamy, ambient music",
    "triumphant": "triumphant, victorious music",
    "tragic": "tragic, mournful music",
    "festive": "festive, celebratory music",
}

# Инструменты + реально проиндексированные жанры FMA_small (см. vlm.py::_LIBRARY_GENRES).
GENRE_PHRASES = {
    "electronic": "electronic music with synthesizers",
    "rock": "rock music with electric guitars",
    "hip_hop": "hip-hop music with a beat",
    "pop": "pop music",
    "folk": "folk music with acoustic guitar",
    "experimental": "experimental, abstract music",
    "piano": "solo piano music",
    "strings": "orchestral strings",
    "drums": "percussion and drums",
    "synth_pads": "ambient synth pads",
}

TOP_K = 3


def _score_all(text_vectors: np.ndarray, labels: list[str], audio_vector: np.ndarray) -> dict[str, float]:
    sims = text_vectors @ audio_vector
    return {label: float(s) for label, s in zip(labels, sims, strict=True)}


def _top_k(scores: dict[str, float], k: int) -> dict[str, float]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return dict(ranked[:k])


class TagPhraseBank:
    """Текстовые эмбеддинги контрастных фраз — считаются один раз на всю библиотеку."""

    def __init__(self, embedder: Embedder) -> None:
        self._vocal_labels = list(VOCAL_PHRASES)
        self._mood_labels = list(MOOD_PHRASES)
        self._genre_labels = list(GENRE_PHRASES)

        phrases = (
            [VOCAL_PHRASES[label] for label in self._vocal_labels]
            + [MOOD_PHRASES[label] for label in self._mood_labels]
            + [GENRE_PHRASES[label] for label in self._genre_labels]
        )
        vectors = normalize(embedder.embed_texts(phrases))

        n_vocal, n_mood = len(self._vocal_labels), len(self._mood_labels)
        self._vocal_vectors = vectors[:n_vocal]
        self._mood_vectors = vectors[n_vocal : n_vocal + n_mood]
        self._genre_vectors = vectors[n_vocal + n_mood :]

    def tags_for(self, audio_vector: np.ndarray) -> dict[str, dict[str, float]]:
        """audio_vector — уже нормализованный tracks.embedding."""
        vocals = _score_all(self._vocal_vectors, self._vocal_labels, audio_vector)
        mood = _top_k(_score_all(self._mood_vectors, self._mood_labels, audio_vector), TOP_K)
        genre = _top_k(_score_all(self._genre_vectors, self._genre_labels, audio_vector), TOP_K)
        return {"vocals": vocals, "mood": mood, "genre": genre}
