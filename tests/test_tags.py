"""TagPhraseBank — тестируется с fake_embedder (детерминированный, без CLAP)."""

import numpy as np

from sound_loops.embeddings import normalize
from sound_loops.tags import GENRE_PHRASES, MOOD_PHRASES, VOCAL_PHRASES, TagPhraseBank


def test_tags_for_has_all_three_categories(fake_embedder):
    bank = TagPhraseBank(fake_embedder)
    audio_vector = normalize(np.random.default_rng(0).normal(size=(1, 512)))[0].astype(np.float32)

    tags = bank.tags_for(audio_vector)

    assert set(tags) == {"vocals", "mood", "genre"}


def test_vocals_keeps_both_phrases(fake_embedder):
    bank = TagPhraseBank(fake_embedder)
    audio_vector = normalize(np.random.default_rng(1).normal(size=(1, 512)))[0].astype(np.float32)

    tags = bank.tags_for(audio_vector)

    assert set(tags["vocals"]) == set(VOCAL_PHRASES)


def test_mood_and_genre_keep_top_3(fake_embedder):
    bank = TagPhraseBank(fake_embedder)
    audio_vector = normalize(np.random.default_rng(2).normal(size=(1, 512)))[0].astype(np.float32)

    tags = bank.tags_for(audio_vector)

    assert len(tags["mood"]) == 3
    assert set(tags["mood"]) <= set(MOOD_PHRASES)
    assert len(tags["genre"]) == 3
    assert set(tags["genre"]) <= set(GENRE_PHRASES)


def test_top_3_are_the_highest_scores(fake_embedder):
    bank = TagPhraseBank(fake_embedder)
    audio_vector = normalize(np.random.default_rng(3).normal(size=(1, 512)))[0].astype(np.float32)

    all_mood_scores = {
        label: float(vec @ audio_vector) for label, vec in zip(bank._mood_labels, bank._mood_vectors, strict=True)
    }
    expected_top_3 = sorted(all_mood_scores, key=all_mood_scores.get, reverse=True)[:3]

    tags = bank.tags_for(audio_vector)

    assert set(tags["mood"]) == set(expected_top_3)
