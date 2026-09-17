"""Pydantic-валидация MusicQuery/SceneObservation/SceneDescription — чистые
функции, без базы, без Ollama и без ffmpeg."""

import pytest
from pydantic import ValidationError

from sound_loops.vlm import MusicQuery, SceneDescription, SceneObservation


def test_music_query_accepts_length_within_bounds():
    query = MusicQuery(query="slow dreamy ambient with soft piano and warm pads", vocals="instrumental")
    assert query.query.startswith("slow dreamy")


def test_music_query_rejects_too_short():
    with pytest.raises(ValidationError):
        MusicQuery(query="soft piano", vocals="instrumental")


def test_music_query_rejects_too_long():
    with pytest.raises(ValidationError):
        MusicQuery(query=" ".join(["word"] * 25), vocals="instrumental")


def test_music_query_rejects_unknown_vocals():
    with pytest.raises(ValidationError):
        MusicQuery(query="slow dreamy ambient with soft piano and warm pads", vocals="maybe")


def test_scene_observation_rejects_empty_mood():
    with pytest.raises(ValidationError):
        SceneObservation(setting="domestic", mood=[])


def test_scene_observation_rejects_too_many_moods():
    with pytest.raises(ValidationError):
        SceneObservation(setting="domestic", mood=["calm", "tense", "joyful", "epic"])


def test_scene_observation_dedupes_repeated_mood():
    observation = SceneObservation(setting="domestic", mood=["dreamy", "dreamy", "dreamy"])
    assert observation.mood == ["dreamy"]


def test_scene_observation_dedupes_while_preserving_order():
    observation = SceneObservation(setting="domestic", mood=["calm", "dreamy", "calm"])
    assert observation.mood == ["calm", "dreamy"]


def test_scene_observation_rejects_unknown_mood():
    with pytest.raises(ValidationError):
        SceneObservation(setting="domestic", mood=["euphoric"])


def test_scene_observation_rejects_unknown_setting():
    with pytest.raises(ValidationError):
        SceneObservation(setting="underwater_basket_weaving", mood=["calm"])


def test_scene_description_rejects_unknown_motion():
    with pytest.raises(ValidationError):
        SceneDescription(setting="domestic", motion="warp-speed", mood=["calm"])


def test_scene_description_rejects_unknown_setting():
    with pytest.raises(ValidationError):
        SceneDescription(setting="underwater_basket_weaving", motion="slow", mood=["calm"])
