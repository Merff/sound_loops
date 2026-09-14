"""MusicQuery._word_count — чистая функция, без базы и без Ollama."""

import pytest
from pydantic import ValidationError

from sound_loops.vlm import MusicQuery, SceneDescription


def test_music_query_accepts_length_within_bounds():
    query = MusicQuery(query="slow dreamy ambient with soft piano and warm pads, instrumental")
    assert query.query.startswith("slow dreamy")


def test_music_query_rejects_too_short():
    with pytest.raises(ValidationError):
        MusicQuery(query="soft piano")


def test_music_query_rejects_too_long():
    with pytest.raises(ValidationError):
        MusicQuery(query=" ".join(["word"] * 25))


def test_scene_description_rejects_empty_mood():
    with pytest.raises(ValidationError):
        SceneDescription(summary="a test scene", motion="slow", mood=[], is_comic=False)


def test_scene_description_rejects_too_many_moods():
    with pytest.raises(ValidationError):
        SceneDescription(
            summary="a test scene",
            motion="slow",
            mood=["calm", "tense", "joyful", "epic"],
            is_comic=False,
        )


def test_scene_description_rejects_unknown_motion():
    with pytest.raises(ValidationError):
        SceneDescription(summary="a test scene", motion="warp-speed", mood=["calm"], is_comic=False)
