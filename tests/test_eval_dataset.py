import json

import pytest
from pydantic import ValidationError

from sound_loops.eval_dataset import LoopAnnotation, load_dataset, save_dataset

_VALID = {
    "loop": "data/loops/loop-1.mp4",
    "setting": "nature",
    "tempo": "slow",
    "mood": ["calm", "dreamy"],
    "vocals": "instrumental",
    "good_tracks": ["data/raw/fma_small/000/000002.mp3", "data/raw/fma_small/000/000005.mp3"],
}


def test_valid_annotation_parses():
    entry = LoopAnnotation.model_validate(_VALID)
    assert entry.mood == ["calm", "dreamy"]


def test_invalid_mood_label_raises_clear_error():
    bad = {**_VALID, "mood": ["not_a_real_mood"]}
    with pytest.raises(ValidationError):
        LoopAnnotation.model_validate(bad)


def test_empty_good_tracks_raises():
    bad = {**_VALID, "good_tracks": []}
    with pytest.raises(ValidationError):
        LoopAnnotation.model_validate(bad)


def test_single_good_track_is_accepted():
    entry = LoopAnnotation.model_validate({**_VALID, "good_tracks": ["only_one.mp3"]})
    assert entry.good_tracks == ["only_one.mp3"]


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.json")


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "dataset.json"
    entries = [LoopAnnotation.model_validate(_VALID)]

    save_dataset(path, entries)
    loaded = load_dataset(path)

    assert loaded == entries


def test_load_dataset_bad_entry_raises_clear_error(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([{**_VALID, "tempo": "medium"}]))

    with pytest.raises(ValidationError):
        load_dataset(path)
