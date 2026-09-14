import os
from pathlib import Path

from sound_loops.hf_cache import ensure_offline_if_cached


def _make_cached_checkpoint(cache_dir: Path, checkpoint: str) -> None:
    snapshot = cache_dir / f"models--{checkpoint.replace('/', '--')}" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")


def test_sets_offline_when_checkpoint_is_cached(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _make_cached_checkpoint(tmp_path, "laion/larger_clap_general")

    ensure_offline_if_cached("laion/larger_clap_general")

    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_leaves_online_when_checkpoint_is_not_cached(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    ensure_offline_if_cached("laion/larger_clap_general")

    assert "HF_HUB_OFFLINE" not in os.environ


def test_does_not_override_explicit_user_choice(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _make_cached_checkpoint(tmp_path, "laion/larger_clap_general")

    ensure_offline_if_cached("laion/larger_clap_general")

    assert os.environ["HF_HUB_OFFLINE"] == "0"
