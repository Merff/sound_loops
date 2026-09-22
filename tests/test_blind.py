from pathlib import Path

import pytest

from sound_loops.analysis import analyze_loop_by_path
from sound_loops.blind import BlindPair, prepare_pairs, score_pairs
from sound_loops.eval_dataset import LoopAnnotation
from sound_loops.index import index_tracks
from sound_loops.ingest import IngestReport, ingest_loop_file, ingest_track_file
from sound_loops.match import MatchError


def _setup(conn, settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder):
    loop_path = get_silent_loop(tmp_path / "loop.mp4", 4.0)
    ingest_loop_file(conn, loop_path, settings)
    track_path = get_tone_track(tmp_path / "000001.mp3", 12.0)
    ingest_track_file(conn, track_path, {}, IngestReport())
    index_tracks(conn, fake_embedder, batch_size=8)
    return loop_path


def _dataset(loop_path):
    return [
        LoopAnnotation(
            loop=str(loop_path),
            setting="domestic",
            tempo="slow",
            mood=["calm"],
            vocals="any",
            good_tracks=["a.mp3", "b.mp3"],
        )
    ]


def test_prepare_pairs_requires_prior_analysis(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    loop_path = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)

    with pytest.raises(MatchError):
        prepare_pairs(db_conn, fake_scene_analyzer, fake_embedder, db_settings, _dataset(loop_path))


def test_prepare_pairs_produces_two_distinct_playable_outputs(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    loop_path = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    analyze_loop_by_path(db_conn, fake_scene_analyzer, db_settings, loop_path)

    pairs = prepare_pairs(db_conn, fake_scene_analyzer, fake_embedder, db_settings, _dataset(loop_path))

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.path_a != pair.path_b
    assert Path(pair.path_a).exists()
    assert Path(pair.path_b).exists()
    assert pair.label_a in ("pipeline", "baseline")


def test_score_pairs_records_which_side_won(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    loop_path = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    analyze_loop_by_path(db_conn, fake_scene_analyzer, db_settings, loop_path)
    pairs = prepare_pairs(db_conn, fake_scene_analyzer, fake_embedder, db_settings, _dataset(loop_path))

    run = score_pairs(pairs, lambda pair: "A")

    assert len(run.answers) == 1
    answer = run.answers[0]
    assert answer.choice == "A"
    assert answer.winner == answer.label_a
    assert run.pipeline_win_rate in (0.0, 1.0)


def test_score_pairs_excludes_ties_from_win_rate():
    pairs = [BlindPair(loop="x.mp4", label_a="pipeline", path_a="a.mp4", path_b="b.mp4")]

    run = score_pairs(pairs, lambda pair: "tie")

    assert run.answers[0].winner == "tie"
    assert run.pipeline_win_rate == 0.0


def test_score_pairs_win_rate_ignores_baseline_label_direction():
    pairs = [
        BlindPair(loop="x.mp4", label_a="pipeline", path_a="a.mp4", path_b="b.mp4"),
        BlindPair(loop="y.mp4", label_a="baseline", path_a="a.mp4", path_b="b.mp4"),
    ]

    # выбираем "сторону B" в обоих случаях: для x.mp4 это baseline, для y.mp4 — pipeline
    run = score_pairs(pairs, lambda pair: "B")

    winners = {a.loop: a.winner for a in run.answers}
    assert winners == {"x.mp4": "baseline", "y.mp4": "pipeline"}
    assert run.pipeline_win_rate == 0.5
