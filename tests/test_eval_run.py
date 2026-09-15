import pytest

from sound_loops.eval_dataset import LoopAnnotation
from sound_loops.eval_run import load_run, run_eval, save_run
from sound_loops.index import index_tracks
from sound_loops.ingest import IngestReport, ingest_loop_file, ingest_track_file


def _setup(conn, settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder):
    loop_path = get_silent_loop(tmp_path / "loop.mp4", 4.0)
    ingest_loop_file(conn, loop_path, settings)

    track_a = get_tone_track(tmp_path / "000001.mp3", 12.0)
    track_b = get_tone_track(tmp_path / "000002.mp3", 12.0)
    ingest_track_file(conn, track_a, {}, IngestReport())
    ingest_track_file(conn, track_b, {}, IngestReport())
    index_tracks(conn, fake_embedder, batch_size=8)

    return loop_path, [str(track_a), str(track_b)]


def _dataset(loop_path, good_tracks):
    return [
        LoopAnnotation(
            loop=str(loop_path),
            setting="domestic",
            tempo="slow",
            mood=["calm"],
            vocals="any",
            good_tracks=good_tracks,
        )
    ]


def test_run_eval_computes_metrics(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    loop_path, track_paths = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)

    run = run_eval(
        db_conn, fake_scene_analyzer, fake_embedder, db_settings, _dataset(loop_path, track_paths), temperature=0.0
    )

    assert len(run.loops) == 1
    result = run.loops[0]
    # fake_scene_analyzer всегда возвращает setting=domestic, mood=[calm] — совпадает с разметкой
    assert result.setting_correct is True
    assert result.mood_overlap == 1.0
    # оба трека размечены как подходящие -> топ-1 гарантированно попадает
    assert result.hit_at_1 is True
    assert result.hit_at_5 is True
    assert result.best_rank == 1
    assert run.aggregates.setting_accuracy == 1.0
    assert run.temperature == 0.0
    assert run.search_depth == db_settings.eval_search_depth


def test_run_eval_empty_dataset_raises(db_conn, db_settings, fake_embedder, fake_scene_analyzer):
    with pytest.raises(ValueError):
        run_eval(db_conn, fake_scene_analyzer, fake_embedder, db_settings, [], temperature=0.0)


def test_run_eval_never_reruns_scene_analysis(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    loop_path, track_paths = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    dataset = _dataset(loop_path, track_paths)

    run_eval(db_conn, fake_scene_analyzer, fake_embedder, db_settings, dataset, temperature=0.0)
    assert fake_scene_analyzer.describe_calls == 1

    run_eval(db_conn, fake_scene_analyzer, fake_embedder, db_settings, dataset, temperature=0.0)
    assert fake_scene_analyzer.describe_calls == 1


def test_save_and_load_run_roundtrip(
    tmp_path, db_conn, db_settings, get_silent_loop, get_tone_track, fake_embedder, fake_scene_analyzer
):
    loop_path, track_paths = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    run = run_eval(
        db_conn, fake_scene_analyzer, fake_embedder, db_settings, _dataset(loop_path, track_paths), temperature=0.0
    )

    path = save_run(run, tmp_path / "runs")
    loaded = load_run(path)

    assert loaded == run
