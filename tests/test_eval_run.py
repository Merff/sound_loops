import numpy as np
import pytest
from pgvector.psycopg import register_vector

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


def test_run_eval_with_filters_relaxes_when_no_attrs(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    """Треки без tag-tracks не имеют tempo_bpm/tags — лестница послаблений
    должна дойти до полного снятия фильтров и найти их всё равно."""
    loop_path, track_paths = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)

    run = run_eval(
        db_conn,
        fake_scene_analyzer,
        fake_embedder,
        db_settings,
        _dataset(loop_path, track_paths),
        temperature=0.0,
        use_filters=True,
    )

    assert run.use_filters is True
    assert run.aggregates.loops_needing_relaxation == 1
    result = run.loops[0]
    assert result.hit_at_1 is True
    assert result.relaxed_filters == [
        "расширен диапазон темпа",
        "снято требование по вокалу",
        "фильтры сняты полностью",
    ]


def _basis(i: int, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


class _OrderedTextEmbedder:
    """embed_texts игнорирует сам текст и всегда отдаёт один и тот же вектор —
    ранжирование треков полностью управляется их вставленными эмбеддингами,
    без хэш-рандомности fake_embedder."""

    model_id = "ordered-text-embedder"
    sample_rate = 8000

    def embed_texts(self, texts):
        return np.tile(_basis(0), (len(texts), 1))

    def embed_audio(self, waveforms):
        raise NotImplementedError


def _insert_track_with_embedding(conn, path: str, embedding: np.ndarray, duration_seconds: float = 12.0) -> None:
    register_vector(conn)
    conn.execute(
        "INSERT INTO tracks (path, duration_seconds, embedding, embedding_model) VALUES (%s, %s, %s, %s)",
        (path, duration_seconds, embedding, _OrderedTextEmbedder.model_id),
    )
    conn.commit()


def test_run_eval_with_rerank_promotes_model_choice_to_hit_at_1(
    db_conn, db_settings, get_silent_loop, tmp_path, make_fake_scene_analyzer
):
    """rerank_choice=2 -> модель предпочла второй по вектору кандидат вместо топ-1;
    hit_at_1/best_rank должны отражать реальный выбор модели, а не исходный
    топ-1 по вектору (места 3+ реранком не тронуты)."""
    loop_path = get_silent_loop(tmp_path / "loop.mp4", 4.0)
    ingest_loop_file(db_conn, loop_path, db_settings)

    top_by_vector = "top_by_vector.mp3"
    second_by_vector = "second_by_vector.mp3"
    _insert_track_with_embedding(db_conn, top_by_vector, _basis(0))
    close = _basis(0) + 0.1 * _basis(1)
    _insert_track_with_embedding(db_conn, second_by_vector, (close / np.linalg.norm(close)).astype(np.float32))

    embedder = _OrderedTextEmbedder()
    analyzer = make_fake_scene_analyzer(rerank_choice=2)
    dataset = _dataset(loop_path, [second_by_vector])  # good — только второй по вектору

    without_rerank = run_eval(db_conn, analyzer, embedder, db_settings, dataset, temperature=0.0)
    assert without_rerank.loops[0].hit_at_1 is False
    assert without_rerank.loops[0].best_rank == 2

    with_rerank = run_eval(db_conn, analyzer, embedder, db_settings, dataset, temperature=0.0, use_rerank=True)
    assert with_rerank.loops[0].hit_at_1 is True
    assert with_rerank.loops[0].best_rank == 1
    assert with_rerank.loops[0].rerank_reasoning == "fake reasoning"
    assert analyzer.rerank_calls == 1


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
