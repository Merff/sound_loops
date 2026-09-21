import json

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
    assert run.aggregates.mean_penalized_rank == 1.0  # найден -> penalized_rank == best_rank
    assert run.temperature == 0.0
    assert run.search_depth == db_settings.eval_search_depth


def test_run_eval_mean_penalized_rank_uses_search_depth_for_not_found(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    loop_path, _track_paths = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)

    run = run_eval(
        db_conn,
        fake_scene_analyzer,
        fake_embedder,
        db_settings,
        _dataset(loop_path, ["nonexistent_track.mp3"]),
        temperature=0.0,
    )

    assert run.loops[0].best_rank is None
    assert run.aggregates.mean_best_rank is None
    assert run.aggregates.mean_penalized_rank == db_settings.eval_search_depth


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


def test_run_eval_agent_computes_hit_at_k_over_merged_query_pools(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    """use_agent=True: hit@k считается по объединению кандидатов всех 3
    query агента (см. docs/sound_loops-iteration-5.md, раздел «Эвалы»)."""
    loop_path, track_paths = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns(
        [[{"query": "query one"}, {"query": "query two"}, {"query": "query three"}]]
    )

    run = run_eval(
        db_conn, fake_scene_analyzer, fake_embedder, db_settings, _dataset(loop_path, track_paths),
        temperature=0.0, use_agent=True,
    )

    assert run.use_agent is True
    result = run.loops[0]
    assert len(result.queries) == 3
    assert result.hit_at_1 is True
    assert result.tool_calls_made == 3
    assert result.fallback_used == 0
    assert run.aggregates.agent_tool_calls_total == 3
    assert run.aggregates.agent_fallback_used_total == 0


def test_run_eval_agent_does_not_render_anything(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    loop_path, track_paths = _setup(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns(
        [[{"query": "query one"}, {"query": "query two"}, {"query": "query three"}]]
    )

    run_eval(
        db_conn, fake_scene_analyzer, fake_embedder, db_settings, _dataset(loop_path, track_paths),
        temperature=0.0, use_agent=True,
    )

    assert db_conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 0


def test_load_run_backfills_mean_penalized_rank_for_old_files(tmp_path):
    """Прогоны, сохранённые до появления mean_penalized_rank (итерация 4),
    не должны стать нечитаемыми — см. load_run."""
    old_format = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "commit": "abc123",
        "model": "m",
        "prompt_version": "v1",
        "temperature": 0.0,
        "search_depth": 100,
        "loops": [
            {
                "loop": "a.mp4",
                "setting_correct": True,
                "predicted_setting": "domestic",
                "true_setting": "domestic",
                "mood_overlap": 1.0,
                "predicted_mood": ["calm"],
                "true_mood": ["calm"],
                "music_query": "q",
                "hit_at_1": True,
                "hit_at_5": True,
                "best_rank": 3,
            },
            {
                "loop": "b.mp4",
                "setting_correct": False,
                "predicted_setting": "urban",
                "true_setting": "nature",
                "mood_overlap": 0.0,
                "predicted_mood": ["calm"],
                "true_mood": ["tense"],
                "music_query": "q2",
                "hit_at_1": False,
                "hit_at_5": False,
                "best_rank": None,
            },
        ],
        "aggregates": {
            "setting_accuracy": 0.5,
            "mean_mood_overlap": 0.5,
            "hit_at_1_rate": 0.5,
            "hit_at_5_rate": 0.5,
            "mean_best_rank": 3.0,
            "not_found_count": 1,
        },
    }
    path = tmp_path / "old_run.json"
    path.write_text(json.dumps(old_format))

    loaded = load_run(path)

    # (3 + 100) / 2 — второй луп не найден, штраф = search_depth.
    assert loaded.aggregates.mean_penalized_rank == 51.5
