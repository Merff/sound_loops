from sound_loops.analysis import analyze_loop, get_cached_analysis, save_analysis
from sound_loops.vlm import SceneDescription


def _insert_loop(conn, path: str = "loop.mp4", duration_seconds: float = 5.0) -> int:
    row = conn.execute(
        "INSERT INTO loops (path, duration_seconds) VALUES (%s, %s) RETURNING id",
        (path, duration_seconds),
    ).fetchone()
    conn.commit()
    return row[0]


def test_get_cached_analysis_returns_none_when_absent(db_conn):
    loop_id = _insert_loop(db_conn)
    assert get_cached_analysis(db_conn, loop_id, "some-model", "v1") is None


def test_save_and_get_cached_analysis_round_trips(db_conn):
    loop_id = _insert_loop(db_conn)
    scene = SceneDescription(summary="a cat stretches", motion="slow", mood=["calm", "dreamy"], is_comic=False)

    saved = save_analysis(db_conn, loop_id, "model-a", "v1", scene)
    cached = get_cached_analysis(db_conn, loop_id, "model-a", "v1")

    assert cached is not None
    assert cached.id == saved.id
    assert cached.scene == scene


def test_get_cached_analysis_ignores_different_model_or_prompt_version(db_conn):
    loop_id = _insert_loop(db_conn)
    scene = SceneDescription(summary="a cat stretches", motion="slow", mood=["calm"], is_comic=False)
    save_analysis(db_conn, loop_id, "model-a", "v1", scene)

    assert get_cached_analysis(db_conn, loop_id, "model-b", "v1") is None
    assert get_cached_analysis(db_conn, loop_id, "model-a", "v2") is None


def test_analyze_loop_calls_analyzer_once_and_caches(db_conn, fake_scene_analyzer):
    loop_id = _insert_loop(db_conn)
    frame_calls = []

    def get_frames():
        frame_calls.append(1)
        return [b"fake-jpeg-bytes"]

    first, first_cached = analyze_loop(db_conn, fake_scene_analyzer, loop_id, get_frames)
    second, second_cached = analyze_loop(db_conn, fake_scene_analyzer, loop_id, get_frames)

    assert first_cached is False
    assert second_cached is True
    assert first.id == second.id
    assert fake_scene_analyzer.describe_calls == 1
    assert len(frame_calls) == 1  # кадры не извлекались повторно при попадании в кеш
