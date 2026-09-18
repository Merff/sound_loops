"""Тесты графа (итерация 5, docs/sound_loops-iteration-5.md, раздел «Тесты»):
условное ребро feedback->plan/конец, счётчик кругов, дедуп по трекам,
резервный путь и восстановление сессии по thread_id."""

import uuid

from langgraph.types import Command

from sound_loops.agent_graph import build_agent_graph, build_eval_graph, initial_state
from sound_loops.index import index_tracks
from sound_loops.ingest import IngestReport, ingest_loop_file, ingest_track_file
from sound_loops.render import set_render_rating


def _seed(conn, settings, get_silent_loop, get_tone_track, tmp_path, embedder, n_tracks=8, duration=4.0):
    loop_path = get_silent_loop(tmp_path / "loop.mp4", duration)
    loop_id, _ = ingest_loop_file(conn, loop_path, settings)
    for i in range(n_tracks):
        track_path = get_tone_track(tmp_path / f"{i:06d}.mp3", 20.0)
        ingest_track_file(conn, track_path, {}, IngestReport())
    index_tracks(conn, embedder, batch_size=n_tracks)
    return loop_id, loop_path


def _three_calls(*labels: str) -> list[list[dict]]:
    return [[{"query": label} for label in labels]]


def test_agent_graph_stops_at_feedback_with_three_deduped_candidates(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer, checkpointer
):
    _loop_id, loop_path = _seed(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns(_three_calls("query a", "query b", "query c") * 3)
    db_settings.agent_max_rounds = 2

    graph = build_agent_graph(db_conn, fake_embedder, fake_scene_analyzer, db_settings, checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    state = graph.invoke(initial_state(str(loop_path), db_settings), config)

    assert graph.get_state(config).next == ("feedback",)
    assert len(state["candidates"]) == 3
    assert len({c["id"] for c in state["candidates"]}) == 3  # без дублей внутри круга
    assert state["rounds"] == 0
    assert len(state["output_paths"]) == 3


def test_agent_graph_reroll_excludes_previous_round_tracks(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer, checkpointer
):
    _loop_id, loop_path = _seed(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    db_settings.agent_max_rounds = 3
    fake_scene_analyzer.set_tool_call_turns(_three_calls("r1 a", "r1 b", "r1 c") * 3)

    graph = build_agent_graph(db_conn, fake_embedder, fake_scene_analyzer, db_settings, checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    first = graph.invoke(initial_state(str(loop_path), db_settings), config)
    round1_ids = {c["id"] for c in first["candidates"]}

    fake_scene_analyzer.set_tool_call_turns(_three_calls("r2 a", "r2 b", "r2 c") * 3)
    second = graph.invoke(Command(resume="darker please"), config)
    round2_ids = {c["id"] for c in second["candidates"]}

    assert round1_ids.isdisjoint(round2_ids)
    assert set(second["rejected_track_ids"]) >= round1_ids
    assert second["feedback_history"] == ["darker please"]
    assert second["rounds"] == 1
    assert graph.get_state(config).next == ("feedback",)


def test_agent_graph_ends_when_round_limit_reached_even_with_feedback_text(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer, checkpointer
):
    _loop_id, loop_path = _seed(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    db_settings.agent_max_rounds = 1
    fake_scene_analyzer.set_tool_call_turns(_three_calls("a", "b", "c") * 3)

    graph = build_agent_graph(db_conn, fake_embedder, fake_scene_analyzer, db_settings, checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    graph.invoke(initial_state(str(loop_path), db_settings), config)
    final_state = graph.invoke(Command(resume="still not right"), config)

    assert graph.get_state(config).next == ()
    assert final_state["rounds"] == 1


def test_agent_graph_ends_when_feedback_is_empty(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer, checkpointer
):
    _loop_id, loop_path = _seed(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    db_settings.agent_max_rounds = 3
    fake_scene_analyzer.set_tool_call_turns(_three_calls("a", "b", "c") * 3)

    graph = build_agent_graph(db_conn, fake_embedder, fake_scene_analyzer, db_settings, checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    graph.invoke(initial_state(str(loop_path), db_settings), config)
    graph.invoke(Command(resume=""), config)

    assert graph.get_state(config).next == ()


def test_agent_graph_fallback_path_used_when_model_never_calls_tool(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer, checkpointer
):
    _loop_id, loop_path = _seed(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns([[], [], [], [], [], []])

    graph = build_agent_graph(db_conn, fake_embedder, fake_scene_analyzer, db_settings, checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    state = graph.invoke(initial_state(str(loop_path), db_settings), config)

    assert len(state["candidates"]) == 3
    assert state["tool_calls_total"] == 0
    assert state["fallback_used_total"] == 3
    assert fake_scene_analyzer.compose_calls == 3


def test_agent_graph_session_resumes_by_thread_id_after_restart(
    db_conn,
    db_settings,
    get_silent_loop,
    get_tone_track,
    tmp_path,
    fake_embedder,
    fake_scene_analyzer,
    test_database_url,
):
    """Новый инстанс чекпойнтера/графа (как будто процесс перезапустился) с тем
    же thread_id должен увидеть то же состояние и уметь продолжить с feedback."""
    from langgraph.checkpoint.postgres import PostgresSaver

    _loop_id, loop_path = _seed(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    db_settings.agent_max_rounds = 2
    fake_scene_analyzer.set_tool_call_turns(_three_calls("a", "b", "c") * 3)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string(test_database_url) as checkpointer_1:
        checkpointer_1.setup()
        graph_1 = build_agent_graph(db_conn, fake_embedder, fake_scene_analyzer, db_settings, checkpointer_1)
        first_state = graph_1.invoke(initial_state(str(loop_path), db_settings), config)

    with PostgresSaver.from_conn_string(test_database_url) as checkpointer_2:
        graph_2 = build_agent_graph(db_conn, fake_embedder, fake_scene_analyzer, db_settings, checkpointer_2)
        restored = graph_2.get_state(config)
        assert restored.next == ("feedback",)
        assert restored.values["candidates"] == first_state["candidates"]

        second_state = graph_2.invoke(Command(resume="brighter"), config)
        assert second_state["rounds"] == 1


def test_agent_graph_excludes_track_rated_bad_for_this_loop_in_a_new_session(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer, checkpointer
):
    """rating='bad' на конкретном рендере исключает трек из поиска для того
    же лупа в НОВОЙ сессии (новый thread_id) — не только в пределах одного
    разговора, как rejected_track_ids (см. render.py::get_bad_rated_track_ids)."""
    _loop_id, loop_path = _seed(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns(_three_calls("a", "b", "c") * 3)

    graph = build_agent_graph(db_conn, fake_embedder, fake_scene_analyzer, db_settings, checkpointer)

    first_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    first = graph.invoke(initial_state(str(loop_path), db_settings), first_config)
    bad_track_id = first["candidates"][0]["id"]
    render_id = first["render_ids"][0]
    set_render_rating(db_conn, render_id, "bad")

    second_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    second = graph.invoke(initial_state(str(loop_path), db_settings), second_config)

    assert bad_track_id not in {c["id"] for c in second["candidates"]}


def test_eval_graph_stops_after_rerank_without_rendering(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    _loop_id, loop_path = _seed(db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns(_three_calls("a", "b", "c"))

    graph = build_eval_graph(db_conn, fake_embedder, fake_scene_analyzer, db_settings)
    state = graph.invoke(initial_state(str(loop_path), db_settings))

    assert len(state["candidates"]) == 3
    assert "output_paths" not in state
    render_count = db_conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0]
    assert render_count == 0
