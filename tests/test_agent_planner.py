from pgvector.psycopg import register_vector

from sound_loops.agent_planner import plan_tracks
from sound_loops.config import Settings
from sound_loops.embeddings import normalize
from sound_loops.vlm import SceneDescription


def _insert_track(conn, embedder, path: str) -> int:
    register_vector(conn)
    vector = normalize(embedder.embed_texts([path]))[0]
    row = conn.execute(
        """
        INSERT INTO tracks (path, duration_seconds, embedding, embedding_model)
        VALUES (%s, %s, %s, %s) RETURNING id
        """,
        (path, 20.0, vector, "fake"),
    ).fetchone()
    conn.commit()
    return row[0]


def _seed_tracks(conn, embedder, n: int = 6) -> None:
    for i in range(n):
        _insert_track(conn, embedder, f"track_{i}.mp3")


SCENE = SceneDescription(setting="combat", motion="chaotic", mood=["tense", "aggressive"])


def test_plan_tracks_uses_model_tool_calls_when_it_calls_all_at_once(db_conn, fake_embedder, fake_scene_analyzer):
    _seed_tracks(db_conn, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns(
        [
            [
                {"query": "aggressive rock with distorted guitars"},
                {"query": "fast electronic with heavy bass"},
                {"query": "chaotic hip-hop with sharp percussion"},
            ]
        ]
    )

    outcome = plan_tracks(
        fake_scene_analyzer, db_conn, fake_embedder, Settings(database_url="postgresql://x/x"), SCENE, 5.0
    )

    assert len(outcome.slots) == 3
    assert outcome.tool_calls_made == 3
    assert outcome.fallback_used == 0
    assert fake_scene_analyzer.compose_calls == 0
    queries = [q for q, _pool in outcome.slots]
    assert queries == [
        "aggressive rock with distorted guitars",
        "fast electronic with heavy bass",
        "chaotic hip-hop with sharp percussion",
    ]


def test_plan_tracks_handles_one_call_per_turn(db_conn, fake_embedder, fake_scene_analyzer):
    _seed_tracks(db_conn, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns(
        [
            [{"query": "slow dreamy ambient with soft piano"}],
            [{"query": "epic orchestral-adjacent folk with strings"}],
            [{"query": "melancholic instrumental with warm pads"}],
        ]
    )

    outcome = plan_tracks(
        fake_scene_analyzer, db_conn, fake_embedder, Settings(database_url="postgresql://x/x"), SCENE, 5.0
    )

    assert len(outcome.slots) == 3
    assert outcome.tool_calls_made == 3
    assert outcome.fallback_used == 0


def test_plan_tracks_falls_back_when_model_never_calls_tool(db_conn, fake_embedder, fake_scene_analyzer):
    """Модель ни разу не вызвала инструмент (пустые ходы) — после повторной
    настойчивой попытки включается резервный путь на весь набор слотов."""
    _seed_tracks(db_conn, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns([[], [], [], [], [], []])

    outcome = plan_tracks(
        fake_scene_analyzer, db_conn, fake_embedder, Settings(database_url="postgresql://x/x"), SCENE, 5.0
    )

    assert len(outcome.slots) == 3
    assert outcome.tool_calls_made == 0
    assert outcome.fallback_used == 3
    assert fake_scene_analyzer.compose_calls == 3


def test_plan_tracks_fallback_passes_feedback_history_to_compose_music_query(
    db_conn, fake_embedder, fake_scene_analyzer
):
    """Регрессия: резервный путь раньше звал compose_music_query(scene) без
    feedback_history — «мрачнее»/«без вокала» молча игнорировалось для
    слотов, закрытых этим путём."""
    _seed_tracks(db_conn, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns([[], [], [], [], [], []])

    plan_tracks(
        fake_scene_analyzer,
        db_conn,
        fake_embedder,
        Settings(database_url="postgresql://x/x"),
        SCENE,
        5.0,
        feedback_history=["darker, no vocals"],
    )

    assert fake_scene_analyzer.last_feedback_history == ["darker, no vocals"]


def test_plan_tracks_fallback_fills_only_missing_slots(db_conn, fake_embedder, fake_scene_analyzer):
    """Модель вызвала инструмент один раз, потом дважды подряд промолчала —
    резервный путь достаёт только недостающие 2 слота, не все 3."""
    _seed_tracks(db_conn, fake_embedder)
    fake_scene_analyzer.set_tool_call_turns([[{"query": "aggressive rock with distorted guitars"}], [], []])

    outcome = plan_tracks(
        fake_scene_analyzer, db_conn, fake_embedder, Settings(database_url="postgresql://x/x"), SCENE, 5.0
    )

    assert len(outcome.slots) == 3
    assert outcome.tool_calls_made == 1
    assert outcome.fallback_used == 2
    assert outcome.slots[0][0] == "aggressive rock with distorted guitars"


def test_plan_tracks_excludes_previously_rejected_track_ids(db_conn, fake_embedder, fake_scene_analyzer):
    _seed_tracks(db_conn, fake_embedder, n=4)
    all_ids = [r[0] for r in db_conn.execute("SELECT id FROM tracks ORDER BY id").fetchall()]
    excluded = all_ids[:3]  # оставляем только один невозвращённый трек

    fake_scene_analyzer.set_tool_call_turns([[{"query": "aggressive rock with distorted guitars", "top_n": 10}]])

    outcome = plan_tracks(
        fake_scene_analyzer,
        db_conn,
        fake_embedder,
        Settings(database_url="postgresql://x/x"),
        SCENE,
        5.0,
        exclude_track_ids=excluded,
    )

    _query, pool = outcome.slots[0]
    assert all(r.id not in excluded for r in pool)
    assert {r.id for r in pool} == {all_ids[3]}


def test_plan_tracks_respects_max_plan_iterations(db_conn, fake_embedder, fake_scene_analyzer):
    """Даже если модель бесконечно вызывает инструмент по одному разу за ход,
    цикл не крутится вечно — после потолка ходов включается резервный путь."""
    _seed_tracks(db_conn, fake_embedder)
    settings = Settings(database_url="postgresql://x/x", agent_max_plan_iterations=2)
    # Каждый ход — по одному вызову инструмента (не 0), значит "no_tool_streak"
    # никогда не триггерится, а потолок итераций — единственное, что остановит цикл.
    fake_scene_analyzer.set_tool_call_turns([[{"query": f"query variant {i}"}] for i in range(10)])

    outcome = plan_tracks(fake_scene_analyzer, db_conn, fake_embedder, settings, SCENE, 5.0)

    assert len(outcome.slots) == 3
    assert outcome.tool_calls_made == 2
    assert outcome.fallback_used == 1
