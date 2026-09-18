"""Граф LangGraph (итерация 5, docs/sound_loops-iteration-5.md): линейная
цепочка итераций 2-4 (analyze -> query -> search -> rerank -> render)
становится графом с явным состоянием и одним условным ребром — из
feedback либо назад в plan, либо в конец.

Состояние хранит только то, что можно (де)сериализовать чекпойнтером:
SceneDescription и SearchResult лежат как dict/список dict, не как сами
объекты — так надёжнее переживает Postgres-чекпойнтер разных версий
langgraph, чем полагаться на то, что он умеет пиклить наши dataclass/pydantic
типы напрямую.

build_agent_graph — интерактивный граф для UI (с чекпойнтером и остановкой
на feedback). build_eval_graph — тот же analyze/plan/rerank без render и
feedback, для eval_run.py: харнесс меряет только первый проход, рендерить
аудио и останавливаться в ожидании пользователя ему незачем.
"""

from __future__ import annotations

import operator
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, TypedDict

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from sound_loops.agent_planner import plan_tracks
from sound_loops.analysis import analyze_loop
from sound_loops.config import Settings
from sound_loops.embeddings import Embedder
from sound_loops.ffmpeg_utils import extract_frames
from sound_loops.motion import estimate_motion
from sound_loops.render import LoopRow, get_bad_rated_track_ids, get_loop_by_path, render_preview
from sound_loops.rerank import RERANK_POOL_SIZE, rerank_candidates
from sound_loops.search import SearchResult
from sound_loops.vlm import Motion, SceneAnalyzer, SceneDescription


class AgentState(TypedDict, total=False):
    loop_path: str
    loop_id: int
    loop_duration_seconds: float
    analysis_id: int
    scene: dict  # SceneDescription.model_dump()

    plan_pools: list[dict]  # [{"query": str, "candidates": [dict(SearchResult), ...]}, ...] — только в пределах круга

    candidates: list[dict]  # dict(SearchResult) выбранных на этот круг треков, длина agent_slot_count
    slot_queries: list[str]
    slot_reasoning: list[str]
    output_paths: list[str]
    render_ids: list[int]  # id строк renders этого круга — для оценки (rating) в UI

    query_log: Annotated[list[dict], operator.add]  # вся история кругов — для UI и для prompt'а plan
    all_render_ids: Annotated[list[int], operator.add]  # render_ids всех кругов сессии — для очистки при завершении
    rejected_track_ids: Annotated[list[int], operator.add]  # id треков, уже показанных в прошлых кругах
    feedback_history: Annotated[list[str], operator.add]
    tool_calls_total: Annotated[int, operator.add]
    fallback_used_total: Annotated[int, operator.add]

    last_feedback: str
    rounds: int
    max_rounds: int


def initial_state(loop_path: str, settings: Settings) -> AgentState:
    return AgentState(
        loop_path=loop_path,
        feedback_history=[],
        rejected_track_ids=[],
        query_log=[],
        all_render_ids=[],
        tool_calls_total=0,
        fallback_used_total=0,
        last_feedback="",
        rounds=0,
        max_rounds=settings.agent_max_rounds,
    )


def _track_duration(conn: psycopg.Connection, track_id: int) -> float:
    return conn.execute("SELECT duration_seconds FROM tracks WHERE id = %s", (track_id,)).fetchone()[0]


def _make_analyze_node(conn: psycopg.Connection, analyzer: SceneAnalyzer, settings: Settings):
    def analyze_node(state: AgentState) -> dict:
        loop = get_loop_by_path(conn, Path(state["loop_path"]), settings)

        def get_frames() -> list[bytes]:
            return extract_frames(
                Path(loop.path), loop.duration_seconds, settings.vlm_frame_count, settings.vlm_frame_max_side
            )

        def get_motion() -> Motion:
            return estimate_motion(Path(loop.path), settings.motion_sample_fps, settings.motion_frame_size)

        record, _cached = analyze_loop(conn, analyzer, loop.id, get_frames, get_motion)
        return {
            "loop_id": loop.id,
            "loop_duration_seconds": loop.duration_seconds,
            "analysis_id": record.id,
            "scene": record.scene.model_dump(),
        }

    return analyze_node


def _make_plan_node(conn: psycopg.Connection, embedder: Embedder, analyzer: SceneAnalyzer, settings: Settings):
    def plan_node(state: AgentState) -> dict:
        scene = SceneDescription.model_validate(state["scene"])
        rejected_summary = [f"round {e['round']}: {e['query']!r}" for e in state.get("query_log", [])]

        # Дедуп не только в пределах этой сессии (rejected_track_ids), но и
        # то, что пользователь когда-то отметил "плохо" для этого же лупа в
        # прошлых сессиях (rating в renders, итерация 5) — постоянно, не
        # только на время текущего разговора.
        exclude_ids = set(state.get("rejected_track_ids", [])) | set(
            get_bad_rated_track_ids(conn, state["loop_id"])
        )

        outcome = plan_tracks(
            analyzer,
            conn,
            embedder,
            settings,
            scene,
            state["loop_duration_seconds"],
            feedback_history=state.get("feedback_history", []),
            rejected_summary=rejected_summary,
            exclude_track_ids=list(exclude_ids),
        )
        pools = [{"query": query, "candidates": [asdict(c) for c in pool]} for query, pool in outcome.slots]
        return {
            "plan_pools": pools,
            "tool_calls_total": outcome.tool_calls_made,
            "fallback_used_total": outcome.fallback_used,
        }

    return plan_node


def _make_rerank_node(analyzer: SceneAnalyzer):
    def rerank_node(state: AgentState) -> dict:
        scene = SceneDescription.model_validate(state["scene"])
        used_ids: set[int] = set()
        chosen: list[tuple[SearchResult, str, str]] = []

        for pool_info in state["plan_pools"]:
            pool = [SearchResult(**c) for c in pool_info["candidates"]]
            available = [c for c in pool if c.id not in used_ids] or pool
            if not available:
                continue  # пустой пул (пустая библиотека) — пропускаем слот, а не падаем

            if len(available) == 1:
                winner, reasoning = available[0], "единственный оставшийся кандидат"
            else:
                result = rerank_candidates(analyzer, scene, available[:RERANK_POOL_SIZE])
                winner, reasoning = result.reordered[0], result.reasoning

            used_ids.add(winner.id)
            chosen.append((winner, reasoning, pool_info["query"]))

        return {
            "candidates": [asdict(c) for c, _reasoning, _query in chosen],
            "slot_reasoning": [reasoning for _c, reasoning, _query in chosen],
            "slot_queries": [query for _c, _reasoning, query in chosen],
        }

    return rerank_node


def _make_render_node(conn: psycopg.Connection, settings: Settings):
    def render_node(state: AgentState) -> dict:
        loop = LoopRow(id=state["loop_id"], path=state["loop_path"], duration_seconds=state["loop_duration_seconds"])
        round_number = state["rounds"]

        output_paths: list[str] = []
        render_ids: list[int] = []
        query_log_entries: list[dict] = []
        for candidate, query, reasoning in zip(
            state["candidates"], state["slot_queries"], state["slot_reasoning"], strict=True
        ):
            duration = _track_duration(conn, candidate["id"])
            render_id, output_path = render_preview(
                conn, settings, loop, candidate["id"], candidate["path"], duration, state["analysis_id"], query
            )
            output_paths.append(str(output_path))
            render_ids.append(render_id)
            query_log_entries.append(
                {
                    "round": round_number,
                    "query": query,
                    "track_id": candidate["id"],
                    "track_title": candidate.get("title") or candidate["path"],
                    "reasoning": reasoning,
                }
            )

        return {
            "output_paths": output_paths,
            "render_ids": render_ids,
            "all_render_ids": render_ids,
            "query_log": query_log_entries,
            "rejected_track_ids": [c["id"] for c in state["candidates"]],
        }

    return render_node


def feedback_node(state: AgentState) -> dict:
    """Останавливает граф штатным interrupt() и ждёт текст пользователя.
    Функция переисполняется целиком при resume (см. langgraph.types.interrupt),
    но до и после interrupt() тут нет побочных эффектов — безопасно."""
    text = interrupt(
        {
            "candidates": state["candidates"],
            "slot_queries": state["slot_queries"],
            "slot_reasoning": state["slot_reasoning"],
            "output_paths": state["output_paths"],
            "round": state["rounds"] + 1,
            "max_rounds": state["max_rounds"],
        }
    )
    return {
        "last_feedback": text or "",
        "feedback_history": [text] if text else [],
        "rounds": state["rounds"] + 1,
    }


def _route_after_feedback(state: AgentState) -> str:
    if state.get("last_feedback") and state["rounds"] < state["max_rounds"]:
        return "plan"
    return END


def build_agent_graph(
    conn: psycopg.Connection,
    embedder: Embedder,
    analyzer: SceneAnalyzer,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
):
    """Полный интерактивный граф: analyze -> plan -> rerank -> render ->
    feedback -> (plan | конец). Нужен чекпойнтер — без него interrupt() в
    feedback_node падает (см. langgraph.types.interrupt)."""
    graph = StateGraph(AgentState)
    graph.add_node("analyze", _make_analyze_node(conn, analyzer, settings))
    graph.add_node("plan", _make_plan_node(conn, embedder, analyzer, settings))
    graph.add_node("rerank", _make_rerank_node(analyzer))
    graph.add_node("render", _make_render_node(conn, settings))
    graph.add_node("feedback", feedback_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "rerank")
    graph.add_edge("rerank", "render")
    graph.add_edge("render", "feedback")
    graph.add_conditional_edges("feedback", _route_after_feedback, {"plan": "plan", END: END})

    return graph.compile(checkpointer=checkpointer)


def build_eval_graph(conn: psycopg.Connection, embedder: Embedder, analyzer: SceneAnalyzer, settings: Settings):
    """analyze -> plan -> rerank -> конец, без render/feedback: eval_run.py
    меряет только первый проход и не должен ни рендерить mp4, ни ждать
    ввода (см. docs/sound_loops-iteration-5.md, раздел «Эвалы»)."""
    graph = StateGraph(AgentState)
    graph.add_node("analyze", _make_analyze_node(conn, analyzer, settings))
    graph.add_node("plan", _make_plan_node(conn, embedder, analyzer, settings))
    graph.add_node("rerank", _make_rerank_node(analyzer))

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "rerank")
    graph.add_edge("rerank", END)

    return graph.compile()
