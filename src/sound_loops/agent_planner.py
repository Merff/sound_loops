"""Узел plan: поиск — не шаг пайплайна, а инструмент, который модель вызывает сама, пока не наберёт
settings.agent_slot_count разных запросов. Резервный путь (модель не
вызвала инструмент) переиспользует уже проверенный compose_music_query
(vlm.py) вместо разбора свободного текста ответа модели — парсинг текста
хрупкий.

Дедупликация треков МЕЖДУ слотами одного круга здесь не делается — это
задача узла rerank (agent_graph.py), у которого есть все три пула сразу.
Здесь exclude_track_ids только про уже показанные в ПРЕДЫДУЩИХ кругах
треки (агент не должен предлагать то, что уже отвергли).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import psycopg
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from sound_loops.config import Settings
from sound_loops.embeddings import Embedder
from sound_loops.rerank import RERANK_POOL_SIZE
from sound_loops.search import SearchResult, search_tracks_filtered
from sound_loops.vlm import LIBRARY_GENRES_WITH_CHARACTER, SceneAnalyzer, SceneDescription

_PLAN_SYSTEM_PROMPT = (
    "You are picking background music for a silent video loop by calling the "
    "search_music tool. Your job is to gather {slot_count} DIFFERENT good "
    "candidate queries — call the tool once per query, using clearly "
    "different wording/genre each time, not near-duplicates. Each call "
    "returns a short list of matching tracks from the library.\n\n"
    f"The music library only has these genres, evenly represented: "
    f"{LIBRARY_GENRES_WITH_CHARACTER}. Never say orchestral, cinematic, "
    "soundtrack, or symphonic — those don't exist here.\n\n"
    "Describe only the music in a query — genre/instruments, tempo, mood — "
    "never the video's subject, people, animals, artist names or track "
    "titles. English only, 5 to 20 words per query.\n\n"
    "Once you have called the tool {slot_count} times, stop calling tools "
    "and reply with a one-sentence summary."
)

_RETRY_NUDGE = (
    "You must call search_music now — {remaining} more distinct quer{plural} "
    "needed. Do not answer in plain text."
)


class SearchMusicArgs(BaseModel):
    query: str = Field(
        description="5 to 20 word English description of the MUSIC itself (genre/instruments, "
        "tempo, mood) — never the video's subject, artist names or track titles."
    )
    tempo_min: float | None = Field(default=None, description="Minimum tempo in BPM, or omit for no lower bound.")
    tempo_max: float | None = Field(default=None, description="Maximum tempo in BPM, or omit for no upper bound.")
    vocals: Literal["instrumental", "with_vocals", "any"] = Field(
        default="any", description="Required vocals, or 'any' for no preference."
    )
    top_n: int = Field(default=RERANK_POOL_SIZE, ge=1, le=10, description="How many candidate tracks to retrieve.")


@dataclass(frozen=True)
class ToolCallRecord:
    query: str
    vocals: str
    results: list[SearchResult]


@dataclass(frozen=True)
class PlanOutcome:
    slots: list[tuple[str, list[SearchResult]]]  # (query, кандидаты) — длина == agent_slot_count
    tool_calls_made: int  # сколько раз модель сама вызвала инструмент
    fallback_used: int  # сколько раз сработал резервный путь (см. модуль docstring)


def _build_human_prompt(
    scene: SceneDescription,
    feedback_history: Sequence[str],
    rejected_summary: Sequence[str],
    slot_count: int,
) -> str:
    parts = [f"Scene: setting={scene.setting}, mood={', '.join(scene.mood)}, motion={scene.motion}."]
    if rejected_summary:
        rejected_lines = "\n".join(f"- {r}" for r in rejected_summary)
        parts.append(f"Already shown and rejected — don't repeat this vibe:\n{rejected_lines}")
    if feedback_history:
        parts.append("User feedback so far (most recent last):\n" + "\n".join(f"- {f}" for f in feedback_history))
    parts.append(f"Call search_music now to gather {slot_count} different candidate queries.")
    return "\n\n".join(parts)


def plan_tracks(
    analyzer: SceneAnalyzer,
    conn: psycopg.Connection,
    embedder: Embedder,
    settings: Settings,
    scene: SceneDescription,
    loop_duration_seconds: float,
    feedback_history: Sequence[str] = (),
    rejected_summary: Sequence[str] = (),
    exclude_track_ids: Sequence[int] = (),
) -> PlanOutcome:
    slot_count = settings.agent_slot_count
    calls_log: list[ToolCallRecord] = []

    def run_search(
        query: str,
        tempo_min: float | None = None,
        tempo_max: float | None = None,
        vocals: str = "any",
        top_n: int = RERANK_POOL_SIZE,
    ) -> str:
        top_n = max(top_n, settings.agent_search_pool_size)
        vocals_arg = None if vocals == "any" else vocals
        tempo_range = None if tempo_min is None and tempo_max is None else (tempo_min or 0.0, tempo_max or 400.0)

        results = search_tracks_filtered(
            conn, embedder, query, top_n,
            min_duration_seconds=loop_duration_seconds,
            tempo_range=tempo_range, vocals=vocals_arg,
            exclude_ids=exclude_track_ids,
        )
        if not results and (tempo_range is not None or vocals_arg is not None):
            # Одноразовое снятие фильтров, если они дали пусто
            results = search_tracks_filtered(
                conn, embedder, query, top_n,
                min_duration_seconds=loop_duration_seconds,
                exclude_ids=exclude_track_ids,
            )

        calls_log.append(ToolCallRecord(query=query, vocals=vocals, results=results))
        if not results:
            return "No matching tracks found — try a different query."
        lines = [
            f"{r.title or r.path.rsplit('/', 1)[-1]!r} (similarity {r.similarity:.2f})" for r in results[:5]
        ]
        return f"Found {len(results)} candidate(s):\n" + "\n".join(lines)

    tool = StructuredTool.from_function(
        func=run_search,
        name="search_music",
        description="Search the music library for tracks matching a text description of the music.",
        args_schema=SearchMusicArgs,
    )

    model = analyzer.bind_tools([tool])
    messages: list = [
        SystemMessage(_PLAN_SYSTEM_PROMPT.format(slot_count=slot_count)),
        HumanMessage(_build_human_prompt(scene, feedback_history, rejected_summary, slot_count)),
    ]

    tool_calls_made = 0
    fallback_used = 0
    no_tool_streak = 0
    iterations = 0

    def use_fallback() -> None:
        # feedback_history передаётся и сюда — иначе "мрачнее"/"без вокала"
        # молча игнорировалось бы для слотов, закрытых этим путём.
        nonlocal fallback_used
        fallback_query = analyzer.compose_music_query(scene, feedback_history)
        run_search(fallback_query.query, vocals=fallback_query.vocals)
        fallback_used += 1

    while len(calls_log) < slot_count and iterations < settings.agent_max_plan_iterations:
        iterations += 1
        response: AIMessage = model.invoke(messages)
        messages.append(response)
        tool_calls = response.tool_calls or []

        if tool_calls:
            no_tool_streak = 0
            for call in tool_calls:
                if len(calls_log) >= slot_count:
                    break
                try:
                    content = tool.invoke(call["args"])
                except Exception as exc:  # noqa: BLE001 — модель могла прислать невалидные аргументы
                    content = f"Error: {exc}"
                tool_calls_made += 1
                messages.append(ToolMessage(content=str(content), tool_call_id=call["id"]))
            continue

        no_tool_streak += 1
        if no_tool_streak == 1:
            remaining = slot_count - len(calls_log)
            messages.append(
                HumanMessage(_RETRY_NUDGE.format(remaining=remaining, plural="ies" if remaining > 1 else "y"))
            )
            continue

        # Второй раз подряд без вызова инструмента — резервный путь.
        use_fallback()
        no_tool_streak = 0

    # Потолок ходов исчерпан, а слотов всё ещё не хватает — молча отдавать
    # меньше settings.agent_slot_count кандидатов нельзя (см. docstring),
    # достающие слоты закрываются тем же резервным путём.
    while len(calls_log) < slot_count:
        use_fallback()

    slots = [(rec.query, rec.results) for rec in calls_log[:slot_count]]
    return PlanOutcome(slots=slots, tool_calls_made=tool_calls_made, fallback_used=fallback_used)
