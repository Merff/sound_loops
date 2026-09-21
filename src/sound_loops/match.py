"""Подбор и наложение трека под уже проанализированный луп: музыкальный запрос ->
CLAP-поиск -> сборка превью. use_filters/use_rerank переключают гибридный поиск и
переранжирование поверх того же пайплайна — конфигурация параметром
вызова, не правкой кода.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psycopg

from sound_loops.analysis import AnalysisRecord, get_cached_analysis
from sound_loops.config import Settings
from sound_loops.embeddings import Embedder
from sound_loops.filters import tempo_range_for_motion
from sound_loops.render import LoopRow, get_loop_by_path, get_random_loop, render_preview
from sound_loops.rerank import RERANK_POOL_SIZE, rerank_candidates
from sound_loops.search import SearchResult, search_tracks, search_tracks_hybrid
from sound_loops.vlm import SceneAnalyzer


class MatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManualMatchResult:
    loop: LoopRow
    query: str
    candidates: list[SearchResult]
    renders: list[tuple[int, Path]]


def manual_match(
    conn: psycopg.Connection,
    embedder: Embedder,
    settings: Settings,
    loop_path: Path,
    query: str,
    top_n: int,
) -> ManualMatchResult:
    """Прямой текстовый запрос пользователя вместо VLM-цепочки -> top_n треков CLAP-поиском -> рендер каждого.
    analysis_id остаётся NULL, music_query — текст пользователя"""
    loop = get_loop_by_path(conn, loop_path, settings)
    candidates = search_tracks(conn, embedder, query, top_n, min_duration_seconds=loop.duration_seconds)

    renders = []
    for candidate in candidates:
        track_duration = conn.execute(
            "SELECT duration_seconds FROM tracks WHERE id = %s", (candidate.id,)
        ).fetchone()[0]
        render_id, output_path = render_preview(
            conn, settings, loop, candidate.id, candidate.path, track_duration, None, query
        )
        renders.append((render_id, output_path))

    return ManualMatchResult(loop=loop, query=query, candidates=candidates, renders=renders)


@dataclass(frozen=True)
class MatchResult:
    loop: LoopRow
    analysis: AnalysisRecord
    music_query: str
    candidates: list[SearchResult]
    output_path: Path
    relaxed_filters: list[str]
    rerank_reasoning: str | None


def match_once(
    conn: psycopg.Connection,
    analyzer: SceneAnalyzer,
    embedder: Embedder,
    settings: Settings,
    loop_path: Path | None = None,
    use_filters: bool = False,
    use_rerank: bool = False,
) -> MatchResult:
    loop = get_loop_by_path(conn, loop_path, settings) if loop_path else get_random_loop(conn)

    analysis = get_cached_analysis(conn, loop.id, analyzer.model_id, analyzer.prompt_version)
    if analysis is None:
        raise MatchError(
            f"для лупа {loop.path} нет сохранённого анализа сцены "
            f"(модель {analyzer.model_id!r}, промпт {analyzer.prompt_version!r}) — "
            f"сначала запустите: sound-loops analyze --loop {loop.path}"
        )

    music_query = analyzer.compose_music_query(analysis.scene)
    top_n = RERANK_POOL_SIZE if use_rerank else 3

    relaxed_filters: list[str] = []
    if use_filters:
        tempo_range = tempo_range_for_motion(analysis.scene.motion)
        candidates, relaxed_filters = search_tracks_hybrid(
            conn, embedder, music_query.query, tempo_range, music_query.vocals, top_n, loop.duration_seconds
        )
    else:
        candidates = search_tracks(
            conn, embedder, music_query.query, top_n, min_duration_seconds=loop.duration_seconds
        )

    rerank_reasoning = None
    if use_rerank:
        rerank_result = rerank_candidates(analyzer, analysis.scene, candidates)
        candidates = rerank_result.reordered
        rerank_reasoning = rerank_result.reasoning

    best = candidates[0]
    track_duration = conn.execute(
        "SELECT duration_seconds FROM tracks WHERE id = %s", (best.id,)
    ).fetchone()[0]
    _render_id, output_path = render_preview(
        conn, settings, loop, best.id, best.path, track_duration, analysis.id, music_query.query
    )

    return MatchResult(
        loop=loop,
        analysis=analysis,
        music_query=music_query.query,
        candidates=candidates,
        output_path=output_path,
        relaxed_filters=relaxed_filters,
        rerank_reasoning=rerank_reasoning,
    )
