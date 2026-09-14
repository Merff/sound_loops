"""Полная цепочка итерации 2: кадры лупа -> VLM-описание сцены -> музыкальный
запрос -> CLAP-поиск -> сборка превью (docs/sound_loops-iteration-2.md).

По структуре — аналог render_once из render.py, только источник трека не
random(), а поиск по смыслу. Рендер помечается ссылкой на video_analyses
(analysis_id) и сохранённым music_query — по ним видно, что превью собрано
этой цепочкой, а не случайным baseline'ом итерации 0.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import psycopg

from sound_loops.analysis import AnalysisRecord, analyze_loop
from sound_loops.config import Settings
from sound_loops.embeddings import Embedder
from sound_loops.ffmpeg_utils import extract_audio_segment, extract_frames, mux_loop_with_audio
from sound_loops.render import LoopRow, get_loop_by_path, get_random_loop, pick_random_start
from sound_loops.search import SearchResult, search_tracks
from sound_loops.vlm import SceneAnalyzer


@dataclass(frozen=True)
class MatchResult:
    loop: LoopRow
    analysis: AnalysisRecord
    analysis_cached: bool
    music_query: str
    candidates: list[SearchResult]
    output_path: Path


def match_once(
    conn: psycopg.Connection,
    analyzer: SceneAnalyzer,
    embedder: Embedder,
    settings: Settings,
    loop_path: Path | None = None,
) -> MatchResult:
    loop = get_loop_by_path(conn, loop_path, settings) if loop_path else get_random_loop(conn)

    def get_frames() -> list[bytes]:
        return extract_frames(
            Path(loop.path),
            loop.duration_seconds,
            settings.vlm_frame_count,
            settings.vlm_frame_max_side,
        )

    analysis, cached = analyze_loop(conn, analyzer, loop.id, get_frames)
    music_query = analyzer.compose_music_query(analysis.scene).query

    candidates = search_tracks(conn, embedder, music_query, top_n=3, min_duration_seconds=loop.duration_seconds)
    best = candidates[0]
    track_duration = conn.execute(
        "SELECT duration_seconds FROM tracks WHERE id = %s", (best.id,)
    ).fetchone()[0]
    start_seconds = pick_random_start(track_duration, loop.duration_seconds)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{Path(loop.path).stem}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = settings.output_dir / output_name

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp_audio_path = Path(tmp.name)
    try:
        extract_audio_segment(
            track_path=Path(best.path),
            start_seconds=start_seconds,
            duration_seconds=loop.duration_seconds,
            fade_seconds=settings.fade_seconds,
            output_path=tmp_audio_path,
        )
        mux_loop_with_audio(Path(loop.path), tmp_audio_path, output_path)
    finally:
        tmp_audio_path.unlink(missing_ok=True)

    conn.execute(
        """
        INSERT INTO renders
            (loop_id, track_id, start_seconds, output_path, duration_seconds, analysis_id, music_query)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (loop.id, best.id, start_seconds, str(output_path), loop.duration_seconds, analysis.id, music_query),
    )
    conn.commit()

    return MatchResult(
        loop=loop,
        analysis=analysis,
        analysis_cached=cached,
        music_query=music_query,
        candidates=candidates,
        output_path=output_path,
    )
