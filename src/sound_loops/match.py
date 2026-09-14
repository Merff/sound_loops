"""Подбор и наложение трека под уже проанализированный луп (шаги B/C/D
итерации 2, docs/sound_loops-iteration-2.md): музыкальный запрос ->
CLAP-поиск -> сборка превью.

Шаг A (VLM-описание сцены + motion) — отдельная команда `analyze`
(см. analysis.py::analyze_loop_by_path и cli.py). Здесь он не запускается:
match полагается на то, что анализ уже лежит в video_analyses — это и есть
смысл разделения на две команды, а не просто их совместный запуск под
одной крышей. Рендер помечается ссылкой на video_analyses (analysis_id) и
сохранённым music_query — по ним видно, что превью собрано этой цепочкой,
а не случайным baseline'ом итерации 0.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import psycopg

from sound_loops.analysis import AnalysisRecord, get_cached_analysis
from sound_loops.config import Settings
from sound_loops.embeddings import Embedder
from sound_loops.ffmpeg_utils import extract_audio_segment, mux_loop_with_audio
from sound_loops.render import LoopRow, get_loop_by_path, get_random_loop, pick_random_start
from sound_loops.search import SearchResult, search_tracks
from sound_loops.vlm import SceneAnalyzer


class MatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatchResult:
    loop: LoopRow
    analysis: AnalysisRecord
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

    analysis = get_cached_analysis(conn, loop.id, analyzer.model_id, analyzer.prompt_version)
    if analysis is None:
        raise MatchError(
            f"для лупа {loop.path} нет сохранённого анализа сцены "
            f"(модель {analyzer.model_id!r}, промпт {analyzer.prompt_version!r}) — "
            f"сначала запустите: sound-loops analyze --loop {loop.path}"
        )

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
        music_query=music_query,
        candidates=candidates,
        output_path=output_path,
    )
