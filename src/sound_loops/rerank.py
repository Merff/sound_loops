"""Переранжирование кандидатов моделью (итерация 4, docs/sound_loops-iteration-4.md):
RAG в чистом виде — контекст (кандидаты с их атрибутами из базы) передаётся
модели вместе с описанием сцены, решение остаётся за ней. Топ-3, не топ-10 —
на 4B-модели длинный список кандидатов ухудшает выбор.

Не считаем заранее, что это помогает: маленькая модель может проиграть
простому топ-1 по вектору — это нормальный результат, его меряет eval-run
(--rerank), не эта функция.
"""

from __future__ import annotations

from dataclasses import dataclass

from sound_loops.search import SearchResult
from sound_loops.vlm import SceneAnalyzer, SceneDescription

RERANK_POOL_SIZE = 3


@dataclass(frozen=True)
class RerankResult:
    reordered: list[SearchResult]  # выбор модели первым, остальные — в исходном порядке
    chosen_rank: int  # 1-based номер выбранного кандидата в исходном списке
    reasoning: str


def clamp_candidate_index(index: int, n_candidates: int) -> int:
    """Модель иногда возвращает номер вне диапазона — не падать на этом,
    брать топ-1 по вектору как безопасный дефолт."""
    if 1 <= index <= n_candidates:
        return index
    return 1


def _describe_candidate(rank: int, candidate: SearchResult) -> str:
    title = candidate.title or candidate.path.rsplit("/", 1)[-1]
    tempo = f"{candidate.tempo_bpm:.0f} BPM" if candidate.tempo_bpm is not None else "tempo unknown"
    tags = candidate.tags or {}
    mood = ", ".join(tags.get("mood", {})) or "unknown"
    genre = ", ".join(tags.get("genre", {})) or "unknown"
    vocal_scores = tags.get("vocals", {})
    vocals = max(vocal_scores, key=vocal_scores.get) if vocal_scores else "unknown"
    return (
        f"{rank}. {title!r} — similarity {candidate.similarity:.3f}, {tempo}, "
        f"mood: {mood}, genre/instruments: {genre}, vocals: {vocals}"
    )


def rerank_candidates(
    analyzer: SceneAnalyzer, scene: SceneDescription, candidates: list[SearchResult]
) -> RerankResult:
    """candidates — уже отобранный пул (top RERANK_POOL_SIZE), не весь топ поиска."""
    if not candidates:
        raise ValueError("нечего переранжировать — пустой список кандидатов")

    descriptions = [_describe_candidate(rank, c) for rank, c in enumerate(candidates, start=1)]
    choice = analyzer.rerank(scene, descriptions)
    chosen_rank = clamp_candidate_index(choice.candidate_index, len(candidates))

    chosen = candidates[chosen_rank - 1]
    rest = [c for rank, c in enumerate(candidates, start=1) if rank != chosen_rank]
    return RerankResult(reordered=[chosen, *rest], chosen_rank=chosen_rank, reasoning=choice.reasoning)
