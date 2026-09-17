import pytest

from sound_loops.rerank import RERANK_POOL_SIZE, clamp_candidate_index, rerank_candidates
from sound_loops.search import SearchResult
from sound_loops.vlm import RerankChoice, SceneDescription


def _candidate(path: str, similarity: float) -> SearchResult:
    return SearchResult(id=1, path=path, title=None, artist=None, genre=None, similarity=similarity)


class FakeRerankAnalyzer:
    model_id = "fake-rerank-analyzer"
    prompt_version = "fake-v1"

    def __init__(self, candidate_index: int, reasoning: str = "fits the mood") -> None:
        self._candidate_index = candidate_index
        self._reasoning = reasoning
        self.rerank_calls = 0
        self.last_candidate_descriptions: list[str] = []

    def rerank(self, scene, candidate_descriptions):
        self.rerank_calls += 1
        self.last_candidate_descriptions = list(candidate_descriptions)
        return RerankChoice(candidate_index=self._candidate_index, reasoning=self._reasoning)


def test_clamp_candidate_index_within_range_unchanged():
    assert clamp_candidate_index(2, 3) == 2


def test_clamp_candidate_index_at_boundaries_unchanged():
    assert clamp_candidate_index(1, 3) == 1
    assert clamp_candidate_index(3, 3) == 3


def test_clamp_candidate_index_below_range_falls_back_to_top_1():
    assert clamp_candidate_index(0, 3) == 1
    assert clamp_candidate_index(-5, 3) == 1


def test_clamp_candidate_index_above_range_falls_back_to_top_1():
    assert clamp_candidate_index(4, 3) == 1
    assert clamp_candidate_index(100, 3) == 1


def test_rerank_candidates_promotes_chosen_to_front():
    candidates = [_candidate("a.mp3", 0.9), _candidate("b.mp3", 0.8), _candidate("c.mp3", 0.7)]
    analyzer = FakeRerankAnalyzer(candidate_index=2)
    scene = SceneDescription(setting="domestic", motion="slow", mood=["calm"])

    result = rerank_candidates(analyzer, scene, candidates)

    assert [c.path for c in result.reordered] == ["b.mp3", "a.mp3", "c.mp3"]
    assert result.chosen_rank == 2
    assert result.reasoning == "fits the mood"
    assert analyzer.rerank_calls == 1


def test_rerank_candidates_keeps_order_when_top_candidate_chosen():
    candidates = [_candidate("a.mp3", 0.9), _candidate("b.mp3", 0.8)]
    analyzer = FakeRerankAnalyzer(candidate_index=1)
    scene = SceneDescription(setting="domestic", motion="slow", mood=["calm"])

    result = rerank_candidates(analyzer, scene, candidates)

    assert [c.path for c in result.reordered] == ["a.mp3", "b.mp3"]


def test_rerank_candidates_clamps_out_of_range_choice():
    candidates = [_candidate("a.mp3", 0.9), _candidate("b.mp3", 0.8)]
    analyzer = FakeRerankAnalyzer(candidate_index=99)
    scene = SceneDescription(setting="domestic", motion="slow", mood=["calm"])

    result = rerank_candidates(analyzer, scene, candidates)

    assert result.chosen_rank == 1
    assert [c.path for c in result.reordered] == ["a.mp3", "b.mp3"]


def test_rerank_candidates_rejects_empty_list():
    analyzer = FakeRerankAnalyzer(candidate_index=1)
    scene = SceneDescription(setting="domestic", motion="slow", mood=["calm"])

    with pytest.raises(ValueError):
        rerank_candidates(analyzer, scene, [])


def test_pool_size_is_three_not_ten():
    """Итерация 4 намеренно уменьшила пул до 3 (не 10) — 4B-модель хуже выбирает на длинных списках."""
    assert RERANK_POOL_SIZE == 3
