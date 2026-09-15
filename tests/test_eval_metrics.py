from sound_loops.eval_metrics import best_rank, hit_at_k, mood_overlap


def test_mood_overlap_full_match():
    assert mood_overlap(["calm", "dreamy"], ["calm", "dreamy"]) == 1.0


def test_mood_overlap_no_overlap():
    assert mood_overlap(["calm"], ["aggressive"]) == 0.0


def test_mood_overlap_partial():
    assert mood_overlap(["calm", "dreamy"], ["calm", "epic"]) == 1 / 3


def test_mood_overlap_ignores_duplicates():
    assert mood_overlap(["calm", "calm"], ["calm"]) == 1.0


def test_best_rank_finds_first_match():
    assert best_rank(["a", "b", "c"], ["c", "z"]) == 3


def test_best_rank_none_when_not_found():
    assert best_rank(["a", "b"], ["z"]) is None


def test_best_rank_empty_ranked_list():
    assert best_rank([], ["a"]) is None


def test_hit_at_k_true_within_k():
    assert hit_at_k(["a", "b", "c"], ["c"], k=5) is True


def test_hit_at_k_false_beyond_k():
    assert hit_at_k(["a", "b", "c"], ["c"], k=2) is False


def test_hit_at_1_only_checks_top_result():
    assert hit_at_k(["a", "b"], ["a"], k=1) is True
    assert hit_at_k(["a", "b"], ["b"], k=1) is False
