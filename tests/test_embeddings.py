import numpy as np

from sound_loops.embeddings import normalize


def test_normalize_produces_unit_vectors():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    result = normalize(vectors)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0], atol=1e-6)


def test_normalize_leaves_zero_vector_unchanged():
    vectors = np.array([[0.0, 0.0]], dtype=np.float32)
    result = normalize(vectors)
    np.testing.assert_array_equal(result, vectors)
