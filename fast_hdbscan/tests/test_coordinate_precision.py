import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from fast_hdbscan import HDBSCAN
from fast_hdbscan.hdbscan import compute_minimum_spanning_tree
from fast_hdbscan.numba_kdtree import build_kdtree, parallel_tree_query


def _data():
    return make_blobs(
        n_samples=300,
        centers=[[-4, 0], [0, 0], [4, 0]],
        cluster_std=0.15,
        random_state=17,
    )[0]


@pytest.mark.parametrize("algorithm", ["boruvka", "kruskal"])
@pytest.mark.parametrize("reproducible", [False, True])
@pytest.mark.parametrize("scenario", ["near", "translated", "outlier"])
def test_float64_mst_matches_distance_oracle(algorithm, reproducible, scenario):
    X = _data()
    if scenario == "translated":
        X += np.array([1e8, -1e8])
    elif scenario == "outlier":
        X = np.vstack([X, [1e10, 1e10]])
    original = X.copy()
    distances = cdist(X, X)
    expected_core = np.sort(distances, axis=1)[:, 5]
    reachability = np.maximum(
        np.maximum(distances, expected_core[:, None]), expected_core[None, :]
    )
    expected_edges = np.sort(minimum_spanning_tree(reachability).data)
    edges, _, core = compute_minimum_spanning_tree(
        X, min_samples=5, algorithm=algorithm, reproducible=reproducible
    )
    assert core.dtype == np.float64
    np.testing.assert_allclose(core, expected_core, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.sort(edges[:, 2]), expected_edges, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tree_preserves_floating_precision(dtype):
    X = _data().astype(dtype)
    original = X.copy()
    tree = build_kdtree(X)
    assert not np.shares_memory(tree.data, X)
    assert tree.data.dtype == dtype
    assert tree.node_bounds.dtype == dtype
    assert tree.radius.dtype == dtype
    np.testing.assert_array_equal(tree.data, X)
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("algorithm", ["boruvka", "kruskal"])
@pytest.mark.parametrize("weighted", [False, True])
def test_fit_large_translation_preserves_labels(algorithm, weighted):
    X = _data()
    shifted = X + np.array([1e8, -1e8])
    original = shifted.copy()
    params = dict(
        min_cluster_size=20,
        min_samples=5,
        reproducible=True,
        algorithm=algorithm,
        knn_k=15 if weighted and algorithm == "kruskal" else None,
    )
    weights = np.ones(len(X), dtype=np.float32) if weighted else None
    baseline = HDBSCAN(**params).fit(X, sample_weight=weights)
    translated = HDBSCAN(**params).fit(shifted, sample_weight=weights)
    assert translated._raw_data.dtype == np.float64
    assert translated._core_distances.dtype == np.float64
    assert adjusted_rand_score(baseline.labels_, translated.labels_) == 1.0
    np.testing.assert_array_equal(shifted, original)


@pytest.mark.parametrize("tree_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("query_dtype", [np.float32, np.float64])
def test_query_accepts_mixed_floating_precision(tree_dtype, query_dtype):
    X = _data().astype(tree_dtype)
    query = np.asfortranarray((X[:10] + 0.01).astype(query_dtype))
    original = query.copy()
    tree = build_kdtree(X)
    distances, indices = parallel_tree_query(tree, query, k=5)
    expected = np.sort(cdist(query.astype(tree_dtype), X), axis=1)[:, :5]
    assert distances.dtype == tree_dtype
    tolerance = 1e-6 if tree_dtype == np.float32 else 1e-12
    np.testing.assert_allclose(distances, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(query, original)
