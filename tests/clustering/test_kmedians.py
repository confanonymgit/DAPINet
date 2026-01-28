import numpy as np
from sklearn.preprocessing import StandardScaler

from dapinet.clustering.k_medians import KMedians


def test_duplicate_points():
    # All points are identical; the center should match exactly and all labels should be the same
    X = np.array([[3, 3]] * 10)
    model = KMedians(n_clusters=1, n_init=5)
    model.fit(X)

    assert np.allclose(model.cluster_centers_[0], [3, 3]), "❌ Center must match duplicate point"
    assert np.all(model.labels_ == 0), "❌ All points should have the same label"


def test_high_dimensional_low_variance():
    # Data with 50 dimensions but variation only in one axis
    X = np.zeros((10, 50))
    X[:, 0] = np.linspace(0, 1, 10)

    model = KMedians(n_clusters=2, n_init=10)
    model.fit(X)

    # All other dimensions should remain zero in centers
    assert np.allclose(model.cluster_centers_[:, 1:], 0), "❌ Only the first dimension should vary"


def test_scale_variance():
    # Create dataset with high variance in Y-dimension, but clustering pattern on X
    X = np.array([[1, 1000], [2, 1000], [3, 1000], [10, 10], [11, 10], [12, 10]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMedians(n_clusters=2, n_init=50, random_state=0)
    model.fit(X_scaled)

    # Check labels: points 0–2 should be in same cluster, and 3–5 in the other
    labels = model.labels_
    cluster_1 = set(labels[:3])
    cluster_2 = set(labels[3:])

    assert len(cluster_1) == 1 and len(cluster_2) == 1 and cluster_1 != cluster_2, (
        f"❌ Points were not clustered based on X-axis despite normalization. Labels: {labels}"
    )


def test_clusters_more_than_unique_points():
    # Fewer unique points than requested clusters; should handle gracefully
    X = np.array([[1, 1], [2, 2], [3, 3]])
    try:
        model = KMedians(n_clusters=5, n_init=5)
        model.fit(X)
    except ValueError:
        pass  # acceptable if explicitly raised
    else:
        assert model.cluster_centers_.shape[0] <= 3, (
            "❌ Cannot create more clusters than unique data points"
        )


def test_rotational_invariance():
    # Rotating the data shouldn't affect label consistency
    X = np.array([[1, 0], [2, 0], [3, 0], [10, 0], [11, 0], [12, 0]])
    R = np.array([[0, 1], [1, 0]])  # simple 90-degree rotation
    X_rot = X @ R

    model1 = KMedians(n_clusters=2, n_init=10)
    model2 = KMedians(n_clusters=2, n_init=10)

    model1.fit(X)
    model2.fit(X_rot)

    labels_1 = np.sort(model1.labels_)
    labels_2 = np.sort(model2.labels_)
    np.testing.assert_array_equal(
        labels_1, labels_2, err_msg="❌ Labels should remain consistent after rotation"
    )


def test_seed_determinism():
    # Same random_state and data must result in identical outputs
    X = np.random.rand(20, 2)

    model1 = KMedians(n_clusters=3, n_init=10, random_state=0)
    model2 = KMedians(n_clusters=3, n_init=10, random_state=0)

    model1.fit(X)
    model2.fit(X)

    np.testing.assert_allclose(
        model1.cluster_centers_,
        model2.cluster_centers_,
        err_msg="❌ Centers should be deterministic",
    )
    np.testing.assert_array_equal(
        model1.labels_, model2.labels_, err_msg="❌ Labels should be deterministic"
    )


def test_deterministic_result_on_clear_data():
    # Simple 2-cluster dataset with clearly separable groups
    X = np.array(
        [
            [1, 2],
            [2, 1],
            [2, 2],  # cluster 1 → median = [2, 2]
            [8, 8],
            [9, 9],
            [8, 9],  # cluster 2 → median = [8, 9]
        ]
    )

    model = KMedians(n_clusters=2, n_init=10, random_state=0)
    model.fit(X)

    labels = model.labels_
    centers = model.cluster_centers_

    # Both clusters should have exactly 3 points
    counts = np.bincount(labels)
    assert np.all(counts == 3), f"❌ Expected two clusters of 3 points each, got: {counts}"

    # Check that centers match medians of their respective points
    cluster_0 = X[labels == 0]
    cluster_1 = X[labels == 1]
    true_median_0 = np.median(cluster_0, axis=0)
    true_median_1 = np.median(cluster_1, axis=0)

    def match_center(c):
        return np.allclose(c, true_median_0, atol=1e-6) or np.allclose(c, true_median_1, atol=1e-6)

    assert match_center(centers[0]), f"❌ First center is incorrect: {centers[0]}"
    assert match_center(centers[1]), f"❌ Second center is incorrect: {centers[1]}"


if __name__ == "__main__":
    test_duplicate_points()
    test_high_dimensional_low_variance()
    test_scale_variance()
    test_clusters_more_than_unique_points()
    test_rotational_invariance()
    test_seed_determinism()
    test_deterministic_result_on_clear_data()
    print("✅ All K-Medians adversarial tests passed successfully.")
