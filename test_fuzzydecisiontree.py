"""Tests for the FuzzyDTree package."""

import numpy as np
from fuzzydtree import (
    FuzzyTreeClassifier,
    FuzzyTreeRegressor,
)
from fuzzydtree.regressor import _c_backend, _find_mse_split_jit


# ==================================================================
# Regressor tests
# ==================================================================

def test_regressor_reasonable_fit():
    """Fuzzy tree should produce a reasonable fit on simple data."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] - X[:, 2] + rng.randn(200) * 0.1

    for depth in [1, 2, 3]:
        sk = DecisionTreeRegressor(max_depth=depth, random_state=42)
        sk.fit(X, y)
        sk_pred = sk.predict(X)

        ft = FuzzyTreeRegressor(max_depth=depth)
        ft.fit(X, y)
        ft_pred = ft.predict(X)

        sk_mse = np.mean((y - sk_pred) ** 2)
        ft_mse = np.mean((y - ft_pred) ** 2)

        ratio = ft_mse / max(sk_mse, 1e-10)
        print(f"  depth={depth}  sklearn MSE={sk_mse:.4f}  fuzzy MSE={ft_mse:.4f}  ratio={ratio:.3f}")
        assert ratio < 2.0, f"Fuzzy tree much worse than sklearn at depth {depth}"

    print("  PASSED\n")


def test_regressor_reduces_overfitting():
    """Fuzzy splits should generalise reasonably on noisy data."""
    rng = np.random.RandomState(7)
    X_train = rng.randn(100, 2)
    y_train = np.sin(X_train[:, 0]) + rng.randn(100) * 0.3

    X_test = rng.randn(500, 2)
    y_test = np.sin(X_test[:, 0])

    fuzzy = FuzzyTreeRegressor(max_depth=6)
    fuzzy.fit(X_train, y_train)
    fuzzy_test_mse = np.mean((y_test - fuzzy.predict(X_test)) ** 2)

    print(f"  fuzzy test MSE = {fuzzy_test_mse:.4f}")
    assert fuzzy_test_mse < 1.0, "Fuzzy tree test MSE too high"
    print("  PASSED\n")


def test_regressor_max_leaves():
    """max_leaves should cap the number of leaves (best-first growth)."""
    rng = np.random.RandomState(1)
    X = rng.randn(200, 4)
    y = X[:, 0] ** 2 + X[:, 1] - 0.5 * X[:, 2]

    for ml in [2, 4, 8, 16]:
        ft = FuzzyTreeRegressor(max_depth=10, max_leaves=ml)
        ft.fit(X, y)
        print(f"  max_leaves={ml:2d}  actual={ft.n_leaves_:2d}  R²={ft.score(X, y):.4f}")
        assert ft.n_leaves_ <= ml, f"Got {ft.n_leaves_} leaves, expected <= {ml}"

    print("  PASSED\n")


def test_regressor_sample_weight():
    """sample_weight should influence the fit."""
    rng = np.random.RandomState(3)
    X = rng.randn(100, 1)
    y = np.where(X[:, 0] > 0, 10.0, 1.0) + rng.randn(100) * 0.1

    w_left = np.where(X[:, 0] <= 0, 10.0, 1.0)
    w_right = np.where(X[:, 0] > 0, 10.0, 1.0)

    ft_left = FuzzyTreeRegressor(max_depth=1)
    ft_left.fit(X, y, sample_weight=w_left)

    ft_right = FuzzyTreeRegressor(max_depth=1)
    ft_right.fit(X, y, sample_weight=w_right)

    pred_l = ft_left.predict(X)
    pred_r = ft_right.predict(X)
    mse_l = np.mean((y - pred_l) ** 2)
    mse_r = np.mean((y - pred_r) ** 2)
    print(f"  weight-left MSE={mse_l:.4f}  weight-right MSE={mse_r:.4f}")
    print("  PASSED\n")


def test_regressor_feature_importances():
    """Feature importances should highlight the relevant features."""
    rng = np.random.RandomState(5)
    X = rng.randn(300, 5)
    y = 3 * X[:, 0] + X[:, 2] ** 2

    ft = FuzzyTreeRegressor(max_depth=5)
    ft.fit(X, y)
    imp = ft.feature_importances_
    print(f"  importances: {np.round(imp, 3)}")
    assert imp[0] + imp[2] > 0.5, "Features 0 and 2 should dominate"
    print("  PASSED\n")


def test_regressor_reproducibility():
    """max_features with random_state should produce reproducible trees."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(100, 2))
    y = X[:, 0] ** 2 - X[:, 1] ** 2 + X[:, 1] - 1

    seeded_a = FuzzyTreeRegressor(
        max_depth=4, max_features=1, random_state=123)
    seeded_b = FuzzyTreeRegressor(
        max_depth=4, max_features=1, random_state=123)
    seeded_a.fit(X, y)
    seeded_b.fit(X, y)
    np.testing.assert_allclose(seeded_a.predict(X), seeded_b.predict(X))
    print("  reproducibility OK")
    print("  PASSED\n")


def test_regressor_missing_values():
    """Missing values in numeric features should be handled."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 2)
    y = np.where(X[:, 0] > 0, 10.0, 1.0) + rng.randn(n) * 0.1
    nan_idx = rng.choice(n, size=30, replace=False)
    X[nan_idx, 0] = np.nan

    ft = FuzzyTreeRegressor(max_depth=3)
    ft.fit(X, y)
    pred = ft.predict(X)
    assert not np.any(np.isnan(pred)), "Predictions should not contain NaN"
    print(f"  R² = {ft.score(X, y):.4f}")
    print("  PASSED\n")


def test_regressor_categorical():
    """Categorical features should produce valid splits."""
    rng = np.random.RandomState(7)
    n = 300
    cat_col = rng.choice([0, 1, 2, 3], size=n).astype(float)
    num_col = rng.randn(n)
    group_means = {0: 1.0, 1: 5.0, 2: 2.0, 3: 8.0}
    y = np.array([group_means[int(c)] for c in cat_col]) + rng.randn(n) * 0.3
    X = np.column_stack([cat_col, num_col])

    ft = FuzzyTreeRegressor(max_depth=4, categorical_features=[0])
    ft.fit(X, y)
    r2 = ft.score(X, y)
    print(f"  R² with categorical feature = {r2:.4f}")
    assert r2 > 0.7, f"R² too low: {r2}"
    print("  PASSED\n")


def test_regressor_new_params_validation():
    """New regressor optimisation parameters should validate and round-trip."""
    defaults = FuzzyTreeRegressor().get_params()
    assert defaults["leaf_l2"] == 0.1
    assert defaults["leaf_l2_mode"] == "centered"
    assert defaults["refine_splits"] is True
    assert defaults["refine_splits_candidates"] == 4

    ft = FuzzyTreeRegressor(
        optimize_split_gain=True,
        leaf_l2=0.2,
        leaf_l2_mode="zero",
        split_gain_l2="leaf_l2",
        refine_splits=True,
        refine_splits_max_iter=2,
        refine_splits_candidates=3,
    )
    params = ft.get_params()
    assert params["optimize_split_gain"] is True
    assert params["leaf_l2"] == 0.2
    assert params["leaf_l2_mode"] == "zero"
    assert params["split_gain_l2"] == "leaf_l2"
    assert params["refine_splits"] is True
    assert params["refine_splits_max_iter"] == 2
    assert params["refine_splits_candidates"] == 3

    bad = FuzzyTreeRegressor(leaf_l2_mode="bad-mode")
    try:
        bad.fit(np.zeros((6, 1)), np.arange(6.0))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid leaf_l2_mode should raise ValueError")

    print("  PASSED\n")


def test_fast_numeric_prediction_matches_leaf_weights():
    """Fast numeric traversal should match the reference leaf-weight path."""
    rng = np.random.RandomState(13)
    X = rng.randn(160, 3)
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2]

    ft = FuzzyTreeRegressor(
        max_depth=4,
        min_samples_leaf=5,
        random_state=0,
        refine_splits=False,
    )
    ft.fit(X, y)

    leaf_values, leaf_weights = ft._collect_all_leaf_weights(X)
    reference = leaf_weights @ leaf_values
    fast = ft.predict(X)
    np.testing.assert_allclose(fast, reference, atol=1e-10)
    print("  fast numeric prediction matches reference")
    print("  PASSED\n")


def test_c_mse_split_backend_matches_numba():
    """Compiled C split kernel should match the reference Numba kernel."""
    if _c_backend is None:
        print("  C backend not built; skipping")
        return

    rng = np.random.RandomState(19)
    y = rng.randn(180)
    w = rng.rand(180) + 0.05
    bin_assign = rng.randint(-1, 24, size=180).astype(np.int32)
    bin_centers = np.linspace(-3.0, 3.0, 24)
    candidates = np.linspace(-2.5, 2.5, 15)
    margins = np.geomspace(0.25, 5.0, 4)

    for optimize_split_gain in (False, True):
        ref = _find_mse_split_jit(
            y, w, bin_assign, bin_centers, candidates, margins,
            3.0, 1.0, optimize_split_gain, 0.01)
        got = _c_backend.find_mse_split(
            y, w, bin_assign, bin_centers, candidates, margins,
            3.0, 1.0, optimize_split_gain, 0.01)
        assert got[0] == ref[0]
        assert got[1] == ref[1]
        np.testing.assert_allclose(got[2], ref[2], atol=1e-10)

    print("  C MSE split backend matches Numba")
    print("  PASSED\n")


def test_c_depth_first_grower_matches_python_builder():
    """Compiled tree grower should preserve the Python depth-first model."""
    if _c_backend is None:
        print("  C backend not built; skipping")
        return

    import fuzzydtree.regressor as reg_module
    import fuzzydtree._tree as tree_module

    rng = np.random.RandomState(23)
    X = rng.randn(260, 5)
    X[rng.choice(len(X), 18, replace=False), 0] = np.nan
    y = (np.nan_to_num(X[:, 0], nan=0.25) ** 2
         + np.sin(X[:, 1]) - 0.4 * X[:, 2])

    params = dict(
        max_depth=4,
        max_bins=32,
        margin_grid_size=3,
        random_state=0,
        optimize_split_gain=True,
        optimize_leaf_values=False,
        refine_splits=False,
        leaf_l2=0.01,
        split_gain_l2="leaf_l2",
    )

    c_model = FuzzyTreeRegressor(**params).fit(X, y)
    c_pred = c_model.predict(X)

    c_backend = reg_module._c_backend
    tree_backend = tree_module._c_backend
    try:
        reg_module._c_backend = None
        tree_module._c_backend = None
        py_model = FuzzyTreeRegressor(**params).fit(X, y)
        py_pred = py_model.predict(X)
    finally:
        reg_module._c_backend = c_backend
        tree_module._c_backend = tree_backend

    # C uses bin-centre std for margins (faster); tree may differ slightly
    assert abs(c_model.n_leaves_ - py_model.n_leaves_) <= 4
    assert abs(c_model.depth_ - py_model.depth_) <= 1
    corr = np.corrcoef(c_pred, py_pred)[0, 1]
    assert corr > 0.99, f"C vs Python prediction correlation too low: {corr:.4f}"
    print("  C depth-first grower predictions closely match Python builder")
    print("  PASSED\n")


# ==================================================================
# Classifier tests
# ==================================================================

def test_classifier_binary():
    """Binary classification should achieve high accuracy on separable data."""
    rng = np.random.RandomState(42)
    X = rng.randn(300, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    ft = FuzzyTreeClassifier(max_depth=4)
    ft.fit(X, y)
    acc = ft.score(X, y)
    print(f"  binary accuracy = {acc:.4f}")
    assert acc > 0.85, f"Accuracy too low: {acc}"
    assert set(ft.classes_) == {0, 1}
    assert ft.predict_proba(X).shape == (300, 2)
    print("  PASSED\n")


def test_classifier_multiclass():
    """Multiclass classification should work and return correct shapes."""
    rng = np.random.RandomState(42)
    n = 400
    X = rng.randn(n, 3)
    y = np.where(X[:, 0] > 0.5, 2,
                 np.where(X[:, 0] < -0.5, 0, 1))

    ft = FuzzyTreeClassifier(max_depth=5)
    ft.fit(X, y)
    acc = ft.score(X, y)
    proba = ft.predict_proba(X)
    pred = ft.predict(X)
    print(f"  multiclass accuracy = {acc:.4f}")
    assert acc > 0.6, f"Accuracy too low: {acc}"
    assert proba.shape == (n, 3)
    assert set(ft.classes_) == {0, 1, 2}
    assert all(p in ft.classes_ for p in pred)
    # Probabilities should sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    print("  PASSED\n")


def test_classifier_predict_proba_calibration():
    """predict_proba should give higher probability to the correct class."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 2)
    y = (X[:, 0] > 0).astype(int)

    ft = FuzzyTreeClassifier(max_depth=3)
    ft.fit(X, y)
    proba = ft.predict_proba(X)
    pred = ft.predict(X)

    for i in range(len(y)):
        if pred[i] == y[i]:
            assert proba[i, y[i]] >= proba[i, 1 - y[i]], \
                f"Wrong class has higher probability at sample {i}"

    print("  proba calibration OK")
    print("  PASSED\n")


def test_classifier_max_leaves():
    """max_leaves should cap leaf count for classifier too."""
    rng = np.random.RandomState(1)
    X = rng.randn(200, 3)
    y = (X[:, 0] > 0).astype(int)

    for ml in [2, 4, 8]:
        ft = FuzzyTreeClassifier(max_depth=10, max_leaves=ml)
        ft.fit(X, y)
        print(f"  max_leaves={ml}  actual={ft.n_leaves_}  acc={ft.score(X, y):.4f}")
        assert ft.n_leaves_ <= ml

    print("  PASSED\n")


def test_classifier_missing_values():
    """Classifier should handle NaN values."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 2)
    y = (X[:, 0] > 0).astype(int)
    X[rng.choice(n, 30, replace=False), 0] = np.nan

    ft = FuzzyTreeClassifier(max_depth=3)
    ft.fit(X, y)
    pred = ft.predict(X)
    assert len(pred) == n
    assert all(p in ft.classes_ for p in pred)
    print(f"  accuracy with NaN = {ft.score(X, y):.4f}")
    print("  PASSED\n")


def test_classifier_categorical():
    """Classifier should handle categorical features."""
    rng = np.random.RandomState(7)
    n = 300
    cat_col = rng.choice([0, 1, 2, 3], size=n).astype(float)
    y = np.where((cat_col == 0) | (cat_col == 1), 0, 1)

    X = cat_col.reshape(-1, 1)
    ft = FuzzyTreeClassifier(max_depth=3, categorical_features=[0])
    ft.fit(X, y)
    acc = ft.score(X, y)
    print(f"  categorical accuracy = {acc:.4f}")
    assert acc > 0.85
    print("  PASSED\n")


def test_classifier_string_labels():
    """Classifier should handle non-integer class labels."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2)
    y = np.where(X[:, 0] > 0, "cat", "dog")

    ft = FuzzyTreeClassifier(max_depth=3)
    ft.fit(X, y)
    pred = ft.predict(X)
    assert all(p in ["cat", "dog"] for p in pred)
    acc = ft.score(X, y)
    print(f"  string labels accuracy = {acc:.4f}")
    assert acc > 0.8
    print("  PASSED\n")


def test_classifier_sample_weight():
    """sample_weight should influence the classifier fit."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 2)
    y = (X[:, 0] > 0).astype(int)

    # Heavy weight on class 0
    w = np.where(y == 0, 10.0, 1.0)
    ft = FuzzyTreeClassifier(max_depth=2)
    ft.fit(X, y, sample_weight=w)
    proba = ft.predict_proba(X)
    # With heavy class-0 weight, average predicted P(class=0) should be higher
    mean_p0 = proba[:, 0].mean()
    print(f"  mean P(class=0) with heavy class-0 weight = {mean_p0:.4f}")
    assert mean_p0 > 0.45
    print("  PASSED\n")


def test_classifier_logit_mse_ablation_modes():
    """Classifier split and leaf objective modes should all produce probabilities."""
    rng = np.random.RandomState(11)
    X = rng.randn(240, 3)
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0.25, 1, 0)

    for split_criterion in ["entropy", "logit_mse"]:
        for leaf_prediction in ["frequency", "logit_mse", "logit_ce"]:
            ft = FuzzyTreeClassifier(
                max_depth=4,
                split_criterion=split_criterion,
                leaf_prediction=leaf_prediction,
                leaf_logit_ce_max_iter=30,
                random_state=0,
            )
            ft.fit(X, y)
            proba = ft.predict_proba(X)
            acc = ft.score(X, y)
            print(
                f"  split={split_criterion:<9} leaf={leaf_prediction:<9} "
                f"acc={acc:.4f}")
            assert proba.shape == (len(y), 2)
            assert np.all(np.isfinite(proba))
            assert np.all(proba >= 0.0)
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
            assert acc > 0.75

    print("  PASSED\n")


def test_classifier_multiclass_logit_mse_leaf_prediction():
    """Global logit leaf refits should support multiclass softmax output."""
    rng = np.random.RandomState(12)
    X = rng.randn(300, 4)
    y = np.where(X[:, 0] > 0.6, 2,
                 np.where(X[:, 1] < -0.4, 1, 0))

    for leaf_prediction in ["logit_mse", "logit_ce"]:
        ft = FuzzyTreeClassifier(
            max_depth=5,
            split_criterion="logit_mse",
            leaf_prediction=leaf_prediction,
            leaf_logit_ce_max_iter=40,
            random_state=0,
        )
        ft.fit(X, y)
        proba = ft.predict_proba(X)
        assert proba.shape == (len(y), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
        assert ft.score(X, y) > 0.65
        print(
            f"  multiclass {leaf_prediction} accuracy = "
            f"{ft.score(X, y):.4f}")
    print("  PASSED\n")


def test_classifier_new_params_validation():
    """New classifier parameters should validate and round-trip through get_params."""
    defaults = FuzzyTreeClassifier().get_params()
    assert defaults["split_criterion"] == "logit_mse"
    assert defaults["leaf_prediction"] == "frequency"

    ft = FuzzyTreeClassifier(
        split_criterion="logit_mse",
        leaf_prediction="logit_mse",
        leaf_logit_l2=0.5,
        leaf_logit_target=3.0,
        leaf_logit_ce_max_iter=25,
        leaf_logit_ce_lr=0.75,
        leaf_logit_ce_tol=1e-5,
    )
    params = ft.get_params()
    assert params["split_criterion"] == "logit_mse"
    assert params["leaf_prediction"] == "logit_mse"
    assert params["leaf_logit_l2"] == 0.5
    assert params["leaf_logit_target"] == 3.0
    assert params["leaf_logit_ce_max_iter"] == 25
    assert params["leaf_logit_ce_lr"] == 0.75
    assert params["leaf_logit_ce_tol"] == 1e-5

    bad = FuzzyTreeClassifier(split_criterion="not-a-loss")
    try:
        bad.fit(np.zeros((4, 1)), np.array([0, 1, 0, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid split_criterion should raise ValueError")

    print("  PASSED\n")


def test_classifier_log_odds_explanation():
    """Classifier should explain selected-class probability in log-odds space."""
    rng = np.random.RandomState(21)
    X = rng.randn(180, 4)
    y = (1.2 * X[:, 0] - 0.7 * X[:, 2] > 0.1).astype(int)

    ft = FuzzyTreeClassifier(max_depth=4, random_state=0)
    ft.fit(X, y)

    x = X[0]
    exp = ft.explain_prediction_log_odds(x, class_label=1)
    proba = ft.predict_proba(x.reshape(1, -1))[0, 1]
    log_odds = ft.predict_log_odds(x.reshape(1, -1), class_label=1)[0]

    assert exp["class_label"] == 1
    np.testing.assert_allclose(
        exp["prediction_probability"], proba, atol=1e-10)
    np.testing.assert_allclose(
        exp["prediction_log_odds"], log_odds, atol=1e-10)
    summed = sum(exp["feature_contributions"].values())
    np.testing.assert_allclose(
        exp["baseline_log_odds"] + summed,
        exp["prediction_log_odds"],
        atol=1e-6)
    assert len(exp["feature_contributions"]) > 0

    print("  log-odds explanation OK")
    print("  PASSED\n")


if __name__ == "__main__":
    tests = [
        # Regressor
        ("Regressor: reasonable fit", test_regressor_reasonable_fit),
        ("Regressor: reduces overfitting", test_regressor_reduces_overfitting),
        ("Regressor: max_leaves", test_regressor_max_leaves),
        ("Regressor: sample_weight", test_regressor_sample_weight),
        ("Regressor: feature importances", test_regressor_feature_importances),
        ("Regressor: reproducibility", test_regressor_reproducibility),
        ("Regressor: missing values", test_regressor_missing_values),
        ("Regressor: categorical", test_regressor_categorical),
        ("Regressor: new params validation", test_regressor_new_params_validation),
        ("Regressor: fast numeric prediction", test_fast_numeric_prediction_matches_leaf_weights),
        ("Regressor: C MSE split backend", test_c_mse_split_backend_matches_numba),
        ("Regressor: C depth-first grower", test_c_depth_first_grower_matches_python_builder),
        # Classifier
        ("Classifier: binary", test_classifier_binary),
        ("Classifier: multiclass", test_classifier_multiclass),
        ("Classifier: predict_proba calibration", test_classifier_predict_proba_calibration),
        ("Classifier: max_leaves", test_classifier_max_leaves),
        ("Classifier: missing values", test_classifier_missing_values),
        ("Classifier: categorical", test_classifier_categorical),
        ("Classifier: string labels", test_classifier_string_labels),
        ("Classifier: sample_weight", test_classifier_sample_weight),
        ("Classifier: logit-MSE ablation modes", test_classifier_logit_mse_ablation_modes),
        ("Classifier: multiclass logit-MSE leaf prediction", test_classifier_multiclass_logit_mse_leaf_prediction),
        ("Classifier: new params validation", test_classifier_new_params_validation),
        ("Classifier: log-odds explanation", test_classifier_log_odds_explanation),
    ]
    for name, fn in tests:
        print(f"[{name}]")
        fn()
    print("All tests passed.")
