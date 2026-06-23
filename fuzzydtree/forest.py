"""
Fuzzy random forests.

Bootstrap-aggregated ensembles of fuzzy decision trees. Each tree is grown on a
bootstrap sample of the rows with per-split feature subsampling (``max_features``),
exactly like a classic random forest — the only twist is that the base learner is
a smooth ``FuzzyTree*`` instead of a hard CART tree.
"""

import numpy as np

from .classifier import FuzzyTreeClassifier
from .regressor import FuzzyTreeRegressor

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:  # pragma: no cover - joblib ships with scikit-learn
    _HAVE_JOBLIB = False


class _BaseFuzzyForest:
    """Shared bagging machinery for the fuzzy forests."""

    _tree_cls = None
    _forest_param_names = ("n_estimators", "max_features", "bootstrap",
                           "max_samples", "n_jobs", "random_state")

    def __init__(self, *, n_estimators=100, max_features="sqrt",
                 bootstrap=True, max_samples=None, n_jobs=None,
                 random_state=None, **tree_params):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.tree_params = tree_params

    # ------------------------------------------------------------------
    # Sklearn-ish interface
    # ------------------------------------------------------------------

    def get_params(self, deep=True):
        params = {name: getattr(self, name) for name in self._forest_param_names}
        params.update(self.tree_params)
        return params

    def set_params(self, **params):
        for key, val in params.items():
            if key in self._forest_param_names:
                setattr(self, key, val)
            else:
                self.tree_params[key] = val
        return self

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _make_tree(self, seed):
        return self._tree_cls(max_features=self.max_features,
                              random_state=int(seed), **self.tree_params)

    def _bootstrap_indices(self, n, rng):
        if not self.bootstrap:
            return np.arange(n)
        if self.max_samples is None:
            size = n
        elif isinstance(self.max_samples, float):
            size = max(1, int(round(self.max_samples * n)))
        else:
            size = int(self.max_samples)
        return rng.randint(0, n, size=size)

    def _validate_forest_params(self):
        if int(self.n_estimators) < 1:
            raise ValueError("n_estimators must be at least 1")
        if self.max_samples is not None:
            if isinstance(self.max_samples, float):
                if not 0.0 < self.max_samples <= 1.0:
                    raise ValueError(
                        "float max_samples must be in (0, 1]")
            elif int(self.max_samples) < 1:
                raise ValueError("int max_samples must be >= 1")

    def fit(self, X, y, sample_weight=None):
        self._validate_forest_params()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = X.shape[0]
        self.n_features_in_ = X.shape[1]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
        self._fit_prepare(y)

        # One master RNG seeds every tree and its bootstrap draw, so the whole
        # forest is reproducible from a single random_state.
        master = np.random.RandomState(self.random_state)
        hi = np.iinfo(np.int32).max
        tree_seeds = master.randint(0, hi, size=int(self.n_estimators))
        boot_seeds = master.randint(0, hi, size=int(self.n_estimators))

        def fit_one(tree_seed, boot_seed):
            rng = np.random.RandomState(boot_seed)
            idx = self._bootstrap_indices(n, rng)
            sw = None if sample_weight is None else sample_weight[idx]
            tree = self._make_tree(tree_seed)
            tree.fit(X[idx], y[idx], sample_weight=sw)
            return tree

        pairs = list(zip(tree_seeds, boot_seeds))
        if _HAVE_JOBLIB and self.n_jobs not in (None, 1):
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_one)(ts, bs) for ts, bs in pairs)
        else:
            self.estimators_ = [fit_one(ts, bs) for ts, bs in pairs]
        return self

    def _fit_prepare(self, y):
        """Hook for subclasses to record label metadata."""


class FuzzyForestRegressor(_BaseFuzzyForest):
    """Bagged ensemble of :class:`FuzzyTreeRegressor` (predictions averaged).

    Parameters
    ----------
    n_estimators : int, default=100
    max_features : int, float, {'sqrt', 'log2'} or None, default='sqrt'
        Per-split feature subsampling passed to each base tree.
    bootstrap : bool, default=True
    max_samples : int, float or None, default=None
        Bootstrap sample size (fraction if float). ``None`` uses ``n_samples``.
    n_jobs : int or None, default=None
        Parallelism over trees (requires joblib).
    random_state : int or None, default=None

    Any other keyword is forwarded to every base ``FuzzyTreeRegressor``.
    """

    _tree_cls = FuzzyTreeRegressor

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.estimators_:
            out += tree.predict(X)
        return out / len(self.estimators_)

    def score(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64).ravel()
        pred = self.predict(X)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            mean = np.average(y, weights=sw)
            ss_res = sw @ ((y - pred) ** 2)
            ss_tot = sw @ ((y - mean) ** 2)
        else:
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class FuzzyForestClassifier(_BaseFuzzyForest):
    """Bagged ensemble of :class:`FuzzyTreeClassifier` (probabilities averaged).

    Parameters
    ----------
    n_estimators : int, default=100
    max_features : int, float, {'sqrt', 'log2'} or None, default='sqrt'
        Per-split feature subsampling passed to each base tree.
    bootstrap : bool, default=True
    max_samples : int, float or None, default=None
        Bootstrap sample size (fraction if float). ``None`` uses ``n_samples``.
    n_jobs : int or None, default=None
        Parallelism over trees (requires joblib).
    random_state : int or None, default=None

    Any other keyword is forwarded to every base ``FuzzyTreeClassifier``.
    """

    _tree_cls = FuzzyTreeClassifier

    def _fit_prepare(self, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        col_of = {c: i for i, c in enumerate(self.classes_)}
        for tree in self.estimators_:
            proba = tree.predict_proba(X)
            # A bootstrap sample may miss a class, so map each tree's columns
            # back onto the forest's global class order before averaging.
            cols = [col_of[c] for c in tree.classes_]
            out[:, cols] += proba
        return out / len(self.estimators_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y, sample_weight=None):
        y = np.asarray(y)
        correct = self.predict(X) == y
        if sample_weight is None:
            return float(np.mean(correct))
        sw = np.asarray(sample_weight, dtype=np.float64).ravel()
        return float((sw * correct).sum() / sw.sum())
