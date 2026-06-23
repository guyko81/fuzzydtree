"""
Fuzzy gradient boosting.

Forward-stagewise gradient boosting with smooth ``FuzzyTreeRegressor`` base
learners. Each round fits a fuzzy regression tree to the negative gradient of
the loss (residuals for regression, ``y - p`` for classification) and takes a
shrunk step. Because the base learner is a *smooth* tree, the additive model is
a sum of smooth functions rather than of staircases.
"""

import numpy as np

from .regressor import FuzzyTreeRegressor


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(F):
    F = F - F.max(axis=1, keepdims=True)
    e = np.exp(F)
    return e / e.sum(axis=1, keepdims=True)


class _BaseFuzzyGB:
    """Shared forward-stagewise boosting machinery."""

    _gb_param_names = ("n_estimators", "learning_rate", "subsample",
                       "max_depth", "random_state")

    def __init__(self, *, n_estimators=100, learning_rate=0.1, subsample=1.0,
                 max_depth=3, random_state=None, **tree_params):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree_params = tree_params

    # ------------------------------------------------------------------
    # Sklearn-ish interface
    # ------------------------------------------------------------------

    def get_params(self, deep=True):
        params = {name: getattr(self, name) for name in self._gb_param_names}
        params.update(self.tree_params)
        return params

    def set_params(self, **params):
        for key, val in params.items():
            if key in self._gb_param_names:
                setattr(self, key, val)
            else:
                self.tree_params[key] = val
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_tree(self, seed):
        return FuzzyTreeRegressor(max_depth=self.max_depth,
                                  random_state=int(seed), **self.tree_params)

    def _subsample_indices(self, n, rng):
        if self.subsample >= 1.0:
            return None
        size = max(1, int(round(self.subsample * n)))
        return rng.choice(n, size=size, replace=False)

    def _validate_gb_params(self):
        if int(self.n_estimators) < 1:
            raise ValueError("n_estimators must be at least 1")
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < float(self.subsample) <= 1.0:
            raise ValueError("subsample must be in (0, 1]")

    def _fit_stage(self, X, target, w, seed, rng):
        """Fit one base tree to ``target`` (a pseudo-residual) and return it."""
        idx = self._subsample_indices(X.shape[0], rng)
        if idx is None:
            Xs, ts, ws = X, target, w
        else:
            Xs, ts, ws = X[idx], target[idx], (None if w is None else w[idx])
        return self._make_tree(seed).fit(Xs, ts, sample_weight=ws)


class FuzzyGradientBoostingRegressor(_BaseFuzzyGB):
    """Gradient boosting of fuzzy trees with squared-error loss.

    Parameters
    ----------
    n_estimators : int, default=100
    learning_rate : float, default=0.1
        Shrinkage applied to each tree's contribution.
    subsample : float, default=1.0
        Row fraction sampled (without replacement) per round; ``< 1`` gives
        stochastic gradient boosting.
    max_depth : int, default=3
        Depth of each base ``FuzzyTreeRegressor``.
    random_state : int or None, default=None

    Any other keyword is forwarded to every base ``FuzzyTreeRegressor``.
    """

    def fit(self, X, y, sample_weight=None):
        self._validate_gb_params()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = X.shape[0]
        self.n_features_in_ = X.shape[1]
        w = None if sample_weight is None else \
            np.asarray(sample_weight, dtype=np.float64).ravel()

        self.init_ = float(np.average(y, weights=w))
        F = np.full(n, self.init_, dtype=np.float64)

        master = np.random.RandomState(self.random_state)
        seeds = master.randint(0, np.iinfo(np.int32).max,
                               size=int(self.n_estimators))
        lr = float(self.learning_rate)
        self.estimators_ = []
        for m in range(int(self.n_estimators)):
            residual = y - F            # negative gradient of 0.5*(y-F)^2
            tree = self._fit_stage(X, residual, w, seeds[m], master)
            F += lr * tree.predict(X)
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape[0], self.init_, dtype=np.float64)
        lr = float(self.learning_rate)
        for tree in self.estimators_:
            out += lr * tree.predict(X)
        return out

    def score(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64).ravel()
        pred = self.predict(X)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            ss_res = sw @ ((y - pred) ** 2)
            ss_tot = sw @ ((y - np.average(y, weights=sw)) ** 2)
        else:
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class FuzzyGradientBoostingClassifier(_BaseFuzzyGB):
    """Gradient boosting of fuzzy trees with logistic / softmax loss.

    Binary and multiclass are both supported (one tree per round for binary,
    one tree per class per round for multiclass). Parameters match
    :class:`FuzzyGradientBoostingRegressor`; extra keywords are forwarded to
    every base ``FuzzyTreeRegressor``.
    """

    def fit(self, X, y, sample_weight=None):
        self._validate_gb_params()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        n = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        w = None if sample_weight is None else \
            np.asarray(sample_weight, dtype=np.float64).ravel()
        idx_of = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([idx_of[v] for v in y], dtype=np.int64)

        master = np.random.RandomState(self.random_state)
        lr = float(self.learning_rate)
        self._multiclass = self.n_classes_ > 2

        if not self._multiclass:
            seeds = master.randint(0, np.iinfo(np.int32).max,
                                   size=int(self.n_estimators))
            y01 = (y_idx == 1).astype(np.float64)
            p0 = np.average(y01, weights=w)
            p0 = min(max(p0, 1e-6), 1 - 1e-6)
            self.init_ = float(np.log(p0 / (1.0 - p0)))
            F = np.full(n, self.init_, dtype=np.float64)
            self.estimators_ = []
            for m in range(int(self.n_estimators)):
                grad = y01 - _sigmoid(F)    # negative gradient of log-loss
                tree = self._fit_stage(X, grad, w, seeds[m], master)
                F += lr * tree.predict(X)
                self.estimators_.append(tree)
        else:
            K = self.n_classes_
            seeds = master.randint(0, np.iinfo(np.int32).max,
                                   size=(int(self.n_estimators), K))
            Y = np.zeros((n, K), dtype=np.float64)
            Y[np.arange(n), y_idx] = 1.0
            prior = np.average(Y, axis=0, weights=w)
            prior = np.clip(prior, 1e-6, 1.0)
            self.init_ = np.log(prior)
            self.init_ -= self.init_.mean()
            F = np.tile(self.init_, (n, 1))
            self.estimators_ = []
            for m in range(int(self.n_estimators)):
                P = _softmax(F)
                round_trees = []
                for k in range(K):
                    grad = Y[:, k] - P[:, k]
                    tree = self._fit_stage(X, grad, w, seeds[m, k], master)
                    F[:, k] += lr * tree.predict(X)
                    round_trees.append(tree)
                self.estimators_.append(round_trees)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        lr = float(self.learning_rate)
        if not self._multiclass:
            F = np.full(X.shape[0], self.init_, dtype=np.float64)
            for tree in self.estimators_:
                F += lr * tree.predict(X)
            return F
        F = np.tile(self.init_, (X.shape[0], 1))
        for round_trees in self.estimators_:
            for k, tree in enumerate(round_trees):
                F[:, k] += lr * tree.predict(X)
        return F

    def predict_proba(self, X):
        F = self.decision_function(X)
        if not self._multiclass:
            p1 = _sigmoid(F)
            return np.column_stack([1.0 - p1, p1])
        return _softmax(F)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y, sample_weight=None):
        y = np.asarray(y)
        correct = self.predict(X) == y
        if sample_weight is None:
            return float(np.mean(correct))
        sw = np.asarray(sample_weight, dtype=np.float64).ravel()
        return float((sw * correct).sum() / sw.sum())
