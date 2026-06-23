"""
Fuzzy Tree Classifier.

Fuzzy decision tree for classification. Uses the base FuzzyDTree
mechanics with histogram-accelerated split kernels. The default split
criterion is logit-MSE/Gini; entropy remains available as an option.
"""

import numpy as np
from numba import njit, prange

from ._tree import FuzzyDTree, _Node, _membership_value

try:
    from . import _c_backend
except ImportError:
    _c_backend = None


# ------------------------------------------------------------------
# JIT-compiled cross-entropy split kernel
# ------------------------------------------------------------------

@njit(cache=True)
def _precompute_bin_class_stats(y_int, w, bin_assign, n_bins, n_classes):
    N = y_int.shape[0]
    bin_wc = np.zeros((n_bins, n_classes))
    nan_wc = np.zeros(n_classes)
    for i in range(N):
        c = int(y_int[i])
        b = bin_assign[i]
        if b < 0:
            nan_wc[c] += w[i]
        else:
            bin_wc[b, c] += w[i]
    return bin_wc, nan_wc


@njit(parallel=True, cache=True)
def _find_entropy_split_jit(y_int, w, bin_assign, bin_centers,
                            candidates, margins,
                            min_samples_leaf, nan_mu, n_classes):
    """Histogram-accelerated cross-entropy split search for one feature."""
    B = bin_centers.shape[0]
    T = candidates.shape[0]
    S = margins.shape[0]
    TS = T * S

    bin_wc, nan_wc = _precompute_bin_class_stats(
        y_int, w, bin_assign, B, n_classes)

    total_wc = np.zeros(n_classes)
    w_total = 0.0
    for c in range(n_classes):
        total_wc[c] = nan_wc[c]
        for b in range(B):
            total_wc[c] += bin_wc[b, c]
        w_total += total_wc[c]

    if w_total < 1e-15:
        return -1, -1, 0.0

    parent_entropy = 0.0
    for c in range(n_classes):
        p = total_wc[c] / w_total
        if p > 1e-15:
            parent_entropy -= p * np.log(p)

    imp_arr = np.zeros(TS)

    for ts in prange(TS):
        ti = ts // S
        si = ts % S
        t = candidates[ti]
        margin = margins[si]
        inv_margin = 1.0 / margin if margin > 1e-12 else 0.0

        left_wc = np.empty(n_classes)
        for c in range(n_classes):
            left_wc[c] = nan_mu * nan_wc[c]

        for b in range(B):
            if inv_margin == 0.0:
                mu = 1.0 if bin_centers[b] <= t else 0.0
            else:
                z = (bin_centers[b] - t) * inv_margin
                mu = _membership_value(z)
            for c in range(n_classes):
                left_wc[c] += mu * bin_wc[b, c]

        sl = 0.0
        for c in range(n_classes):
            sl += left_wc[c]
        sr = w_total - sl

        if sl < min_samples_leaf or sr < min_samples_leaf:
            continue

        H_left = 0.0
        for c in range(n_classes):
            p = left_wc[c] / sl
            if p > 1e-15:
                H_left -= p * np.log(p)

        H_right = 0.0
        for c in range(n_classes):
            right_c = total_wc[c] - left_wc[c]
            p = right_c / sr
            if p > 1e-15:
                H_right -= p * np.log(p)

        child_entropy = (sl * H_left + sr * H_right) / w_total
        imp_arr[ts] = parent_entropy - child_entropy

    best_ts = 0
    best_val = imp_arr[0]
    for ts in range(1, TS):
        if imp_arr[ts] > best_val:
            best_val = imp_arr[ts]
            best_ts = ts

    return best_ts // S, best_ts % S, best_val


@njit(parallel=True, cache=True)
def _find_logit_mse_split_jit(y_int, w, bin_assign, bin_centers,
                              candidates, margins,
                              min_samples_leaf, nan_mu, n_classes):
    """Histogram split search for one-vs-rest logit-MSE targets.

    With targets +K for the true class and -K for other classes, the local
    least-squares impurity is proportional to multiclass Gini impurity. The
    scale K cancels out for split ranking, so this kernel uses the unscaled
    objective.
    """
    B = bin_centers.shape[0]
    T = candidates.shape[0]
    S = margins.shape[0]
    TS = T * S

    bin_wc, nan_wc = _precompute_bin_class_stats(
        y_int, w, bin_assign, B, n_classes)

    total_wc = np.zeros(n_classes)
    w_total = 0.0
    for c in range(n_classes):
        total_wc[c] = nan_wc[c]
        for b in range(B):
            total_wc[c] += bin_wc[b, c]
        w_total += total_wc[c]

    if w_total < 1e-15:
        return -1, -1, 0.0

    parent_sq = 0.0
    for c in range(n_classes):
        p = total_wc[c] / w_total
        parent_sq += p * p
    parent_impurity = 1.0 - parent_sq

    imp_arr = np.zeros(TS)

    for ts in prange(TS):
        ti = ts // S
        si = ts % S
        t = candidates[ti]
        margin = margins[si]
        inv_margin = 1.0 / margin if margin > 1e-12 else 0.0

        left_wc = np.empty(n_classes)
        for c in range(n_classes):
            left_wc[c] = nan_mu * nan_wc[c]

        for b in range(B):
            if inv_margin == 0.0:
                mu = 1.0 if bin_centers[b] <= t else 0.0
            else:
                z = (bin_centers[b] - t) * inv_margin
                mu = _membership_value(z)
            for c in range(n_classes):
                left_wc[c] += mu * bin_wc[b, c]

        sl = 0.0
        for c in range(n_classes):
            sl += left_wc[c]
        sr = w_total - sl

        if sl < min_samples_leaf or sr < min_samples_leaf:
            continue

        left_sq = 0.0
        right_sq = 0.0
        for c in range(n_classes):
            lp = left_wc[c] / sl
            rc = total_wc[c] - left_wc[c]
            rp = rc / sr
            left_sq += lp * lp
            right_sq += rp * rp

        H_left = 1.0 - left_sq
        H_right = 1.0 - right_sq
        child_impurity = (sl * H_left + sr * H_right) / w_total
        imp_arr[ts] = parent_impurity - child_impurity

    best_ts = 0
    best_val = imp_arr[0]
    for ts in range(1, TS):
        if imp_arr[ts] > best_val:
            best_val = imp_arr[ts]
            best_ts = ts

    return best_ts // S, best_ts % S, best_val


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------

class FuzzyTreeClassifier(FuzzyDTree):
    """Fuzzy decision tree for classification.

    Splits use a logit-MSE/Gini objective by default. Leaf class probabilities
    are computed from the fuzzy-weighted training samples that reach each leaf.

    Parameters
    ----------
    split_criterion : {'logit_mse', 'gini', 'entropy'}, default='logit_mse'
        Objective used while growing splits. ``logit_mse`` uses the
        least-squares impurity induced by one-vs-rest finite-logit targets;
        for split ranking this is equivalent to multiclass Gini. ``entropy``
        uses cross-entropy information gain.
    leaf_prediction : {'frequency', 'logit_mse', 'logit_ce'}, default='frequency'
        Post-fit leaf model. ``frequency`` keeps the historical per-leaf class
        distributions. ``logit_mse`` jointly refits all leaves in logit space
        with ridge least-squares, then predicts with softmax. ``logit_ce``
        refits the same leaf logits with softmax cross-entropy.
    leaf_logit_l2 : float, default=1e-2
        L2 regularisation for logit leaf refits.
    leaf_logit_target : float, default=4.0
        Finite target logit magnitude used by the logit-MSE leaf refit.
    leaf_logit_ce_max_iter : int, default=100
        Maximum line-search iterations for ``leaf_prediction='logit_ce'``.
    leaf_logit_ce_lr : float, default=1.0
        Initial line-search step size for ``leaf_prediction='logit_ce'``.
    leaf_logit_ce_tol : float, default=1e-6
        Relative improvement tolerance for ``leaf_prediction='logit_ce'``.
    optimize_leaf_values : bool, default=False
        If True, jointly re-fit the leaf log-odds after building (and after any
        split refinement). Maps a ``frequency`` leaf model up to ``logit_mse``;
        if ``leaf_prediction`` is already a logit mode it is honoured as-is.
    refine_splits : bool, default=False
        After tree construction, greedily refine numeric thresholds and margins
        against weighted log-loss while keeping the tree topology fixed.
    refine_splits_max_iter : int, default=1
        Number of coordinate-refinement passes over internal numeric nodes.
    refine_splits_candidates : int, default=4
        Number of threshold and margin candidates tried around each split.

    Inherited from FuzzyDTree (see its docstring for full descriptions)
    ------------------------------------------------------------------
    max_depth : int, default=5
    max_leaves : int or None, default=None
    min_samples_leaf : float, default=1.0
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
    categorical_features : list of int, 'auto', or None, default=None
    max_cat_threshold : int, default=64
    random_state : int or None, default=None
    margin_grid_size : int, default=10
    margin_min_scale : float, default=1e-4
    margin_max_scale : float, default=20.0
    include_hard_splits : bool, default=True
    margin_cv_folds : int, default=10
    margin_cv_repeats : int, default=3
    lookahead_horizon : int, default=0
    lookahead_candidates : int, default=4
    lookahead_val_fraction : float, default=0.3
    lookahead_min_val : float, default=5.0
    max_bins : int, default=256
    margin_depth_decay : float, default=1.0
    min_train_weight_fraction : float, default=0.01
    prebin_numeric : bool, default=True
    """

    def __init__(self, *,
                 # --- inherited FuzzyDTree parameters (listed for IDE hover) ---
                 max_depth=5, max_leaves=None, min_samples_leaf=1.0,
                 max_features=None, categorical_features=None,
                 max_cat_threshold=64, random_state=None,
                 margin_grid_size=10, margin_min_scale=1e-4,
                 margin_max_scale=20.0, include_hard_splits=True,
                 max_bins=256, margin_depth_decay=1.0,
                 min_train_weight_fraction=0.01, prebin_numeric=True,
                 margin_cv_folds=10, margin_cv_repeats=3,
                 lookahead_horizon=0, lookahead_candidates=4,
                 lookahead_val_fraction=0.3, lookahead_min_val=5.0,
                 # --- classifier-specific parameters ---
                 split_criterion="logit_mse",
                 leaf_prediction="frequency", leaf_logit_l2=1e-2,
                 leaf_logit_target=4.0, leaf_logit_ce_max_iter=100,
                 leaf_logit_ce_lr=1.0, leaf_logit_ce_tol=1e-6,
                 optimize_leaf_values=False, refine_splits=False,
                 refine_splits_max_iter=1, refine_splits_candidates=4):
        super().__init__(
            max_depth=max_depth, max_leaves=max_leaves,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            categorical_features=categorical_features,
            max_cat_threshold=max_cat_threshold, random_state=random_state,
            margin_grid_size=margin_grid_size, margin_min_scale=margin_min_scale,
            margin_max_scale=margin_max_scale,
            include_hard_splits=include_hard_splits, max_bins=max_bins,
            margin_depth_decay=margin_depth_decay,
            min_train_weight_fraction=min_train_weight_fraction,
            prebin_numeric=prebin_numeric, margin_cv_folds=margin_cv_folds,
            margin_cv_repeats=margin_cv_repeats,
            lookahead_horizon=lookahead_horizon,
            lookahead_candidates=lookahead_candidates,
            lookahead_val_fraction=lookahead_val_fraction,
            lookahead_min_val=lookahead_min_val)
        self.split_criterion = split_criterion
        self.leaf_prediction = leaf_prediction
        self.leaf_logit_l2 = leaf_logit_l2
        self.leaf_logit_target = leaf_logit_target
        self.leaf_logit_ce_max_iter = leaf_logit_ce_max_iter
        self.leaf_logit_ce_lr = leaf_logit_ce_lr
        self.leaf_logit_ce_tol = leaf_logit_ce_tol
        self.optimize_leaf_values = optimize_leaf_values
        self.refine_splits = refine_splits
        self.refine_splits_max_iter = refine_splits_max_iter
        self.refine_splits_candidates = refine_splits_candidates

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["split_criterion"] = self.split_criterion
        params["leaf_prediction"] = self.leaf_prediction
        params["leaf_logit_l2"] = self.leaf_logit_l2
        params["leaf_logit_target"] = self.leaf_logit_target
        params["leaf_logit_ce_max_iter"] = self.leaf_logit_ce_max_iter
        params["leaf_logit_ce_lr"] = self.leaf_logit_ce_lr
        params["leaf_logit_ce_tol"] = self.leaf_logit_ce_tol
        params["optimize_leaf_values"] = self.optimize_leaf_values
        params["refine_splits"] = self.refine_splits
        params["refine_splits_max_iter"] = self.refine_splits_max_iter
        params["refine_splits_candidates"] = self.refine_splits_candidates
        return params

    # ------------------------------------------------------------------
    # Loss-specific hooks
    # ------------------------------------------------------------------

    def _prepare_y(self, y):
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self._class_to_idx[v] for v in y], dtype=np.float64)

    @staticmethod
    def _compute_leaf_value(y, w):
        """Majority class (as float index)."""
        if w.sum() < 1e-15:
            return 0.0
        classes = np.unique(y)
        best_class = classes[0]
        best_weight = 0.0
        for c in classes:
            cw = w[y == c].sum()
            if cw > best_weight:
                best_weight = cw
                best_class = c
        return float(best_class)

    def _child_pred(self, y, w):
        """Fuzzy-weighted class-probability vector for a child."""
        s = w.sum()
        if s < 1e-12:
            return None
        p = np.zeros(self.n_classes_, dtype=np.float64)
        np.add.at(p, y.astype(np.intp), w)
        return p / s

    @staticmethod
    def _blend_loss(y_val, w_val, mu_val, left_pred, right_pred):
        """Weighted multiclass log-loss of the blended class probabilities."""
        prob = (mu_val[:, None] * left_pred[None, :]
                + (1.0 - mu_val)[:, None] * right_pred[None, :])
        true_p = prob[np.arange(len(y_val)), y_val.astype(np.intp)]
        true_p = np.clip(true_p, 1e-12, 1.0)
        return float(-(w_val @ np.log(true_p)))

    def _la_leaf_value(self, y, w):
        """Class-probability vector stored at a rollout leaf."""
        s = w.sum()
        p = np.zeros(self.n_classes_, dtype=np.float64)
        if s > 1e-12:
            np.add.at(p, y.astype(np.intp), w)
            p /= s
        else:
            p[:] = 1.0 / self.n_classes_
        return p

    @staticmethod
    def _la_loss(y, w, pred):
        """Weighted log-loss of a rollout subtree's blended probabilities."""
        s = w.sum()
        pt = np.clip(pred[np.arange(len(y)), y.astype(np.intp)], 1e-12, 1.0)
        return float(-(w @ np.log(pt)) / s) if s > 1e-12 else \
            float(-np.mean(np.log(pt)))

    @staticmethod
    def _compute_gini_impurity(y, w, n_classes=None):
        s = w.sum()
        if s < 1e-15:
            return 0.0
        if n_classes is None:
            classes = np.unique(y)
        else:
            classes = range(n_classes)
        sum_sq = 0.0
        for c in classes:
            p = w[y == c].sum() / s
            sum_sq += p * p
        return 1.0 - sum_sq

    def _validate_classifier_params(self):
        if self.split_criterion not in ("entropy", "logit_mse", "gini"):
            raise ValueError(
                "split_criterion must be 'entropy', 'logit_mse', or 'gini'")
        if self.leaf_prediction not in ("frequency", "logit_mse", "logit_ce"):
            raise ValueError(
                "leaf_prediction must be 'frequency', 'logit_mse', "
                "or 'logit_ce'")
        if self.leaf_logit_l2 < 0:
            raise ValueError("leaf_logit_l2 must be non-negative")
        if self.leaf_logit_target <= 0:
            raise ValueError("leaf_logit_target must be positive")
        if self.leaf_logit_ce_max_iter < 0:
            raise ValueError("leaf_logit_ce_max_iter must be non-negative")
        if self.leaf_logit_ce_lr <= 0:
            raise ValueError("leaf_logit_ce_lr must be positive")
        if self.leaf_logit_ce_tol < 0:
            raise ValueError("leaf_logit_ce_tol must be non-negative")
        if self.refine_splits_max_iter < 0:
            raise ValueError("refine_splits_max_iter must be non-negative")
        if self.refine_splits_candidates < 2:
            raise ValueError("refine_splits_candidates must be at least 2")

    def _compute_impurity(self, y, w):
        if self.split_criterion in ("logit_mse", "gini"):
            return self._compute_gini_impurity(y, w, self.n_classes_)
        return self._compute_entropy_impurity(y, w)

    @staticmethod
    def _compute_entropy_impurity(y, w):
        """Weighted cross-entropy."""
        s = w.sum()
        if s < 1e-15:
            return 0.0
        classes = np.unique(y)
        entropy = 0.0
        for c in classes:
            p = w[y == c].sum() / s
            if p > 1e-15:
                entropy -= p * np.log(p)
        return entropy

    def _eval_numeric_split(self, y, w, bin_assign, bin_centers,
                            candidates, margins, min_samples_leaf, nan_mu):
        if self.split_criterion in ("logit_mse", "gini"):
            return _find_logit_mse_split_jit(
                y, w, bin_assign, bin_centers,
                candidates, margins,
                min_samples_leaf, nan_mu, self.n_classes_)
        return _find_entropy_split_jit(
            y, w, bin_assign, bin_centers,
            candidates, margins,
            min_samples_leaf, nan_mu, self.n_classes_)

    def _try_build_c_lookahead(self, X, y, w, X_bins):
        if _c_backend is None:
            return None
        if not hasattr(_c_backend, "grow_lookahead_classifier"):
            return None
        if X_bins is None:
            return None
        if self.max_features is not None:
            return None
        if getattr(self, "is_categorical_", None) is None:
            return None
        if self.is_categorical_.any():
            return None
        if int(self.max_depth) > 20 or int(self.lookahead_horizon) > 12:
            return None

        rng = np.random.RandomState(
            0 if self.random_state is None else self.random_state)
        val = rng.random(X.shape[0]) < float(self.lookahead_val_fraction)
        if val.sum() < 2 or (~val).sum() < 2:
            return None

        packed = self._pack_numeric_bins_for_c()
        if packed is None:
            return None
        thresholds, centers, n_thresholds = packed
        split_criterion = 1 if self.split_criterion == "entropy" else 0

        arrays = _c_backend.grow_lookahead_classifier(
            np.ascontiguousarray(X[~val], dtype=np.float64),
            np.ascontiguousarray(X_bins[~val], dtype=np.int32),
            np.ascontiguousarray(y[~val], dtype=np.float64),
            np.ascontiguousarray(w[~val], dtype=np.float64),
            np.ascontiguousarray(X[val], dtype=np.float64),
            np.ascontiguousarray(y[val], dtype=np.float64),
            np.ascontiguousarray(w[val], dtype=np.float64),
            thresholds,
            centers,
            n_thresholds,
            int(self.max_depth),
            float(self.min_samples_leaf),
            int(self.margin_grid_size),
            float(self.margin_min_scale),
            float(self.margin_max_scale),
            bool(self.include_hard_splits),
            float(self.margin_depth_decay),
            float(self.min_train_weight_fraction),
            int(self.lookahead_horizon),
            int(self.lookahead_candidates),
            float(self.lookahead_min_val),
            int(self.margin_cv_folds),
            int(self.margin_cv_repeats),
            0 if self.random_state is None else int(self.random_state),
            int(self.n_classes_),
            int(split_criterion),
        )
        root = self._tree_from_c_arrays(arrays)
        self._la_refit_values(root, X, y, w)
        return root

    def _pack_numeric_bins_for_c(self):
        if not self._bin_thresholds_:
            return None
        n_features = len(self._bin_thresholds_)
        max_thresholds = max(
            [len(t) for t in self._bin_thresholds_ if t is not None] + [0])
        if max_thresholds <= 0:
            return None
        thresholds = np.zeros((n_features, max_thresholds), dtype=np.float64)
        centers = np.zeros((n_features, max_thresholds + 1), dtype=np.float64)
        n_thresholds = np.zeros(n_features, dtype=np.int32)
        for j, vals in enumerate(self._bin_thresholds_):
            if vals is None:
                continue
            vals = np.asarray(vals, dtype=np.float64)
            ctrs = np.asarray(self._bin_centers_[j], dtype=np.float64)
            n = len(vals)
            n_thresholds[j] = n
            thresholds[j, :n] = vals
            centers[j, :n + 1] = ctrs
        return (
            np.ascontiguousarray(thresholds),
            np.ascontiguousarray(centers),
            np.ascontiguousarray(n_thresholds),
        )

    def _tree_from_c_arrays(self, arrays):
        (features, thresholds, margins, lefts, rights, values,
         n_samples, imps, nan_left) = arrays

        def build(i):
            i = int(i)
            if int(lefts[i]) < 0:
                return _Node(value=float(values[i]),
                             n_samples=float(n_samples[i]))
            return _Node(
                value=float(values[i]),
                feature=int(features[i]),
                threshold=float(thresholds[i]),
                margin=float(margins[i]),
                left=build(lefts[i]),
                right=build(rights[i]),
                n_samples=float(n_samples[i]),
                impurity_reduction=float(imps[i]),
                nan_go_left=bool(nan_left[i]),
                is_categorical=False,
                categories_left=None,
            )

        return build(0)

    def _post_fit(self, X, y, w):
        """Fit per-leaf distributions, then optionally refine the splits."""
        leaf_weights = self._fit_leaf_distributions(
            X, y, w, compute_node_probs=not self.refine_splits)
        if self.refine_splits:
            self._refine_splits(X, y, w, leaf_weights=leaf_weights)
            self._fit_node_class_probs(X, y, w)

    def _fit_leaf_distributions(self, X, y, w, compute_node_probs=True):
        """Compute per-leaf class probabilities (and optimised leaf logits)."""
        _, leaf_weights = self._collect_all_leaf_weights(X)
        y_int = y.astype(np.int64)
        class_weights = np.zeros((len(y_int), self.n_classes_),
                                 dtype=np.float64)
        class_weights[np.arange(len(y_int)), y_int] = w
        self.leaf_probs_ = leaf_weights.T @ class_weights
        row_sums = self.leaf_probs_.sum(axis=1, keepdims=True)
        self.leaf_probs_ = self.leaf_probs_ / np.where(
            row_sums > 1e-15, row_sums, 1.0)
        empty = (row_sums[:, 0] <= 1e-15)
        if empty.any():
            self.leaf_probs_[empty] = 1.0 / self.n_classes_

        self.leaf_logits_ = None
        mode = self.leaf_prediction
        if self.optimize_leaf_values and mode == "frequency":
            mode = "logit_mse"   # optimize_leaf_values turns on the logit refit
        if mode == "logit_mse":
            self._optimize_leaf_logits(leaf_weights, y, w)
        elif mode == "logit_ce":
            self._optimize_leaf_logits_ce(leaf_weights, y, w)

        if compute_node_probs:
            self._fit_node_class_probs(X, y, w)
        return leaf_weights

    def _fit_node_class_probs(self, X, y, w):
        self._leaf_prob_by_node_id_ = {
            id(node): prob.copy()
            for node, prob in zip(self._leaf_nodes_in_weight_order(),
                                  self.leaf_probs_)
        }
        self._node_class_probs_ = {}
        self._compute_node_class_probs(self.tree_, X, y, w)

    def fit(self, X, y, sample_weight=None):
        self._validate_classifier_params()
        return super().fit(X, y, sample_weight=sample_weight)

    def _compute_node_class_probs(self, node, X, y, w):
        if node.is_leaf:
            self._node_class_probs_[id(node)] = (
                self._leaf_prob_by_node_id_[id(node)].copy())
            return

        probs = np.zeros(self.n_classes_, dtype=np.float64)
        weight_sum = w.sum()
        if weight_sum > 1e-15:
            for c in range(self.n_classes_):
                probs[c] = w[y == c].sum() / weight_sum
        else:
            probs[:] = 1.0 / self.n_classes_
        self._node_class_probs_[id(node)] = probs

        mu_l = self._compute_mu_left(
            X[:, node.feature], node.feature, node.threshold, node.margin,
            node.is_categorical, node.categories_left, node.nan_go_left)
        wl = w * mu_l
        wr = w * (1.0 - mu_l)

        wmax = w.max() if w.size else 0.0
        eps = 1e-6 * wmax if wmax > 0 else 1e-10
        if self.min_train_weight_fraction > 0:
            eps = max(eps, self.min_train_weight_fraction * wmax)
        lm = wl > eps
        rm = wr > eps

        self._compute_node_class_probs(node.left, X[lm], y[lm], wl[lm])
        self._compute_node_class_probs(node.right, X[rm], y[rm], wr[rm])

    # ------------------------------------------------------------------
    # Leaf logit optimisation
    # ------------------------------------------------------------------

    def _optimize_leaf_logits(self, leaf_weights, y, w):
        if leaf_weights.shape[1] == 0:
            return

        target = np.full((len(y), self.n_classes_),
                         -self.leaf_logit_target, dtype=np.float64)
        target[np.arange(len(y)), y.astype(np.int64)] = self.leaf_logit_target

        sqrt_w = np.sqrt(w)
        Aw = leaf_weights * sqrt_w[:, None]
        Bw = target * sqrt_w[:, None]
        lhs = Aw.T @ Aw
        if self.leaf_logit_l2 > 0:
            lhs.flat[::lhs.shape[0] + 1] += self.leaf_logit_l2
        rhs = Aw.T @ Bw

        try:
            self.leaf_logits_ = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            self.leaf_logits_ = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    def _initial_leaf_logits_from_probs(self):
        probs = np.clip(self.leaf_probs_, 1e-6, 1.0)
        logits = np.log(probs)
        logits -= logits.mean(axis=1, keepdims=True)
        return logits

    def _optimize_leaf_logits_ce(self, leaf_weights, y, w):
        if leaf_weights.shape[1] == 0:
            return

        y_int = y.astype(np.int64)
        weight_sum = w.sum()
        if weight_sum <= 1e-15:
            return

        target = np.zeros((len(y_int), self.n_classes_), dtype=np.float64)
        target[np.arange(len(y_int)), y_int] = 1.0
        Z = self._initial_leaf_logits_from_probs()

        def objective_and_grad(values):
            scores = leaf_weights @ values
            probs = self._softmax(scores)
            true_probs = np.clip(probs[np.arange(len(y_int)), y_int],
                                 1e-15, 1.0)
            ce = -(w @ np.log(true_probs)) / weight_sum
            penalty = 0.5 * self.leaf_logit_l2 * np.sum(values * values)
            residual = (probs - target) * (w / weight_sum)[:, None]
            grad = leaf_weights.T @ residual
            if self.leaf_logit_l2 > 0:
                grad += self.leaf_logit_l2 * values
            return ce + penalty, grad

        current_obj, grad = objective_and_grad(Z)
        for _ in range(self.leaf_logit_ce_max_iter):
            grad_norm = np.linalg.norm(grad)
            if grad_norm <= self.leaf_logit_ce_tol:
                break

            step = self.leaf_logit_ce_lr
            accepted = False
            while step > 1e-8:
                candidate = Z - step * grad
                candidate -= candidate.mean(axis=1, keepdims=True)
                candidate_obj, candidate_grad = objective_and_grad(candidate)
                if candidate_obj <= current_obj:
                    rel = ((current_obj - candidate_obj)
                           / max(abs(current_obj), 1e-12))
                    Z = candidate
                    current_obj = candidate_obj
                    grad = candidate_grad
                    accepted = True
                    if rel <= self.leaf_logit_ce_tol:
                        step = 0.0
                    break
                step *= 0.5

            if not accepted or step == 0.0:
                break

        self.leaf_logits_ = Z

    @staticmethod
    def _softmax(scores):
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        sums = exp_scores.sum(axis=1, keepdims=True)
        return exp_scores / np.where(sums > 1e-15, sums, 1.0)

    # ------------------------------------------------------------------
    # Fixed-topology split refinement (against weighted log-loss)
    # ------------------------------------------------------------------

    def _internal_numeric_nodes(self):
        nodes = []
        stack = [self.tree_]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                continue
            if not node.is_categorical:
                nodes.append(node)
            stack.append(node.right)
            stack.append(node.left)
        return nodes

    def _candidate_thresholds(self, values, current, n_candidates):
        valid = np.sort(np.unique(values[np.isfinite(values)]))
        if valid.size < 2:
            return np.array([current], dtype=np.float64)
        idx = np.searchsorted(valid, current)
        lo = max(0, idx - n_candidates)
        hi = min(valid.size - 1, idx + n_candidates)
        local = valid[lo:hi + 1]
        mids = 0.5 * (local[:-1] + local[1:])
        if mids.size == 0:
            return np.array([current], dtype=np.float64)
        order = np.argsort(np.abs(mids - current))
        keep = order[:n_candidates]
        return np.unique(np.r_[current, mids[keep]]).astype(np.float64)

    def _candidate_margins(self, current, n_candidates):
        if current <= 1e-12:
            return np.array([current], dtype=np.float64)
        multipliers = np.geomspace(0.25, 4.0, n_candidates)
        return np.unique(np.r_[current, current * multipliers]).astype(
            np.float64)

    def _weighted_logloss_from_raw(self, raw, y, w, use_logits):
        if use_logits:
            probs = self._softmax(raw)
        else:
            row = raw.sum(axis=1, keepdims=True)
            probs = raw / np.where(row > 1e-15, row, 1.0)
        y_int = y.astype(np.int64)
        p_true = np.clip(probs[np.arange(len(y_int)), y_int], 1e-15, 1.0)
        s = w.sum()
        return float(-(w @ np.log(p_true)) / s) if s > 1e-15 else 0.0

    def _weighted_logloss(self, X, y, w):
        _, leaf_weights = self._collect_all_leaf_weights(X)
        use_logits = getattr(self, "leaf_logits_", None) is not None
        leaf_values = self.leaf_logits_ if use_logits else self.leaf_probs_
        return self._weighted_logloss_from_raw(
            leaf_weights @ leaf_values, y, w, use_logits)

    def _split_refinement_cache(self, X, leaf_weights=None):
        if leaf_weights is None:
            _, leaf_weights = self._collect_all_leaf_weights(X)
        use_logits = getattr(self, "leaf_logits_", None) is not None
        leaf_values = self.leaf_logits_ if use_logits else self.leaf_probs_

        return {
            "leaf_weights": leaf_weights,
            "leaf_values": leaf_values,
            "leaf_slices": self._leaf_slices_by_node(),
            "base_raw": leaf_weights @ leaf_values,
            "prefixes": self._node_prefix_weights(X),
            "use_logits": use_logits,
        }

    def _split_refinement_context(self, cache, X, node):
        leaf_values = cache["leaf_values"]
        leaf_slices = cache["leaf_slices"]

        start, stop = leaf_slices[id(node)]
        old_node_raw = (
            cache["leaf_weights"][:, start:stop] @ leaf_values[start:stop])

        left_start, left_stop = leaf_slices[id(node.left)]
        right_start, right_stop = leaf_slices[id(node.right)]
        left_weights = self._collect_subtree_leaf_weights(X, node.left)
        right_weights = self._collect_subtree_leaf_weights(X, node.right)

        return {
            "base_raw": cache["base_raw"],
            "old_node_raw": old_node_raw,
            "left_raw": left_weights @ leaf_values[left_start:left_stop],
            "right_raw": right_weights @ leaf_values[right_start:right_stop],
            "prefix": cache["prefixes"][id(node)],
            "use_logits": cache["use_logits"],
        }

    def _candidate_refinement_logloss(self, context, X, y, w, node,
                                      threshold, margin):
        mu_l = self._node_mu_left(X, node, threshold, margin)
        prefix = context["prefix"][:, None]
        new_node_raw = prefix * (
            mu_l[:, None] * context["left_raw"]
            + (1.0 - mu_l)[:, None] * context["right_raw"])
        raw = context["base_raw"] - context["old_node_raw"] + new_node_raw
        return self._weighted_logloss_from_raw(
            raw, y, w, context["use_logits"])

    def _refine_splits(self, X, y, w, leaf_weights=None):
        max_iter = int(self.refine_splits_max_iter)
        if max_iter <= 0:
            return
        n_candidates = int(self.refine_splits_candidates)
        for _ in range(max_iter):
            improved = False
            cache = None
            for node in self._internal_numeric_nodes():
                old_threshold = node.threshold
                old_margin = node.margin
                best_threshold = old_threshold
                best_margin = old_margin
                if cache is None:
                    cache = self._split_refinement_cache(
                        X, leaf_weights=leaf_weights)
                    leaf_weights = None
                context = self._split_refinement_context(cache, X, node)
                best_loss = self._weighted_logloss_from_raw(
                    context["base_raw"], y, w, context["use_logits"])
                node_improved = False

                thresholds = self._candidate_thresholds(
                    X[:, node.feature], old_threshold, n_candidates)
                if int(self.margin_cv_folds) >= 2:
                    # Keep the cross-validated margin; only refine thresholds.
                    margins = np.array([old_margin], dtype=np.float64)
                else:
                    margins = self._candidate_margins(old_margin, n_candidates)

                for threshold in thresholds:
                    for m in margins:
                        loss = self._candidate_refinement_logloss(
                            context, X, y, w, node, float(threshold),
                            float(m))
                        if loss + 1e-12 < best_loss:
                            best_loss = loss
                            best_threshold = float(threshold)
                            best_margin = float(m)
                            node_improved = True

                if node_improved:
                    node.threshold = best_threshold
                    node.margin = best_margin
                    leaf_weights = self._fit_leaf_distributions(
                        X, y, w, compute_node_probs=False)
                    cache = None
                    improved = True

            if not improved:
                break

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        """Return class probabilities for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        X = self._encode_predict(X)
        _, leaf_weights = self._collect_all_leaf_weights(X)
        if getattr(self, "leaf_logits_", None) is not None:
            return self._softmax(leaf_weights @ self.leaf_logits_)

        probs = leaf_weights @ self.leaf_probs_
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.where(row_sums > 1e-15, row_sums, 1.0)
        return probs

    def predict(self, X):
        """Return predicted class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

    # ------------------------------------------------------------------
    # Log-odds explanations
    # ------------------------------------------------------------------

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid_from_logit(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _class_index(self, class_label=None):
        if class_label is None:
            if self.n_classes_ == 2:
                return 1
            return 0
        matches = np.where(self.classes_ == class_label)[0]
        if len(matches) == 0:
            raise ValueError(f"Unknown class label {class_label!r}")
        return int(matches[0])

    def predict_log_odds(self, X, class_label=None):
        """Return one-vs-rest log-odds for a selected class."""
        class_idx = self._class_index(class_label)
        probs = self.predict_proba(X)[:, class_idx]
        return self._logit(probs)

    def explain_prediction_log_odds(self, x, *, class_label=None,
                                    feature_names=None):
        """Explain a class probability as feature contributions in log-odds.

        The decomposition first follows the fuzzy tree in probability space for
        the selected class, then converts each step into the equivalent change
        in one-vs-rest log-odds along the waterfall path. This explanation is
        exact for the default ``leaf_prediction='frequency'`` probability
        model.
        """
        if getattr(self, "leaf_logits_", None) is not None:
            raise RuntimeError(
                "Log-odds explanations are currently defined for "
                "leaf_prediction='frequency'.")

        x_arr = np.atleast_2d(x)
        x_enc = self._encode_predict(x_arr)[0]

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(self.n_features_in_)]

        class_idx = self._class_index(class_label)
        class_value = self.classes_[class_idx]
        steps = []
        self._explain_proba_recurse(
            x_enc, self.tree_, 1.0, steps, feature_names, class_idx)

        baseline_probability = self._node_class_probs_[id(self.tree_)][class_idx]
        predicted_probability = self.predict_proba(x_arr)[0, class_idx]
        baseline_log_odds = self._logit(baseline_probability)
        predicted_log_odds = self._logit(predicted_probability)

        running_p = baseline_probability
        for step in steps:
            next_p = np.clip(
                running_p + step["probability_contribution"],
                1e-12, 1.0 - 1e-12)
            step["log_odds_contribution"] = (
                self._logit(next_p) - self._logit(running_p))
            running_p = next_p

        feature_contributions = {}
        probability_contributions = {}
        for step in steps:
            fname = step["feature"]
            feature_contributions[fname] = (
                feature_contributions.get(fname, 0.0)
                + step["log_odds_contribution"])
            probability_contributions[fname] = (
                probability_contributions.get(fname, 0.0)
                + step["probability_contribution"])

        feature_contributions = dict(sorted(
            feature_contributions.items(),
            key=lambda kv: abs(kv[1]), reverse=True))
        probability_contributions = dict(sorted(
            probability_contributions.items(),
            key=lambda kv: abs(kv[1]), reverse=True))

        return {
            "class_label": class_value,
            "class_index": class_idx,
            "baseline_probability": baseline_probability,
            "prediction_probability": predicted_probability,
            "baseline_log_odds": baseline_log_odds,
            "prediction_log_odds": predicted_log_odds,
            "steps": steps,
            "feature_contributions": feature_contributions,
            "probability_contributions": probability_contributions,
        }

    def _explain_proba_recurse(self, x, node, weight, steps, feature_names,
                               class_idx):
        if node.is_leaf:
            return

        val = x[node.feature]
        fname = feature_names[node.feature]

        if np.isnan(val):
            mu_l = 1.0 if node.nan_go_left else 0.0
            go_l, go_r = node.nan_go_left, not node.nan_go_left
            desc = (f"{fname} = NaN "
                    f"(routed {'left' if node.nan_go_left else 'right'})")
        elif node.is_categorical:
            goes_left = (node.categories_left is not None
                         and val in node.categories_left)
            mu_l = 1.0 if goes_left else 0.0
            go_l, go_r = goes_left, not goes_left
            cats = sorted(node.categories_left) if node.categories_left else []
            desc = (f"{fname} = {val:.0f} "
                    f"({'in' if goes_left else 'not in'} "
                    f"{{{','.join(str(int(c)) for c in cats)}}})")
        else:
            mu_l = float(self._membership_left(
                np.array([val]), node.threshold, node.margin)[0])
            go_l = go_r = True
            direction = "left" if mu_l > 0.5 else "right"
            desc = (f"{fname} = {val:.4g} "
                    f"(split {node.threshold:.4g}, "
                    f"margin={node.margin:.4g}, "
                    f"mu_left={mu_l:.2f} -> {direction})")

        node_p = self._node_class_probs_[id(node)][class_idx]
        left_p = self._node_class_probs_[id(node.left)][class_idx]
        right_p = self._node_class_probs_[id(node.right)][class_idx]

        if go_l and go_r:
            contribution = weight * (
                mu_l * left_p + (1.0 - mu_l) * right_p - node_p)
            steps.append(dict(
                feature=fname, feature_idx=node.feature,
                feature_value=val, threshold=node.threshold,
                margin=node.margin, mu_left=mu_l, weight=weight,
                probability_contribution=contribution,
                node_probability=node_p,
                left_probability=left_p,
                right_probability=right_p,
                description=desc))
            self._explain_proba_recurse(
                x, node.left, weight * mu_l, steps, feature_names, class_idx)
            self._explain_proba_recurse(
                x, node.right, weight * (1.0 - mu_l), steps,
                feature_names, class_idx)
        elif go_l:
            contribution = weight * (left_p - node_p)
            steps.append(dict(
                feature=fname, feature_idx=node.feature,
                feature_value=val, threshold=node.threshold,
                margin=node.margin, mu_left=mu_l, weight=weight,
                probability_contribution=contribution,
                node_probability=node_p,
                left_probability=left_p,
                right_probability=right_p,
                description=desc))
            self._explain_proba_recurse(
                x, node.left, weight, steps, feature_names, class_idx)
        else:
            contribution = weight * (right_p - node_p)
            steps.append(dict(
                feature=fname, feature_idx=node.feature,
                feature_value=val, threshold=node.threshold,
                margin=node.margin, mu_left=mu_l, weight=weight,
                probability_contribution=contribution,
                node_probability=node_p,
                left_probability=left_p,
                right_probability=right_p,
                description=desc))
            self._explain_proba_recurse(
                x, node.right, weight, steps, feature_names, class_idx)

    def plot_log_odds_contributions(self, x, *, class_label=None,
                                    feature_names=None, ax=None,
                                    max_features=10):
        """Waterfall chart of class log-odds feature contributions."""
        import matplotlib.pyplot as plt

        exp = self.explain_prediction_log_odds(
            x, class_label=class_label, feature_names=feature_names)
        contribs = list(exp["feature_contributions"].items())[:max_features]
        if not contribs:
            raise RuntimeError("No feature contributions to plot.")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        labels = [k for k, _ in contribs]
        values = np.array([v for _, v in contribs])
        y_pos = np.arange(len(values))
        colors = np.where(values >= 0, "#2ca02c", "#d62728")

        ax.barh(y_pos, values, color=colors, alpha=0.82)
        ax.axvline(0.0, color="#333333", lw=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution to log-odds")
        title = (f"Class {exp['class_label']} log-odds: "
                 f"{exp['baseline_log_odds']:.2f} -> "
                 f"{exp['prediction_log_odds']:.2f} "
                 f"(p={exp['prediction_probability']:.1%})")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        return ax

    def score(self, X, y, sample_weight=None):
        """Return mean accuracy."""
        y_pred = self.predict(X)
        y_true = np.asarray(y).ravel()
        correct = (y_pred == y_true).astype(np.float64)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            return (sw @ correct) / sw.sum()
        return correct.mean()
