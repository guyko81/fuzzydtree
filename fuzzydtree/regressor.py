"""
Fuzzy Tree Regressor.

MSE-based fuzzy decision tree for regression. Uses the base FuzzyDTree
mechanics with a histogram-accelerated MSE split kernel.
"""

import numpy as np
from numba import njit, prange

from ._tree import FuzzyDTree, _Node, _membership_value

try:
    from . import _c_backend
except ImportError:
    _c_backend = None


# ------------------------------------------------------------------
# JIT-compiled MSE split kernel
# ------------------------------------------------------------------

@njit(cache=True)
def _precompute_bin_stats(y, w, bin_assign, n_bins):
    N = y.shape[0]
    bin_w = np.zeros(n_bins)
    bin_wy = np.zeros(n_bins)
    bin_wy2 = np.zeros(n_bins)
    nan_w = 0.0
    nan_wy = 0.0
    nan_wy2 = 0.0
    for i in range(N):
        b = bin_assign[i]
        if b < 0:
            nan_w += w[i]
            nan_wy += w[i] * y[i]
            nan_wy2 += w[i] * y[i] * y[i]
        else:
            bin_w[b] += w[i]
            bin_wy[b] += w[i] * y[i]
            bin_wy2[b] += w[i] * y[i] * y[i]
    return bin_w, bin_wy, bin_wy2, nan_w, nan_wy, nan_wy2


@njit(parallel=True, cache=True)
def _find_mse_split_jit(y, w, bin_assign, bin_centers,
                        candidates, margins,
                        min_samples_leaf, nan_mu, optimize_split_gain,
                        split_gain_l2):
    """Histogram-accelerated MSE split search for one feature."""
    N = y.shape[0]
    B = bin_centers.shape[0]
    T = candidates.shape[0]
    S = margins.shape[0]
    TS = T * S

    bin_w, bin_wy, bin_wy2, nan_w, nan_wy, nan_wy2 = \
        _precompute_bin_stats(y, w, bin_assign, B)

    w_total = nan_w
    wy_total = nan_wy
    wy2_total = nan_wy2
    for b in range(B):
        w_total += bin_w[b]
        wy_total += bin_wy[b]
        wy2_total += bin_wy2[b]

    if w_total < 1e-15:
        return -1, -1, 0.0

    parent_mean = wy_total / w_total
    parent_loss = wy2_total / w_total - parent_mean * parent_mean

    imp_arr = np.zeros(TS)

    for ts in prange(TS):
        ti = ts // S
        si = ts % S
        t = candidates[ti]
        margin = margins[si]
        inv_margin = 1.0 / margin if margin > 1e-12 else 0.0

        mu_bins = np.empty(B)
        sl = nan_mu * nan_w
        wy_l = nan_mu * nan_wy

        for b in range(B):
            if inv_margin == 0.0:
                mu = 1.0 if bin_centers[b] <= t else 0.0
            else:
                z = (bin_centers[b] - t) * inv_margin
                mu = _membership_value(z)
            mu_bins[b] = mu
            sl += mu * bin_w[b]
            wy_l += mu * bin_wy[b]

        sr = w_total - sl
        if sl < min_samples_leaf or sr < min_samples_leaf:
            continue

        mean_l = wy_l / sl
        mean_r = (wy_total - wy_l) / sr

        if optimize_split_gain:
            aa = nan_mu * nan_mu * nan_w
            ab = nan_mu * (1.0 - nan_mu) * nan_w
            bb = (1.0 - nan_mu) * (1.0 - nan_mu) * nan_w
            rhs_a = nan_mu * nan_wy
            rhs_b = (1.0 - nan_mu) * nan_wy
            for b in range(B):
                mu = mu_bins[b]
                om = 1.0 - mu
                bw = bin_w[b]
                aa += mu * mu * bw
                ab += mu * om * bw
                bb += om * om * bw
                rhs_a += mu * bin_wy[b]
                rhs_b += om * bin_wy[b]

            ridge = split_gain_l2 + 1e-12 * w_total
            aa += ridge
            bb += ridge
            det = aa * bb - ab * ab
            if det <= 1e-18:
                continue
            gain_fit = ((bb * rhs_a * rhs_a - 2.0 * ab * rhs_a * rhs_b
                         + aa * rhs_b * rhs_b) / det)
            child_loss = (wy2_total - gain_fit) / w_total
            if child_loss < 0.0:
                child_loss = 0.0
        else:
            wy2_l = nan_mu * nan_wy2
            for b in range(B):
                wy2_l += mu_bins[b] * bin_wy2[b]
            var_l = wy2_l / sl - mean_l * mean_l
            var_r = (wy2_total - wy2_l) / sr - mean_r * mean_r
            if var_l < 0.0:
                var_l = 0.0
            if var_r < 0.0:
                var_r = 0.0
            child_loss = (sl * var_l + sr * var_r) / w_total

        imp_arr[ts] = parent_loss - child_loss

    best_ts = 0
    best_val = imp_arr[0]
    for ts in range(1, TS):
        if imp_arr[ts] > best_val:
            best_val = imp_arr[ts]
            best_ts = ts

    return best_ts // S, best_ts % S, best_val


# ------------------------------------------------------------------
# Regressor
# ------------------------------------------------------------------

class FuzzyTreeRegressor(FuzzyDTree):
    """Fuzzy decision tree for regression (MSE loss).

    Parameters
    ----------
    optimize_split_gain : bool, default=False
        Use exact two-leaf least-squares objective for split evaluation
        instead of the faster weighted child-mean approximation.
    optimize_leaf_values : bool, default=True
        After tree building, jointly re-solve all leaf values via
        weighted least-squares.
    leaf_l2 : float, default=1e-1
        L2 regularisation strength for the leaf value optimisation.
    leaf_l2_mode : {'zero', 'centered'}, default='centered'
        ``zero`` shrinks post-fit leaf values toward zero. ``centered``
        shrinks them toward the weighted target mean.
    split_gain_l2 : float or 'leaf_l2', default=0.0
        Ridge strength used by the exact two-leaf split objective when
        ``optimize_split_gain=True``. ``'leaf_l2'`` reuses ``leaf_l2``.
    refine_splits : bool, default=True
        After tree construction, greedily refine numeric thresholds and margins
        against training MSE while keeping the tree topology fixed.
    refine_splits_max_iter : int, default=1
        Number of coordinate-refinement passes over internal numeric nodes.
    refine_splits_candidates : int, default=4
        Number of threshold and margin candidates tried around each split.

    All other parameters are inherited from FuzzyDTree.
    """

    def __init__(self, *, optimize_split_gain=False,
                 optimize_leaf_values=True, leaf_l2=1e-1,
                 leaf_l2_mode="centered", split_gain_l2=0.0,
                 refine_splits=True, refine_splits_max_iter=1,
                 refine_splits_candidates=4, **kwargs):
        super().__init__(**kwargs)
        self.optimize_split_gain = optimize_split_gain
        self.optimize_leaf_values = optimize_leaf_values
        self.leaf_l2 = leaf_l2
        self.leaf_l2_mode = leaf_l2_mode
        self.split_gain_l2 = split_gain_l2
        self.refine_splits = refine_splits
        self.refine_splits_max_iter = refine_splits_max_iter
        self.refine_splits_candidates = refine_splits_candidates

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["optimize_split_gain"] = self.optimize_split_gain
        params["optimize_leaf_values"] = self.optimize_leaf_values
        params["leaf_l2"] = self.leaf_l2
        params["leaf_l2_mode"] = self.leaf_l2_mode
        params["split_gain_l2"] = self.split_gain_l2
        params["refine_splits"] = self.refine_splits
        params["refine_splits_max_iter"] = self.refine_splits_max_iter
        params["refine_splits_candidates"] = self.refine_splits_candidates
        return params

    # ------------------------------------------------------------------
    # Loss-specific hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_leaf_value(y, w):
        s = w.sum()
        return (w @ y) / s if s > 1e-15 else 0.0

    @staticmethod
    def _compute_impurity(y, w):
        s = w.sum()
        if s < 1e-15:
            return 0.0
        mean = (w @ y) / s
        return (w @ ((y - mean) ** 2)) / s

    def _eval_numeric_split(self, y, w, bin_assign, bin_centers,
                            candidates, margins, min_samples_leaf, nan_mu):
        split_l2 = self.leaf_l2 if self.split_gain_l2 == "leaf_l2" \
            else float(self.split_gain_l2)
        if _c_backend is not None:
            return _c_backend.find_mse_split(
                y, w, bin_assign, bin_centers,
                candidates, margins,
                min_samples_leaf, nan_mu, self.optimize_split_gain, split_l2)
        return _find_mse_split_jit(
            y, w, bin_assign, bin_centers,
            candidates, margins,
            min_samples_leaf, nan_mu, self.optimize_split_gain, split_l2)

    def _post_fit(self, X, y, w):
        self._validate_regressor_params()
        if self.optimize_leaf_values:
            self._optimize_leaf_values(X, y, w, l2=self.leaf_l2)
            self._fast_numeric_arrays_ = None
        if self.refine_splits:
            self._refine_splits(X, y, w)
            self._fast_numeric_arrays_ = None
            if self.optimize_leaf_values:
                self._optimize_leaf_values(X, y, w, l2=self.leaf_l2)
                self._fast_numeric_arrays_ = None

    def _validate_regressor_params(self):
        if self.leaf_l2 < 0:
            raise ValueError("leaf_l2 must be non-negative")
        if self.leaf_l2_mode not in ("zero", "centered"):
            raise ValueError("leaf_l2_mode must be 'zero' or 'centered'")
        if self.split_gain_l2 != "leaf_l2" and float(self.split_gain_l2) < 0:
            raise ValueError("split_gain_l2 must be non-negative or 'leaf_l2'")
        if self.refine_splits_max_iter < 0:
            raise ValueError("refine_splits_max_iter must be non-negative")
        if self.refine_splits_candidates < 2:
            raise ValueError("refine_splits_candidates must be at least 2")

    # ------------------------------------------------------------------
    # Compiled tree growth
    # ------------------------------------------------------------------

    def _try_build_c_depth_first(self, X, y, w, X_bins):
        if _c_backend is None:
            return None
        if X_bins is None or self.max_leaves is not None:
            return None
        if getattr(self, "is_categorical_", None) is None:
            return None
        if self.is_categorical_.any():
            return None
        if int(self.max_depth) > 30:
            return None

        split_l2 = self.leaf_l2 if self.split_gain_l2 == "leaf_l2" \
            else float(self.split_gain_l2)
        packed = self._pack_numeric_bins_for_c()
        if packed is None:
            return None
        thresholds, centers, n_thresholds = packed
        feature_choices, feature_counts = self._feature_choices_for_c(
            X.shape[1])

        arrays = _c_backend.grow_depth_first_regression(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(X_bins, dtype=np.int32),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(w, dtype=np.float64),
            thresholds,
            centers,
            n_thresholds,
            feature_choices,
            feature_counts,
            int(self.max_depth),
            float(self.min_samples_leaf),
            int(self.margin_grid_size),
            float(self.margin_depth_decay),
            float(self.min_train_weight_fraction),
            bool(self.optimize_split_gain),
            float(split_l2),
        )
        return self._tree_from_c_arrays(arrays)

    def _feature_choices_for_c(self, n_features):
        max_nodes = (1 << (int(self.max_depth) + 1)) - 1
        choices = np.full((max_nodes, n_features), -1, dtype=np.int32)
        counts = np.zeros(max_nodes, dtype=np.int32)
        if self.max_features is None:
            choices[:, :n_features] = np.arange(n_features, dtype=np.int32)
            counts[:] = n_features
            return choices, counts

        # Compute k (feature subset size) once.
        k = len(self._feature_indices(n_features))
        counts[:] = k

        # Generate all max_nodes subsets in one vectorised pass:
        # draw uniform noise, argsort each row, take first k columns.
        noise = self._rng.random((max_nodes, n_features))
        order = np.argsort(noise, axis=1, kind="quicksort").astype(np.int32)
        choices[:, :k] = order[:, :k]
        return (
            np.ascontiguousarray(choices, dtype=np.int32),
            np.ascontiguousarray(counts, dtype=np.int32),
        )

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

    # ------------------------------------------------------------------
    # Leaf value optimisation
    # ------------------------------------------------------------------

    def _optimize_leaf_values(self, X, target, w, l2=1e-2):
        prior = 0.0
        if self.leaf_l2_mode == "centered":
            prior = self._wmean(target, w)

        y_eff = target - prior

        # Fast C path: diagonal approximation via tree traversal.
        # O(N * depth) — avoids materialising the (N * K) leaf-weight matrix.
        if _c_backend is not None:
            arrays = self._numeric_tree_arrays()
            if arrays is not None:
                # arrays = (features, thresholds, margins, lefts, rights, values, nan_go_left)
                features, thresholds, margins, lefts, rights, _, nan_go_left = arrays
                num, den = _c_backend.accumulate_leaf_stats(
                    np.ascontiguousarray(X, dtype=np.float64),
                    features, thresholds, margins, lefts, rights, nan_go_left,
                    np.ascontiguousarray(y_eff, dtype=np.float64),
                    np.ascontiguousarray(w, dtype=np.float64),
                )
                values = num / np.where(den > 0, den + l2, 1.0)
                for node, v in zip(self._leaf_nodes_in_weight_order(), values):
                    node.value = float(v + prior)
                self._fast_numeric_arrays_ = None
                return

        # Fallback: full system solve via dense leaf-weight matrix.
        _, leaf_weights = self._collect_all_leaf_weights(X)
        if leaf_weights.shape[1] == 0:
            return

        sqrt_w = np.sqrt(w)
        Aw = leaf_weights * sqrt_w[:, None]
        bw = y_eff * sqrt_w
        lhs = Aw.T @ Aw
        if l2 > 0:
            lhs.flat[::lhs.shape[0] + 1] += l2
        rhs = Aw.T @ bw

        try:
            values = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            values = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        for node, value in zip(self._leaf_nodes_in_weight_order(), values):
            node.value = float(value + prior)
        self._fast_numeric_arrays_ = None

    # ------------------------------------------------------------------
    # Fixed-topology split refinement
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

    def _weighted_mse(self, X, y, w):
        leaf_values, leaf_weights = self._collect_all_leaf_weights(X)
        pred = leaf_weights @ leaf_values
        return float((w @ ((y - pred) ** 2)) / max(w.sum(), 1e-15))

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

    def _refine_splits(self, X, y, w):
        max_iter = int(self.refine_splits_max_iter)
        if max_iter <= 0:
            return

        n_candidates = int(self.refine_splits_candidates)
        for _ in range(max_iter):
            improved = False
            for node in self._internal_numeric_nodes():
                old_threshold = node.threshold
                old_margin = node.margin
                best_threshold = old_threshold
                best_margin = old_margin
                best_loss = self._weighted_mse(X, y, w)

                thresholds = self._candidate_thresholds(
                    X[:, node.feature], old_threshold, n_candidates)
                margins = self._candidate_margins(old_margin, n_candidates)

                for threshold in thresholds:
                    for m in margins:
                        node.threshold = float(threshold)
                        node.margin = float(m)
                        loss = self._weighted_mse(X, y, w)
                        if loss + 1e-12 < best_loss:
                            best_loss = loss
                            best_threshold = float(threshold)
                            best_margin = float(m)
                            improved = True

                node.threshold = best_threshold
                node.margin = best_margin
                if self.optimize_leaf_values:
                    self._optimize_leaf_values(X, y, w, l2=self.leaf_l2)

            if not improved:
                break

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            ss_res = (sw @ ((y - y_pred) ** 2))
            ss_tot = (sw @ ((y - self._wmean(y, sw)) ** 2))
        else:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
