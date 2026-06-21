"""
Fuzzy Decision Tree — base class.

A decision tree where splits use fuzzy membership functions instead of hard
thresholds. Each observation partially belongs to both children of a split,
producing a weighted blend over all reachable leaves.

This module provides the loss-agnostic tree mechanics. Subclasses
(FuzzyTreeRegressor, FuzzyTreeClassifier) plug in their own loss, split
kernel, and prediction logic.
"""

import numpy as np
from numba import njit

try:
    from . import _c_backend
except ImportError:
    _c_backend = None


# ------------------------------------------------------------------
# JIT helpers (shared by all subclass kernels)
# ------------------------------------------------------------------

@njit(cache=True)
def _membership_value(z):
    """Compute left-membership from normalised distance z = (x - threshold) / margin.

    Uses a quadratic S-curve with compact support: returns exactly 1.0
    for z <= -1 and exactly 0.0 for z >= 1, with a smooth transition
    in between.  Unlike sigmoid, samples far from the split boundary
    contribute zero weight to the wrong child, eliminating weight
    dilution in deep trees.
    """
    if z <= -1.0:
        return 1.0
    elif z <= 0.0:
        t = z + 1.0
        return 1.0 - 0.5 * t * t
    elif z < 1.0:
        t = z - 1.0
        return 0.5 * t * t
    else:
        return 0.0


@njit(cache=True)
def _predict_numeric_tree_jit(X, features, thresholds, margins, lefts, rights,
                              values, nan_go_left):
    """Predict a numeric-only fuzzy tree stored as flat arrays."""
    n_samples = X.shape[0]
    n_nodes = values.shape[0]
    out = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        pred = 0.0
        stack_nodes = np.empty(n_nodes, dtype=np.int64)
        stack_weights = np.empty(n_nodes, dtype=np.float64)
        top = 0
        stack_nodes[top] = 0
        stack_weights[top] = 1.0
        top += 1

        while top > 0:
            top -= 1
            node_idx = stack_nodes[top]
            weight = stack_weights[top]

            left_idx = lefts[node_idx]
            if left_idx < 0:
                pred += weight * values[node_idx]
                continue

            feature = features[node_idx]
            x = X[i, feature]

            if np.isnan(x):
                mu_l = 1.0 if nan_go_left[node_idx] else 0.0
            else:
                margin = margins[node_idx]
                if margin < 1e-12:
                    mu_l = 1.0 if x <= thresholds[node_idx] else 0.0
                else:
                    mu_l = _membership_value((x - thresholds[node_idx])
                                             / margin)

            if mu_l > 0.0:
                stack_nodes[top] = left_idx
                stack_weights[top] = weight * mu_l
                top += 1
            if mu_l < 1.0:
                stack_nodes[top] = rights[node_idx]
                stack_weights[top] = weight * (1.0 - mu_l)
                top += 1

        out[i] = pred

    return out


# ------------------------------------------------------------------
# Node
# ------------------------------------------------------------------

class _Node:
    __slots__ = ("feature", "threshold", "margin", "left", "right", "value",
                 "n_samples", "impurity_reduction",
                 "nan_go_left", "is_categorical", "categories_left")

    def __init__(self, *, value=None, n_samples=0.0, feature=None,
                 threshold=None, margin=None, left=None, right=None,
                 impurity_reduction=0.0, nan_go_left=True,
                 is_categorical=False, categories_left=None):
        self.feature = feature
        self.threshold = threshold
        self.margin = margin
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.impurity_reduction = impurity_reduction
        self.nan_go_left = nan_go_left
        self.is_categorical = is_categorical
        self.categories_left = categories_left

    @property
    def is_leaf(self):
        return self.left is None


# ------------------------------------------------------------------
# Base tree
# ------------------------------------------------------------------

class FuzzyDTree:
    """Loss-agnostic fuzzy decision tree base class.

    Subclasses must implement:
        _compute_leaf_value(y, w) — value stored at a leaf node.
        _compute_impurity(y, w) — node impurity (e.g. MSE, entropy).
        _eval_numeric_split(y, w, bin_assign, ...) — JIT kernel wrapper.

    Parameters
    ----------
    max_depth : int, default=5
    max_leaves : int or None, default=None
    min_samples_leaf : float, default=1.0
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
    categorical_features : list of int, 'auto', or None, default=None
    max_cat_threshold : int, default=64
    random_state : int or None, default=None
    margin_grid_size : int, default=10
    max_bins : int, default=256
    margin_depth_decay : float, default=1.0
    min_train_weight_fraction : float, default=0.01
    prebin_numeric : bool, default=True
    """

    def __init__(self, *, max_depth=5, max_leaves=None, min_samples_leaf=1.0,
                 max_features=None, categorical_features=None,
                 max_cat_threshold=64, random_state=None,
                 margin_grid_size=10, max_bins=256,
                 margin_depth_decay=1.0, min_train_weight_fraction=0.01,
                 prebin_numeric=True):
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.categorical_features = categorical_features
        self.max_cat_threshold = max_cat_threshold
        self.random_state = random_state
        self.margin_grid_size = margin_grid_size
        self.max_bins = max_bins
        self.margin_depth_decay = margin_depth_decay
        self.min_train_weight_fraction = min_train_weight_fraction
        self.prebin_numeric = prebin_numeric

    # ------------------------------------------------------------------
    # Sklearn interface helpers
    # ------------------------------------------------------------------

    def get_params(self, deep=True):
        return {
            "max_depth": self.max_depth,
            "max_leaves": self.max_leaves,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "categorical_features": self.categorical_features,
            "max_cat_threshold": self.max_cat_threshold,
            "random_state": self.random_state,
            "margin_grid_size": self.margin_grid_size,
            "max_bins": self.max_bins,
            "margin_depth_decay": self.margin_depth_decay,
            "min_train_weight_fraction": self.min_train_weight_fraction,
            "prebin_numeric": self.prebin_numeric,
        }

    def set_params(self, **params):
        for key, val in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key}")
            setattr(self, key, val)
        return self

    # ------------------------------------------------------------------
    # Fuzzy membership
    # ------------------------------------------------------------------

    def _membership_left(self, x, threshold, margin):
        x = np.asarray(x, dtype=np.float64)
        if margin < 1e-12:
            return np.where(x <= threshold, 1.0, 0.0)
        z = (x - threshold) / margin
        return np.where(z <= -1.0, 1.0,
               np.where(z <= 0.0, 1.0 - 0.5 * (z + 1.0) ** 2,
               np.where(z < 1.0, 0.5 * (z - 1.0) ** 2,
               0.0)))

    # ------------------------------------------------------------------
    # Weighted statistics (shared utilities)
    # ------------------------------------------------------------------

    @staticmethod
    def _wmean(y, w):
        s = w.sum()
        return (w @ y) / s if s > 1e-15 else 0.0

    @staticmethod
    def _wstd(x, w):
        s = w.sum()
        if s < 1e-15:
            return 0.0
        mean = (w @ x) / s
        return np.sqrt((w @ ((x - mean) ** 2)) / s)

    def _feature_indices(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                k = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                k = int(np.log2(n_features))
            else:
                raise ValueError(
                    "max_features must be None, int, float, 'sqrt', or 'log2'")
        elif isinstance(self.max_features, (int, np.integer)):
            k = int(self.max_features)
        elif isinstance(self.max_features, (float, np.floating)):
            if not 0 < float(self.max_features) <= 1:
                raise ValueError("float max_features must be in (0, 1]")
            k = int(np.ceil(float(self.max_features) * n_features))
        else:
            raise ValueError(
                "max_features must be None, int, float, 'sqrt', or 'log2'")

        k = max(1, min(n_features, k))
        return self._rng.choice(n_features, size=k, replace=False)

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    def _prepare_y(self, y):
        """Convert raw y to internal representation. Override in subclass."""
        return np.asarray(y, dtype=np.float64).ravel()

    def _compute_leaf_value(self, y, w):
        raise NotImplementedError

    def _compute_impurity(self, y, w):
        raise NotImplementedError

    def _eval_numeric_split(self, y, w, bin_assign, bin_centers,
                            candidates, margins, min_samples_leaf, nan_mu):
        raise NotImplementedError

    def _post_fit(self, X, y, w):
        """Hook called at the end of fit() with encoded X, prepared y, weights."""
        pass

    def _try_build_c_depth_first(self, X, y, w, X_bins):
        """Optional subclass hook for compiled depth-first tree growth."""
        return None

    # ------------------------------------------------------------------
    # Split finding
    # ------------------------------------------------------------------

    def _margin_candidates(self, feat_std, depth=0):
        if feat_std < 1e-12:
            return np.array([1e-12])
        lo = feat_std * 0.4
        hi = feat_std * 20.0 * (self.margin_depth_decay ** depth)
        hi = max(hi, lo)
        return np.geomspace(lo, hi, self.margin_grid_size)

    def _bin_candidates(self, uniq):
        midpoints = 0.5 * (uniq[:-1] + uniq[1:])
        if len(midpoints) <= self.max_bins:
            return midpoints
        idx = np.linspace(0, len(midpoints) - 1, self.max_bins).astype(int)
        return midpoints[idx]

    def _prebin_numeric_features(self, X):
        n_samples, n_features = X.shape
        X_bins = np.full((n_samples, n_features), -1, dtype=np.int32)
        thresholds = [None] * n_features
        centers = [None] * n_features

        if not self.prebin_numeric:
            return None, thresholds, centers

        for j in range(n_features):
            if self.is_categorical_[j]:
                continue
            col = X[:, j]
            valid = col[~np.isnan(col)]
            uniq = np.unique(valid)
            if len(uniq) < 2:
                continue
            candidates = self._bin_candidates(uniq)
            if len(candidates) == 0:
                continue
            n_bins = len(candidates) + 1
            bin_centers = np.empty(n_bins, dtype=np.float64)
            bin_centers[0] = 0.5 * (uniq[0] + candidates[0])
            for bi in range(1, len(candidates)):
                bin_centers[bi] = 0.5 * (candidates[bi - 1] + candidates[bi])
            bin_centers[-1] = 0.5 * (candidates[-1] + uniq[-1])
            safe_col = np.where(np.isnan(col), np.inf, col)
            X_bins[:, j] = np.searchsorted(
                candidates, safe_col, side="left").astype(np.int32)
            X_bins[np.isnan(col), j] = -1
            thresholds[j] = candidates
            centers[j] = bin_centers

        return X_bins, thresholds, centers

    def _find_best_split(self, X, y, w, depth=0, X_bins=None):
        """Return (feature, threshold, margin, imp,
        nan_go_left, is_categorical, categories_left)."""
        n_samples, n_features = X.shape
        w_total = w.sum()
        parent_impurity = self._compute_impurity(y, w)

        best_imp = 0.0
        best_feat = best_thresh = best_margin = None
        best_nan_go_left = True
        best_is_cat = False
        best_cats_left = None

        for j in self._feature_indices(n_features):
            col = X[:, j]
            nan_mask = np.isnan(col)
            has_nan = nan_mask.any()
            valid_mask = ~nan_mask

            if self.is_categorical_[j]:
                result = self._find_best_categorical_split(
                    col, y, w, j, parent_impurity, w_total,
                    nan_mask, valid_mask, has_nan)
                if result is not None:
                    imp, cats_left, nan_left = result
                    if imp > best_imp:
                        best_imp = imp
                        best_feat = j
                        best_thresh = None
                        best_margin = 0.0
                        best_nan_go_left = nan_left
                        best_is_cat = True
                        best_cats_left = cats_left
                continue

            valid_col = col[valid_mask] if has_nan else col
            valid_w = w[valid_mask] if has_nan else w

            feat_std = self._wstd(valid_col, valid_w)
            margins = self._margin_candidates(feat_std, depth)

            if X_bins is not None and self._bin_thresholds_[j] is not None:
                candidates = self._bin_thresholds_[j]
                bin_centers = self._bin_centers_[j]
                bin_assign = X_bins[:, j]
                active_bins = np.unique(bin_assign[valid_mask])
                if len(active_bins) < 2:
                    continue
            else:
                uniq = np.unique(valid_col)
                if len(uniq) < 2:
                    continue
                candidates = self._bin_candidates(uniq)
                T = len(candidates)
                n_bins = T + 1
                bin_centers = np.empty(n_bins)
                bin_centers[0] = 0.5 * (uniq[0] + candidates[0])
                for bi in range(1, T):
                    bin_centers[bi] = 0.5 * (candidates[bi - 1]
                                             + candidates[bi])
                bin_centers[T] = 0.5 * (candidates[-1] + uniq[-1])
                safe_col = np.where(np.isnan(col), np.inf, col)
                bin_assign = np.searchsorted(
                    candidates, safe_col, side="left").astype(np.int32)
                if has_nan:
                    bin_assign[nan_mask] = -1

            nan_mus = [1.0, 0.0] if has_nan else [1.0]
            for nan_mu in nan_mus:
                ti, mi, imp = self._eval_numeric_split(
                    y, w, bin_assign, bin_centers,
                    candidates, margins,
                    self.min_samples_leaf, nan_mu)

                if ti < 0:
                    continue
                if imp > best_imp:
                    best_imp = imp
                    best_feat = j
                    best_thresh = float(candidates[ti])
                    best_margin = float(margins[mi])
                    best_nan_go_left = (nan_mu == 1.0)
                    best_is_cat = False
                    best_cats_left = None

        return (best_feat, best_thresh, best_margin,
                best_imp,
                best_nan_go_left, best_is_cat, best_cats_left)

    def _find_best_categorical_split(self, col, y, w, feature,
                                     parent_impurity, w_total,
                                     nan_mask, valid_mask, has_nan):
        """Find best binary partition of categories (sorted-by-mean heuristic).

        Returns (imp, categories_left_set, nan_go_left) or None.
        """
        valid_col = col[valid_mask]
        valid_y = y[valid_mask]
        valid_w = w[valid_mask]

        categories = np.unique(valid_col)
        if len(categories) < 2:
            return None

        cat_means = np.array([
            self._wmean(valid_y[valid_col == c], valid_w[valid_col == c])
            for c in categories
        ])
        order = np.argsort(cat_means)
        sorted_cats = categories[order]

        best_imp = 0.0
        best_cats_left = None
        best_nan_left = True

        for k in range(1, len(sorted_cats)):
            left_cats = set(sorted_cats[:k].tolist())
            cat_mu_l = np.array([
                1.0 if c in left_cats else 0.0 for c in col
            ], dtype=np.float64)

            if has_nan:
                for try_left in (True, False):
                    mu_l_try = cat_mu_l.copy()
                    mu_l_try[nan_mask] = 1.0 if try_left else 0.0

                    wl = w * mu_l_try
                    wr = w * (1.0 - mu_l_try)
                    sl = wl.sum()
                    sr = wr.sum()

                    if (sl < self.min_samples_leaf
                            or sr < self.min_samples_leaf):
                        continue

                    child_imp = (sl * self._compute_impurity(y, wl)
                                 + sr * self._compute_impurity(y, wr)
                                 ) / w_total
                    imp = parent_impurity - child_imp
                    if imp > best_imp:
                        best_imp = imp
                        best_cats_left = left_cats
                        best_nan_left = try_left
            else:
                wl = w * cat_mu_l
                wr = w * (1.0 - cat_mu_l)
                sl = wl.sum()
                sr = wr.sum()

                if (sl < self.min_samples_leaf
                        or sr < self.min_samples_leaf):
                    continue

                child_imp = (sl * self._compute_impurity(y, wl)
                             + sr * self._compute_impurity(y, wr)
                             ) / w_total
                imp = parent_impurity - child_imp
                if imp > best_imp:
                    best_imp = imp
                    best_cats_left = left_cats
                    best_nan_left = True

        if best_cats_left is None:
            return None
        return best_imp, best_cats_left, best_nan_left

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _compute_mu_left(self, col, feat, thresh, margin, is_cat,
                         categories_left, nan_go_left):
        nan_mask = np.isnan(col)
        if is_cat:
            mu_l = np.array([
                1.0 if c in categories_left else 0.0
                for c in col
            ], dtype=np.float64)
        else:
            mu_l = self._membership_left(col, thresh, margin)
        if nan_mask.any():
            mu_l[nan_mask] = 1.0 if nan_go_left else 0.0
        return mu_l

    def _build_depth_first(self, X, y, w, depth, X_bins=None):
        eff = w.sum()
        val = self._compute_leaf_value(y, w)

        if (depth >= self.max_depth
                or X.shape[0] < 2
                or eff < 2 * self.min_samples_leaf):
            return _Node(value=val, n_samples=eff)

        (feat, thresh, margin, imp,
         nan_go_left, is_cat, cats_left) = self._find_best_split(
            X, y, w, depth=depth, X_bins=X_bins)
        if feat is None:
            return _Node(value=val, n_samples=eff)

        mu_l = self._compute_mu_left(
            X[:, feat], feat, thresh, margin, is_cat, cats_left, nan_go_left)
        wl = w * mu_l
        wr = w * (1.0 - mu_l)

        eps = 1e-6 * w.max() if w.max() > 0 else 1e-10
        if self.min_train_weight_fraction > 0:
            eps = max(eps, self.min_train_weight_fraction * w.max())
        lm = wl > eps
        rm = wr > eps

        if lm.sum() < 1 or rm.sum() < 1:
            return _Node(value=val, n_samples=eff)

        left_bins = X_bins[lm] if X_bins is not None else None
        right_bins = X_bins[rm] if X_bins is not None else None
        left = self._build_depth_first(
            X[lm], y[lm], wl[lm], depth + 1, left_bins)
        right = self._build_depth_first(
            X[rm], y[rm], wr[rm], depth + 1, right_bins)

        return _Node(value=val, feature=feat, threshold=thresh, margin=margin,
                     left=left, right=right, n_samples=eff,
                     impurity_reduction=imp * eff,
                     nan_go_left=nan_go_left,
                     is_categorical=is_cat,
                     categories_left=cats_left)

    def _build_best_first(self, X, y, w, X_bins=None):
        import heapq

        root_val = self._compute_leaf_value(y, w)
        root = _Node(value=root_val, n_samples=w.sum())

        (feat, thresh, margin, imp,
         nan_go_left, is_cat, cats_left) = self._find_best_split(
            X, y, w, depth=0, X_bins=X_bins)
        if feat is None:
            return root

        counter = 0
        heap = []
        heapq.heappush(heap, (-imp, counter, root,
                               np.arange(len(y)), w.copy(), 0,
                               feat, thresh, margin,
                               nan_go_left, is_cat, cats_left))
        counter += 1
        n_leaves = 1

        while heap and n_leaves < self.max_leaves:
            (_, _, node, indices, weights, depth,
             feat, thresh, margin,
             nan_go_left, is_cat, cats_left) = heapq.heappop(heap)

            if not node.is_leaf or depth >= self.max_depth:
                continue

            X_n = X[indices]
            y_n = y[indices]

            mu_l = self._compute_mu_left(
                X_n[:, feat], feat, thresh, margin,
                is_cat, cats_left, nan_go_left)
            wl = weights * mu_l
            wr = weights * (1.0 - mu_l)

            eps = 1e-6 * weights.max() if weights.max() > 0 else 1e-10
            if self.min_train_weight_fraction > 0:
                eps = max(eps, self.min_train_weight_fraction * weights.max())
            lm = wl > eps
            rm = wr > eps
            if lm.sum() < 1 or rm.sum() < 1:
                continue

            node.feature = feat
            node.threshold = thresh
            node.margin = margin
            node.nan_go_left = nan_go_left
            node.is_categorical = is_cat
            node.categories_left = cats_left

            parent_imp = self._compute_impurity(y_n, weights)
            child_imp = (wl.sum() * self._compute_impurity(y_n, wl)
                         + wr.sum() * self._compute_impurity(y_n, wr)
                         ) / weights.sum()
            node.impurity_reduction = (parent_imp - child_imp) * weights.sum()

            node.left = _Node(
                value=self._compute_leaf_value(y_n[lm], wl[lm]),
                n_samples=wl[lm].sum())
            node.right = _Node(
                value=self._compute_leaf_value(y_n[rm], wr[rm]),
                n_samples=wr[rm].sum())
            n_leaves += 1

            for child, mask, cw in [(node.left, lm, wl[lm]),
                                     (node.right, rm, wr[rm])]:
                ci = indices[mask]
                if depth + 1 < self.max_depth and len(ci) >= 2:
                    child_bins = X_bins[ci] if X_bins is not None else None
                    (f, t, s, im,
                     ngl, isc, cl) = self._find_best_split(
                        X[ci], y[ci], cw, depth + 1, child_bins)
                    if f is not None and im > 0:
                        heapq.heappush(heap, (-im, counter, child,
                                               ci, cw, depth + 1,
                                               f, t, s,
                                               ngl, isc, cl))
                        counter += 1

        return root

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _detect_categorical(self, X_raw):
        n_features = X_raw.shape[1] if X_raw.ndim == 2 else 1
        is_cat = np.zeros(n_features, dtype=bool)

        if self.categorical_features is None:
            return is_cat

        if isinstance(self.categorical_features, str):
            if self.categorical_features != "auto":
                raise ValueError(
                    "categorical_features must be None, 'auto', or a list "
                    f"of int, got '{self.categorical_features}'")
            try:
                import pandas as pd
                if isinstance(X_raw, pd.DataFrame):
                    for i, dtype in enumerate(X_raw.dtypes):
                        if pd.api.types.is_categorical_dtype(dtype):
                            is_cat[i] = True
                        elif pd.api.types.is_object_dtype(dtype):
                            is_cat[i] = True
                        elif pd.api.types.is_integer_dtype(dtype):
                            if X_raw.iloc[:, i].nunique() <= self.max_cat_threshold:
                                is_cat[i] = True
                    return is_cat
            except ImportError:
                pass
            X_arr = np.asarray(X_raw)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            for i in range(n_features):
                col = X_arr[:, i]
                valid = col[~np.isnan(col)] if np.issubdtype(
                    col.dtype, np.floating) else col
                if len(np.unique(valid)) <= self.max_cat_threshold:
                    if np.issubdtype(col.dtype, np.integer):
                        is_cat[i] = True
            return is_cat

        for idx in self.categorical_features:
            if not 0 <= idx < n_features:
                raise ValueError(
                    f"categorical_features index {idx} out of range "
                    f"for {n_features} features")
            is_cat[idx] = True
        return is_cat

    def _encode_categorical(self, X_raw):
        try:
            import pandas as pd
            is_df = isinstance(X_raw, pd.DataFrame)
        except ImportError:
            is_df = False

        X = np.array(X_raw, dtype=object) if is_df else np.asarray(X_raw)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        X_out = np.empty((n_samples, n_features), dtype=np.float64)
        encoders = [None] * n_features

        for j in range(n_features):
            if not self.is_categorical_[j]:
                X_out[:, j] = np.asarray(X[:, j], dtype=np.float64)
                continue

            col = X[:, j]
            nan_mask = np.array([
                v is None or (isinstance(v, float) and np.isnan(v))
                or (isinstance(v, str) and v == "")
                for v in col
            ]) if col.dtype == object else np.isnan(
                np.asarray(col, dtype=np.float64))

            unique_vals = sorted(set(col[~nan_mask]))
            val_to_code = {v: float(i) for i, v in enumerate(unique_vals)}
            encoders[j] = val_to_code

            encoded = np.full(n_samples, np.nan, dtype=np.float64)
            for i in range(n_samples):
                if not nan_mask[i]:
                    encoded[i] = val_to_code[col[i]]
            X_out[:, j] = encoded

        return X_out, encoders

    def _encode_predict(self, X_raw):
        try:
            import pandas as pd
            is_df = isinstance(X_raw, pd.DataFrame)
        except ImportError:
            is_df = False

        if not self.is_categorical_.any():
            X = np.asarray(X_raw, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X

        X = np.array(X_raw, dtype=object) if is_df else np.asarray(X_raw)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        X_out = np.empty((n_samples, n_features), dtype=np.float64)

        for j in range(n_features):
            if not self.is_categorical_[j]:
                X_out[:, j] = np.asarray(X[:, j], dtype=np.float64)
                continue
            encoder = self.encoders_[j]
            col = X[:, j]
            encoded = np.full(n_samples, np.nan, dtype=np.float64)
            for i in range(n_samples):
                v = col[i]
                is_missing = (v is None
                              or (isinstance(v, float) and np.isnan(v))
                              or (isinstance(v, str) and v == ""))
                if not is_missing and v in encoder:
                    encoded[i] = encoder[v]
            X_out[:, j] = encoded

        return X_out

    def fit(self, X, y, sample_weight=None):
        self._fast_numeric_arrays_ = None
        X_raw = X

        if hasattr(X_raw, "shape"):
            ndim = X_raw.ndim if hasattr(X_raw, "ndim") else np.asarray(X_raw).ndim
        else:
            ndim = np.asarray(X_raw).ndim
        n_features = (X_raw.shape[1] if ndim == 2 else 1)
        self.is_categorical_ = self._detect_categorical(X_raw)

        if self.is_categorical_.any():
            X, self.encoders_ = self._encode_categorical(X_raw)
        else:
            X = np.asarray(X_raw, dtype=np.float64)
            self.encoders_ = [None] * n_features

        y = self._prepare_y(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.n_features_in_ = X.shape[1]
        self._rng = np.random.RandomState(self.random_state)
        X_bins, self._bin_thresholds_, self._bin_centers_ = \
            self._prebin_numeric_features(X)

        w = np.ones(len(y), dtype=np.float64)
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).ravel()

        c_tree = None
        if self.max_leaves is None:
            c_tree = self._try_build_c_depth_first(X, y, w, X_bins)

        if c_tree is not None:
            self.tree_ = c_tree
        elif self.max_leaves is not None:
            self.tree_ = self._build_best_first(X, y, w, X_bins)
        else:
            self.tree_ = self._build_depth_first(X, y, w, 0, X_bins)

        self.n_leaves_ = self._count_leaves(self.tree_)
        self.depth_ = self._tree_depth(self.tree_)
        self._compute_feature_importances()
        self._post_fit(X, y, w)

        return self

    # ------------------------------------------------------------------
    # Predict (default: weighted mean of leaf values)
    # ------------------------------------------------------------------

    def _collect_all_leaf_weights(self, X):
        """Vectorised DFS: compute every sample's weight at every leaf.

        Returns (leaf_values, leaf_weights) where leaf_weights has shape
        (n_samples, n_leaves) and leaf_values has shape (n_leaves,).
        """
        n = X.shape[0]
        n_leaves = getattr(self, 'n_leaves_', None) or self._count_leaves(self.tree_)
        if n_leaves == 0:
            return np.empty(0), np.empty((n, 0))

        leaf_values = np.empty(n_leaves, dtype=np.float64)
        leaf_weights = np.empty((n, n_leaves), dtype=np.float64)
        leaf_col = [0]

        stack = [(self.tree_, np.ones(n, dtype=np.float64))]
        while stack:
            node, weights = stack.pop()

            if node.is_leaf:
                col = leaf_col[0]
                leaf_values[col] = node.value
                leaf_weights[:, col] = weights
                leaf_col[0] += 1
                continue

            vals = X[:, node.feature]
            nan_mask = np.isnan(vals)

            if node.is_categorical:
                if node.categories_left is not None:
                    left_mask = np.array(
                        [v in node.categories_left for v in vals])
                else:
                    left_mask = np.zeros(n, dtype=bool)
                mu_l = np.where(left_mask, 1.0, 0.0)
            else:
                mu_l = self._membership_left(vals, node.threshold, node.margin)

            if nan_mask.any():
                mu_l = mu_l.copy()
                mu_l[nan_mask] = 1.0 if node.nan_go_left else 0.0

            stack.append((node.right, weights * (1.0 - mu_l)))
            stack.append((node.left, weights * mu_l))

        return leaf_values, leaf_weights

    def _leaf_nodes_in_weight_order(self):
        """Return leaves in the same DFS order as _collect_all_leaf_weights."""
        leaves = []
        stack = [self.tree_]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves.append(node)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return leaves

    def _numeric_tree_arrays(self):
        """Return flat arrays for fast prediction, or None for categorical trees."""
        cached = getattr(self, "_fast_numeric_arrays_", None)
        if cached is not None:
            return cached

        if getattr(self, "is_categorical_", None) is not None:
            if self.is_categorical_.any():
                return None

        nodes = []
        index_by_id = {}
        stack = [self.tree_]
        while stack:
            node = stack.pop()
            if node.is_categorical:
                return None
            index_by_id[id(node)] = len(nodes)
            nodes.append(node)
            if not node.is_leaf:
                stack.append(node.right)
                stack.append(node.left)

        n_nodes = len(nodes)
        features = np.full(n_nodes, -1, dtype=np.int64)
        thresholds = np.zeros(n_nodes, dtype=np.float64)
        margins = np.zeros(n_nodes, dtype=np.float64)
        lefts = np.full(n_nodes, -1, dtype=np.int64)
        rights = np.full(n_nodes, -1, dtype=np.int64)
        values = np.zeros(n_nodes, dtype=np.float64)
        nan_go_left = np.ones(n_nodes, dtype=np.bool_)

        for i, node in enumerate(nodes):
            values[i] = float(node.value)
            nan_go_left[i] = bool(node.nan_go_left)
            if node.is_leaf:
                continue
            features[i] = int(node.feature)
            thresholds[i] = float(node.threshold)
            margins[i] = float(node.margin)
            lefts[i] = index_by_id[id(node.left)]
            rights[i] = index_by_id[id(node.right)]

        arrays = (features, thresholds, margins, lefts, rights, values,
                  nan_go_left)
        self._fast_numeric_arrays_ = arrays
        return arrays

    def _predict_numeric_tree_fast(self, X):
        arrays = self._numeric_tree_arrays()
        if arrays is None:
            return None
        if _c_backend is not None:
            return _c_backend.predict_numeric_tree(X, *arrays)
        return _predict_numeric_tree_jit(X, *arrays)

    def predict(self, X):
        X = self._encode_predict(X)
        fast_pred = self._predict_numeric_tree_fast(X)
        if fast_pred is not None:
            return fast_pred
        leaf_values, leaf_weights = self._collect_all_leaf_weights(X)
        return leaf_weights @ leaf_values

    # ------------------------------------------------------------------
    # Prediction explanation
    # ------------------------------------------------------------------

    def explain_prediction(self, x, feature_names=None):
        """Explain a single prediction as a waterfall of split contributions.

        Returns dict with baseline, prediction, steps,
        feature_contributions, and leaves.
        """
        x_arr = np.atleast_2d(x)
        x_enc = self._encode_predict(x_arr)[0]

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(self.n_features_in_)]

        steps = []
        self._explain_recurse(x_enc, self.tree_, 1.0, steps, feature_names)

        baseline = self.tree_.value
        prediction = baseline + sum(s["contribution"] for s in steps)

        feature_contributions = {}
        for step in steps:
            fname = step["feature"]
            feature_contributions[fname] = (
                feature_contributions.get(fname, 0.0) + step["contribution"])
        feature_contributions = dict(sorted(
            feature_contributions.items(),
            key=lambda kv: abs(kv[1]), reverse=True))

        leaves = self._collect_leaves_with_paths(
            x_enc, self.tree_, 1.0, [], feature_names)
        leaves.sort(key=lambda l: l["weight"], reverse=True)

        return {
            "baseline": baseline,
            "prediction": prediction,
            "steps": steps,
            "feature_contributions": feature_contributions,
            "leaves": leaves,
        }

    def _collect_leaves_with_paths(self, x, node, weight, path,
                                   feature_names):
        if node.is_leaf:
            return [{"value": node.value, "weight": weight,
                     "contribution": weight * node.value,
                     "path": list(path)}]

        val = x[node.feature]
        fname = feature_names[node.feature]

        if np.isnan(val):
            mu_l = 1.0 if node.nan_go_left else 0.0
            go_l, go_r = node.nan_go_left, not node.nan_go_left
        elif node.is_categorical:
            goes_left = (node.categories_left is not None
                         and val in node.categories_left)
            mu_l = 1.0 if goes_left else 0.0
            go_l, go_r = goes_left, not goes_left
        else:
            mu_l = float(self._membership_left(
                np.array([val]), node.threshold, node.margin)[0])
            go_l = go_r = True

        def _step(direction, branch_w, child):
            return {"feature": fname, "feature_value": val,
                    "threshold": node.threshold, "margin": node.margin,
                    "mu_left": mu_l, "direction": direction,
                    "branch_weight": branch_w,
                    "node_value": node.value,
                    "child_value": child.value}

        leaves = []
        if go_l and go_r:
            leaves.extend(self._collect_leaves_with_paths(
                x, node.left, weight * mu_l,
                path + [_step("left", mu_l, node.left)], feature_names))
            leaves.extend(self._collect_leaves_with_paths(
                x, node.right, weight * (1.0 - mu_l),
                path + [_step("right", 1.0 - mu_l, node.right)],
                feature_names))
        elif go_l:
            leaves.extend(self._collect_leaves_with_paths(
                x, node.left, weight,
                path + [_step("left", 1.0, node.left)], feature_names))
        else:
            leaves.extend(self._collect_leaves_with_paths(
                x, node.right, weight,
                path + [_step("right", 1.0, node.right)], feature_names))
        return leaves

    def _explain_recurse(self, x, node, weight, steps, feature_names):
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

        left_val = node.left.value
        right_val = node.right.value

        if go_l and go_r:
            contribution = weight * (
                mu_l * left_val + (1.0 - mu_l) * right_val - node.value)
            steps.append(dict(
                feature=fname, feature_idx=node.feature,
                feature_value=val, threshold=node.threshold,
                margin=node.margin, mu_left=mu_l, weight=weight,
                contribution=contribution, description=desc))
            self._explain_recurse(
                x, node.left, weight * mu_l, steps, feature_names)
            self._explain_recurse(
                x, node.right, weight * (1.0 - mu_l), steps, feature_names)
        elif go_l:
            contribution = weight * (left_val - node.value)
            steps.append(dict(
                feature=fname, feature_idx=node.feature,
                feature_value=val, threshold=node.threshold,
                margin=node.margin, mu_left=mu_l, weight=weight,
                contribution=contribution, description=desc))
            self._explain_recurse(
                x, node.left, weight, steps, feature_names)
        else:
            contribution = weight * (right_val - node.value)
            steps.append(dict(
                feature=fname, feature_idx=node.feature,
                feature_value=val, threshold=node.threshold,
                margin=node.margin, mu_left=mu_l, weight=weight,
                contribution=contribution, description=desc))
            self._explain_recurse(
                x, node.right, weight, steps, feature_names)

    # ------------------------------------------------------------------
    # Tree inspection
    # ------------------------------------------------------------------

    @staticmethod
    def _count_leaves(node):
        if node.is_leaf:
            return 1
        return (_count_leaves_r(node.left)
                + _count_leaves_r(node.right))

    @staticmethod
    def _tree_depth(node):
        if node.is_leaf:
            return 0
        return 1 + max(_tree_depth_r(node.left),
                       _tree_depth_r(node.right))

    def _compute_feature_importances(self):
        imp = np.zeros(self.n_features_in_, dtype=np.float64)
        self._accumulate_importances(self.tree_, imp)
        total = imp.sum()
        self.feature_importances_ = imp / total if total > 0 else imp

    def _accumulate_importances(self, node, imp):
        if node.is_leaf:
            return
        imp[node.feature] += node.impurity_reduction
        self._accumulate_importances(node.left, imp)
        self._accumulate_importances(node.right, imp)

    def plot_tree(self, *, feature_names=None, ax=None, fontsize=9,
                  node_color="#E8F4FD", leaf_color="#C8E6C9",
                  edge_color="#555555"):
        """Plot the tree structure using matplotlib."""
        import matplotlib.pyplot as plt

        if not hasattr(self, "tree_"):
            raise RuntimeError("Call fit() before plot_tree().")

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(self.n_features_in_)]

        if ax is None:
            depth = self._tree_depth(self.tree_)
            width = 2 ** depth
            fig, ax = plt.subplots(
                figsize=(max(8, width * 1.6), max(4, (depth + 1) * 2)))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        positions = {}
        self._layout(self.tree_, 0.0, 1.0, 0,
                     self._tree_depth(self.tree_), positions)
        self._draw_edges(self.tree_, positions, ax, edge_color)
        self._draw_nodes(self.tree_, positions, ax, feature_names,
                         fontsize, node_color, leaf_color)
        return ax

    def _layout(self, node, x_lo, x_hi, depth, max_depth, positions):
        x = (x_lo + x_hi) / 2
        y = 1.0 - depth / (max_depth + 1) - 0.05
        positions[id(node)] = (x, y)
        if not node.is_leaf:
            self._layout(node.left, x_lo, x, depth + 1, max_depth, positions)
            self._layout(node.right, x, x_hi, depth + 1, max_depth, positions)

    def _draw_edges(self, node, positions, ax, edge_color):
        if node.is_leaf:
            return
        x, y = positions[id(node)]
        for child in (node.left, node.right):
            cx, cy = positions[id(child)]
            ax.plot([x, cx], [y, cy], color=edge_color, linewidth=1.2,
                    zorder=1)
        self._draw_edges(node.left, positions, ax, edge_color)
        self._draw_edges(node.right, positions, ax, edge_color)

    @staticmethod
    def _draw_nodes(node, positions, ax, feature_names, fontsize,
                    node_color, leaf_color):
        stack = [node]
        while stack:
            n = stack.pop()
            x, y = positions[id(n)]
            if n.is_leaf:
                label = f"val={n.value:.3f}\nn={n.n_samples:.1f}"
                color = leaf_color
            else:
                fname = feature_names[n.feature]
                if n.is_categorical and n.categories_left is not None:
                    cats = sorted(n.categories_left)
                    cats_str = ",".join(str(int(c)) for c in cats)
                    if len(cats_str) > 20:
                        cats_str = cats_str[:17] + "..."
                    label = (f"{fname} ∈ {{{cats_str}}}\n"
                             f"n={n.n_samples:.1f}")
                else:
                    label = (f"{fname} <= {n.threshold:.3f}\n"
                             f"m={n.margin:.3f}  n={n.n_samples:.1f}")
                color = node_color
                stack.append(n.right)
                stack.append(n.left)
            bbox = dict(boxstyle="round,pad=0.3", facecolor=color,
                        edgecolor="#999999", linewidth=0.8)
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=fontsize, bbox=bbox, zorder=2)

    def plot_feature_contributions(self, x, *, feature_names=None,
                                      actual=None, ax=None,
                                      positive_color="#2ca02c",
                                      negative_color="#d62728"):
        """Waterfall chart of per-feature contributions for a single sample.

        Parameters
        ----------
        x : array-like of shape (n_features,)
        feature_names : list of str, optional
        actual : float, optional
            Ground-truth value shown as a dashed line.
        ax : matplotlib Axes, optional
        positive_color : str
        negative_color : str

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        exp = self.explain_prediction(x, feature_names=feature_names)
        contribs = exp["feature_contributions"]
        names = list(contribs.keys())
        vals = np.array([contribs[n] for n in names])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names))))

        running = exp["baseline"]
        starts = np.empty(len(vals))
        for i, v in enumerate(vals):
            starts[i] = running
            running += v

        colors = [positive_color if v >= 0 else negative_color for v in vals]
        ax.barh(range(len(names)), vals, left=starts, color=colors,
                edgecolor="white", height=0.6)
        ax.axvline(exp["baseline"], color="gray", ls="--", lw=1,
                   label=f"Baseline {exp['baseline']:.2f}")
        ax.axvline(exp["prediction"], color="black", ls="-", lw=1.5,
                   label=f"Prediction {exp['prediction']:.2f}")
        if actual is not None:
            ax.axvline(actual, color="#1f77b4", ls=":", lw=1.5,
                       label=f"Actual {actual:.2f}")

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="best")
        return ax

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"


# Module-level helpers for static-method recursion
def _count_leaves_r(node):
    if node.is_leaf:
        return 1
    return _count_leaves_r(node.left) + _count_leaves_r(node.right)


def _tree_depth_r(node):
    if node.is_leaf:
        return 0
    return 1 + max(_tree_depth_r(node.left), _tree_depth_r(node.right))


FuzzyDTree._count_leaves = staticmethod(_count_leaves_r)
FuzzyDTree._tree_depth = staticmethod(_tree_depth_r)
