"""
Fuzzy Tree Classifier.

Fuzzy decision tree for classification. Uses the base FuzzyDTree
mechanics with histogram-accelerated split kernels. The default split
criterion is logit-MSE/Gini; entropy remains available as an option.
"""

import numpy as np
from numba import njit, prange

from ._tree import FuzzyDTree, _membership_value


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

    All other parameters are inherited from FuzzyDTree.
    """

    def __init__(self, *, split_criterion="logit_mse",
                 leaf_prediction="frequency", leaf_logit_l2=1e-2,
                 leaf_logit_target=4.0, leaf_logit_ce_max_iter=100,
                 leaf_logit_ce_lr=1.0, leaf_logit_ce_tol=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.split_criterion = split_criterion
        self.leaf_prediction = leaf_prediction
        self.leaf_logit_l2 = leaf_logit_l2
        self.leaf_logit_target = leaf_logit_target
        self.leaf_logit_ce_max_iter = leaf_logit_ce_max_iter
        self.leaf_logit_ce_lr = leaf_logit_ce_lr
        self.leaf_logit_ce_tol = leaf_logit_ce_tol

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["split_criterion"] = self.split_criterion
        params["leaf_prediction"] = self.leaf_prediction
        params["leaf_logit_l2"] = self.leaf_logit_l2
        params["leaf_logit_target"] = self.leaf_logit_target
        params["leaf_logit_ce_max_iter"] = self.leaf_logit_ce_max_iter
        params["leaf_logit_ce_lr"] = self.leaf_logit_ce_lr
        params["leaf_logit_ce_tol"] = self.leaf_logit_ce_tol
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

    def _post_fit(self, X, y, w):
        """Compute per-leaf class probability distributions."""
        _, leaf_weights = self._collect_all_leaf_weights(X)
        n_leaves = leaf_weights.shape[1]
        self.leaf_probs_ = np.zeros((n_leaves, self.n_classes_),
                                    dtype=np.float64)
        for j in range(n_leaves):
            wj = w * leaf_weights[:, j]
            for c in range(self.n_classes_):
                self.leaf_probs_[j, c] = wj[y == c].sum()
            row_sum = self.leaf_probs_[j].sum()
            if row_sum > 1e-15:
                self.leaf_probs_[j] /= row_sum
            else:
                self.leaf_probs_[j] = 1.0 / self.n_classes_

        self._leaf_prob_by_node_id_ = {
            id(node): prob.copy()
            for node, prob in zip(self._leaf_nodes_in_weight_order(),
                                  self.leaf_probs_)
        }
        self._node_class_probs_ = {}
        self._compute_node_class_probs(self.tree_, X, y, w)

        self.leaf_logits_ = None
        if self.leaf_prediction == "logit_mse":
            self._optimize_leaf_logits(leaf_weights, y, w)
        elif self.leaf_prediction == "logit_ce":
            self._optimize_leaf_logits_ce(leaf_weights, y, w)

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

        eps = 1e-6 * w.max() if w.max() > 0 else 1e-10
        if self.min_train_weight_fraction > 0:
            eps = max(eps, self.min_train_weight_fraction * w.max())
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
