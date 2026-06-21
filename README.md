# FuzzyDTree

**Fuzzy Decision Trees for Regression and Classification**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Documentation: https://guyko81.github.io/fuzzydtree/

FuzzyDTree replaces the hard binary splits of traditional decision trees with smooth **compact-support membership functions**. Instead of routing each sample to a single leaf, every observation partially belongs to multiple leaves, weighted by its degree of membership at each split. The result is a single, interpretable tree that produces smooth, continuous prediction surfaces — without the ensemble overhead.

**Install:** `pip install FuzzyDTree`

**Dependencies:** `numpy`, `numba`, `scikit-learn`

## Models

| Model | Task | Loss Function | Key Output |
|-------|------|---------------|------------|
| `FuzzyTreeRegressor` | Regression | MSE | Continuous predictions |
| `FuzzyTreeClassifier` | Classification | Logit-MSE/Gini splits | Class labels and probabilities |

Both models share the same core mechanics — S-curve splits, per-split margin optimisation, histogram-accelerated JIT training — and follow the scikit-learn estimator API (`fit`, `predict`, `score`, `get_params`, `set_params`).

## Quick Start

### Regression

```python
from fuzzydtree import FuzzyTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = FuzzyTreeRegressor(max_depth=8)
model.fit(X_train, y_train)

print(f"R² = {model.score(X_test, y_test):.4f}")
```

### Classification

```python
from fuzzydtree import FuzzyTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = FuzzyTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

print(f"Accuracy = {clf.score(X_test, y_test):.1%}")
probs = clf.predict_proba(X_test)  # shape: (n_samples, n_classes)
```

### Multiclass Classification

```python
from fuzzydtree import FuzzyTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

clf = FuzzyTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

print(f"Accuracy = {clf.score(X_test, y_test):.1%}")
print(clf.classes_)              # array([0, 1, 2])
probs = clf.predict_proba(X_test) # shape: (n_samples, 3)
```

## How It Works

### Fuzzy Splits

At each internal node, a quadratic S-curve with compact support assigns a continuous degree of belonging to the left and right children:

```
μ_left(x) = S((x - threshold) / margin)
μ_right(x) = 1 - μ_left(x)
```

where S is a smooth quadratic spline that transitions from 1 to 0 over the interval `[threshold - margin, threshold + margin]` and is exactly 0 or 1 outside it. Samples far from the split boundary contribute zero weight to the wrong child, eliminating the weight dilution problem that affects sigmoid-based fuzzy trees in deep configurations. The final prediction for each sample is the weighted sum of all leaf values, where the weights are products of membership values along each path from root to leaf.

### Per-Split Margin Optimisation

The transition width (margin) is not a global hyperparameter — it is optimised independently at every split node. During training, a geometric grid of margin candidates (ranging from 0.4× to 20× the feature's standard deviation) is evaluated in parallel alongside threshold candidates. The (threshold, margin) pair that maximises the split criterion is selected. This allows the tree to learn sharp boundaries where the data demands them and gradual transitions where they improve generalisation.

### Joint Leaf Value Optimisation (Regressor)

After the tree structure is grown, all leaf values are re-solved jointly via weighted least-squares using the full fuzzy weight matrix, followed by a coordinate-refinement pass over numeric thresholds and margins. Because each sample contributes partially to every leaf it reaches, this global optimisation step accounts for the interactions between overlapping leaf regions and produces substantially better predictions than computing leaf means independently.

### Logit-MSE Splitting (Classifier)

The classifier uses a dedicated split kernel that operates on per-class fuzzy weight sums. By default, splits are selected with a logit-MSE objective, equivalent to multiclass Gini for split ranking. Leaf probabilities are then estimated from the fuzzy-weighted class frequencies in each leaf and mixed by each sample's leaf weights. This extends naturally to multiclass problems and produces probabilistic outputs via `predict_proba`.

## Parameters

### FuzzyTreeRegressor

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 5 | Maximum tree depth |
| `max_leaves` | None | Maximum number of leaf nodes (enables best-first growth when set) |
| `min_samples_leaf` | 1.0 | Minimum effective (weighted) sample count per child |
| `max_features` | None | Number of features considered per split (`None` = all; also accepts `'sqrt'`, `'log2'`, `int`, or `float`) |
| `margin_grid_size` | 10 | Number of margin candidates in the geometric grid per split |
| `margin_depth_decay` | 1.0 | Exponential decay factor for the margin upper bound at each depth level (unnecessary with S-curve splits; kept for compatibility) |
| `max_bins` | 256 | Maximum histogram bins for threshold candidates |
| `min_train_weight_fraction` | 0.01 | Minimum relative training weight retained when propagating fuzzy child weights |
| `prebin_numeric` | True | Pre-bin numeric features once before split search for faster histogram-based training |
| `optimize_leaf_values` | True | Re-solve all leaf values jointly via weighted least-squares after tree construction |
| `optimize_split_gain` | False | Use exact two-leaf least-squares objective for split evaluation (slower and experimental) |
| `leaf_l2` | 0.1 | L2 regularisation strength for the joint leaf-value optimisation |
| `leaf_l2_mode` | `'centered'` | Ridge target for leaf optimisation: `'centered'` shrinks toward the training target mean; `'zero'` shrinks toward zero |
| `split_gain_l2` | 0.0 | Ridge strength for `optimize_split_gain=True`; use `'leaf_l2'` to align split ranking with leaf optimisation |
| `refine_splits` | True | Run a post-fit coordinate-refinement pass over numeric thresholds and margins using training MSE |
| `refine_splits_max_iter` | 1 | Maximum split-refinement passes |
| `refine_splits_candidates` | 4 | Number of local threshold/margin candidates tried per numeric split during refinement |
| `categorical_features` | None | Indices of categorical columns, or `'auto'` for automatic detection |
| `max_cat_threshold` | 64 | Maximum number of unique values for a feature to be treated as categorical (used with `'auto'` detection) |
| `random_state` | None | Random seed for reproducibility when using feature subsampling |

### FuzzyTreeClassifier

The classifier accepts all base tree parameters listed above, excluding the regressor-specific parameters (`optimize_leaf_values`, `optimize_split_gain`, `leaf_l2`, `leaf_l2_mode`, `split_gain_l2`, `refine_splits`, `refine_splits_max_iter`, and `refine_splits_candidates`).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split_criterion` | `'logit_mse'` | Split objective: `'logit_mse'`/`'gini'` for the local least-squares finite-logit objective, or `'entropy'` for cross-entropy information gain |
| `leaf_prediction` | `'frequency'` | Post-fit probability model: `'frequency'` keeps per-leaf class frequencies, `'logit_mse'` globally refits leaf logits via ridge least-squares, and `'logit_ce'` refits leaf logits with softmax cross-entropy |
| `leaf_logit_l2` | `0.01` | L2 regularisation for logit leaf refits |
| `leaf_logit_target` | `4.0` | Finite one-vs-rest target logit magnitude for the logit-MSE leaf refit |
| `leaf_logit_ce_max_iter` | `100` | Maximum line-search iterations for `leaf_prediction='logit_ce'` |
| `leaf_logit_ce_lr` | `1.0` | Initial line-search step size for `leaf_prediction='logit_ce'` |
| `leaf_logit_ce_tol` | `1e-6` | Relative improvement tolerance for `leaf_prediction='logit_ce'` |

## Features

- **scikit-learn compatible API** — `fit`, `predict`, `score`, `get_params`, `set_params`
- **Probabilistic classification** — `predict_proba` returns calibrated class probabilities from the fuzzy leaf mixture
- **Classifier log-odds explanations** — `explain_prediction_log_odds()` and `plot_log_odds_contributions()` show how features move class evidence and probability
- **Missing value handling** — NaN values are routed deterministically using a learned direction per split
- **Categorical feature support** — binary partition of categories via the sorted-gradient heuristic, with optional auto-detection
- **Feature importances** — normalised impurity reduction, accessible via `feature_importances_`
- **Tree visualisation** — `plot_tree()` renders the full tree structure with matplotlib
- **Prediction explanations** — `explain_prediction()` decomposes a single prediction into per-feature contributions
- **Waterfall plots** — `plot_feature_contributions()` visualises the per-feature contribution breakdown
- **JIT-compiled training** — Numba-parallelised, histogram-based split search with pre-binned features

## Notebooks

| Notebook | Description |
|----------|-------------|
| `regressor_demo.ipynb` | Synthetic surface comparison, California Housing benchmark, feature importance and waterfall plots |
| `classifier_demo.ipynb` | Decision boundaries, probability heatmaps, multiclass examples |

## Documentation

Open `docs/index.html` for the full documentation with interactive demos, API reference, and generated charts.

To regenerate all chart images:

```
python docs/generate_charts.py
```

## License

MIT
