"""Generate all chart images for the docs pages.

Run from the repo root:
    python docs/generate_charts.py

Outputs PNGs to docs/img/.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import (load_breast_cancer, load_iris, make_moons,
                              make_circles, make_classification, make_blobs)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state
from sklearn.base import clone

from fuzzydtree import FuzzyTreeRegressor, FuzzyTreeClassifier

IMG_DIR = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(IMG_DIR, exist_ok=True)

DPI = 150

# ======================================================================
# REGRESSOR CHARTS
# ======================================================================

# --- Synthetic data ---
x0g = np.arange(-1, 1, .1)
x1g = np.arange(-1, 1, .1)
x0g, x1g = np.meshgrid(x0g, x1g)
y_truth = x0g**2 - x1g**2 + x1g - 1

rng = check_random_state(0)
X_train_r = rng.uniform(-1, 1, size=(100, 2))
y_train_r = X_train_r[:, 0]**2 - X_train_r[:, 1]**2 + X_train_r[:, 1] - 1
X_test_r = rng.uniform(-1, 1, size=(100, 2))
y_test_r = X_test_r[:, 0]**2 - X_test_r[:, 1]**2 + X_test_r[:, 1] - 1

est_fuzzy = FuzzyTreeRegressor(max_depth=5)
est_fuzzy.fit(X_train_r, y_train_r)
est_tree = DecisionTreeRegressor(max_depth=5)
est_tree.fit(X_train_r, y_train_r)
est_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
est_rf.fit(X_train_r, y_train_r)

# 1. Surface comparison
X_grid = np.c_[x0g.ravel(), x1g.ravel()]
y_fuzzy = est_fuzzy.predict(X_grid).reshape(x0g.shape)
y_tree = est_tree.predict(X_grid).reshape(x0g.shape)
y_rf = est_rf.predict(X_grid).reshape(x0g.shape)

fig = plt.figure(figsize=(12, 10))
for i, (y_surf, score, title) in enumerate([
    (y_truth, None, "Ground Truth"),
    (y_fuzzy, est_fuzzy.score(X_test_r, y_test_r), "FuzzyTreeRegressor"),
    (y_tree, est_tree.score(X_test_r, y_test_r), "DecisionTreeRegressor"),
    (y_rf, est_rf.score(X_test_r, y_test_r), "RandomForestRegressor"),
]):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.plot_surface(x0g, x1g, y_surf, rstride=1, cstride=1, color="green", alpha=0.5)
    ax.scatter(X_train_r[:, 0], X_train_r[:, 1], y_train_r, s=5)
    if score is not None:
        ax.text(-0.7, 1, 0.2, f"$R^2 = {score:.4f}$", "x", fontsize=12)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "tree_surface_comparison.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  tree_surface_comparison.png")

# 2. Feature importances
feature_names_r = ["$X_0$", "$X_1$"]
imp = est_fuzzy.feature_importances_
fig, ax = plt.subplots(figsize=(4, 3))
ax.barh(feature_names_r, imp, color=["steelblue", "coral"])
ax.set_xlabel("Importance (normalised impurity reduction)")
ax.set_title("FuzzyTreeRegressor feature importances")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "tree_feature_importances.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  tree_feature_importances.png")

# 3. Tree structure
fig, ax = plt.subplots(figsize=(14, 8))
est_fuzzy.plot_tree(feature_names=["$X_0$", "$X_1$"], ax=ax, fontsize=8)
ax.set_title("FuzzyTreeRegressor (depth=5)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "tree_structure.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  tree_structure.png")

# 4. California Housing — predictions + waterfall
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X_cal, y_cal = data.data, data.target
cal_names = list(data.feature_names)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)

ft_cal = FuzzyTreeRegressor(max_depth=8)
ft_cal.fit(X_tr_c, y_tr_c)

# Waterfall for two contrasting samples
idx_high = np.argmax(y_te_c)
idx_low = np.argmin(y_te_c[y_te_c > 0.5]) + np.where(y_te_c > 0.5)[0][0]
# Find a low-value sample
low_mask = y_te_c < 1.0
if low_mask.any():
    idx_low = np.where(low_mask)[0][0]
else:
    idx_low = np.argmin(y_te_c)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cases = [
    (idx_high, f"High-value home (actual={y_te_c[idx_high]:.2f})"),
    (idx_low, f"Low-value home (actual={y_te_c[idx_low]:.2f})"),
]
for ax, (idx, label) in zip(axes, cases):
    ft_cal.plot_feature_contributions(
        X_te_c[idx], feature_names=cal_names, actual=y_te_c[idx], ax=ax)
    ax.set_xlabel("Predicted value ($100k)")
    ax.set_title(label, fontsize=11)
fig.suptitle("FuzzyTreeRegressor — Feature contribution waterfall (California Housing)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "tree_explainability_waterfall.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  tree_explainability_waterfall.png")

# 5. Predicted vs actual scatter
y_pred_cal = ft_cal.predict(X_te_c)
dt_cal = DecisionTreeRegressor(max_depth=8)
dt_cal.fit(X_tr_c, y_tr_c)
y_pred_dt = dt_cal.predict(X_te_c)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_p, name, r2 in [
    (ax1, y_pred_cal, "FuzzyTreeRegressor", ft_cal.score(X_te_c, y_te_c)),
    (ax2, y_pred_dt, "DecisionTreeRegressor", dt_cal.score(X_te_c, y_te_c)),
]:
    ax.scatter(y_te_c, y_p, s=4, alpha=0.4)
    lo, hi = min(y_te_c.min(), y_p.min()), max(y_te_c.max(), y_p.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"{name}  (R²={r2:.4f})")
    ax.set_aspect("equal")
fig.suptitle("California Housing — Predicted vs Actual", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "tree_predicted_vs_actual.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  tree_predicted_vs_actual.png")


# ======================================================================
# CLASSIFIER CHARTS
# ======================================================================

def multiclass_brier(y_true, proba, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    target = np.zeros_like(proba)
    target[np.arange(len(y_true)), [class_to_idx[v] for v in y_true]] = 1.0
    return np.mean(np.sum((proba - target) ** 2, axis=1))


def classifier_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    classes = model.classes_ if hasattr(model, "classes_") else np.unique(y_train)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "log_loss": log_loss(y_test, proba, labels=classes),
        "brier": multiclass_brier(y_test, proba, classes),
    }, model


# 6. Real classification benchmarks: Breast Cancer + Iris
benchmark_datasets = []
cancer = load_breast_cancer()
benchmark_datasets.append(("Breast Cancer", cancer.data, cancer.target))
iris = load_iris()
benchmark_datasets.append(("Iris", iris.data, iris.target))

benchmark_models = [
    ("FuzzyDTree", FuzzyTreeClassifier(
        max_depth=5, min_samples_leaf=3, margin_grid_size=8, random_state=0)),
    ("DecisionTree", DecisionTreeClassifier(max_depth=5, random_state=0)),
    ("RandomForest", RandomForestClassifier(
        max_depth=5, n_estimators=100, random_state=0)),
    ("kNN", KNeighborsClassifier(n_neighbors=5)),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=False)
metric_specs = [
    ("accuracy", "Accuracy", "higher is better"),
    ("log_loss", "Log-loss", "lower is better"),
    ("brier", "Brier", "lower is better"),
]
colors = ["#059669", "#64748b", "#2563eb", "#f59e0b"]

benchmark_results = {}
for row, (ds_name, X_raw, y_raw) in enumerate(benchmark_datasets):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=0.35, random_state=42, stratify=y_raw)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    benchmark_results[ds_name] = {}

    for model_name, model in benchmark_models:
        metrics, fitted = classifier_metrics(clone(model), X_tr, X_te, y_tr, y_te)
        benchmark_results[ds_name][model_name] = metrics

    for col, (key, label, subtitle) in enumerate(metric_specs):
        ax = axes[row, col]
        vals = [benchmark_results[ds_name][name][key]
                for name, _ in benchmark_models]
        ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.9)
        ax.set_title(f"{ds_name}: {label}\n{subtitle}", fontsize=11)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels([name for name, _ in benchmark_models],
                           rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Classifier Benchmarks on Real Datasets", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "classifier_real_benchmarks.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  classifier_real_benchmarks.png")

# 7. Breast Cancer feature importances
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.35, random_state=42,
    stratify=cancer.target)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
cancer_clf = FuzzyTreeClassifier(
    max_depth=5, min_samples_leaf=3, margin_grid_size=8, random_state=0)
cancer_clf.fit(X_tr_s, y_tr)
importances = cancer_clf.feature_importances_
top_idx = np.argsort(importances)[-12:][::-1]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(top_idx))
ax.barh(y_pos, importances[top_idx], color="#059669", alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels([cancer.feature_names[i] for i in top_idx])
ax.invert_yaxis()
ax.set_xlabel("Normalized split-objective reduction")
ax.set_title("FuzzyTreeClassifier Feature Importances (Breast Cancer)")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "classifier_feature_importances.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  classifier_feature_importances.png")

# 8. Breast Cancer log-odds contribution waterfall
probs = cancer_clf.predict_proba(X_te_s)
target_class = 0  # malignant in sklearn's breast cancer dataset
candidates = np.where(y_te == target_class)[0]
sample_idx = candidates[np.argmax(probs[candidates, target_class])]
fig, ax = plt.subplots(figsize=(10, 6))
cancer_clf.plot_log_odds_contributions(
    X_te_s[sample_idx],
    class_label=target_class,
    feature_names=cancer.feature_names,
    max_features=10,
    ax=ax,
)
ax.set_title("Breast Cancer: Feature Contributions to Malignant Log-Odds")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "classifier_log_odds_waterfall.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  classifier_log_odds_waterfall.png")

# 9. Iris multiclass pairplot with per-cell decision boundaries
iris_feature_names = list(iris.feature_names)
iris_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
iris_cmap = ListedColormap(iris_colors)
X_iris = iris.data
y_iris = iris.target
n_features = X_iris.shape[1]
iris_full_clf = FuzzyTreeClassifier(
    max_depth=5, min_samples_leaf=3, margin_grid_size=8, random_state=0)
iris_full_clf.fit(X_iris, y_iris)
iris_full_probs = iris_full_clf.predict_proba(X_iris)
iris_full_pred = np.argmax(iris_full_probs, axis=1)

fig, axes = plt.subplots(n_features, n_features, figsize=(11, 10))

for row in range(n_features):
    for col in range(n_features):
        ax = axes[row, col]
        x = X_iris[:, col]
        y_feat = X_iris[:, row]

        if row == col:
            x_grid = np.linspace(x.min() - 0.2, x.max() + 0.2, 240)
            for class_idx, color in enumerate(iris_colors):
                vals = x[y_iris == class_idx]
                bw = max(0.08, 0.25 * np.std(vals))
                density = np.exp(
                    -0.5 * ((x_grid[:, None] - vals[None, :]) / bw) ** 2)
                density = density.mean(axis=1) / (bw * np.sqrt(2 * np.pi))
                ax.fill_between(x_grid, density, color=color, alpha=0.18)
                ax.plot(x_grid, density, color=color, lw=1.3)
            ax.set_xlim(x_grid.min(), x_grid.max())
            ax.set_yticks([])
        else:
            x_pad = 0.08 * (x.max() - x.min())
            y_pad = 0.08 * (y_feat.max() - y_feat.min())
            xx, yy = np.meshgrid(
                np.linspace(x.min() - x_pad, x.max() + x_pad, 180),
                np.linspace(y_feat.min() - y_pad, y_feat.max() + y_pad, 180),
            )
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            sample_xy = X_iris[:, [col, row]]
            span = np.array([x.max() - x.min(), y_feat.max() - y_feat.min()])
            bandwidth = max(0.18 * np.linalg.norm(span), 1e-6)
            diff = grid_points[:, None, :] - sample_xy[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            weights = np.exp(-0.5 * d2 / (bandwidth * bandwidth)) + 1e-12
            smooth_probs = weights @ iris_full_probs
            smooth_probs /= weights.sum(axis=1, keepdims=True)
            Z = np.argmax(smooth_probs, axis=1).reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                        cmap=iris_cmap, alpha=0.12)
            ax.contour(xx, yy, Z, levels=[0.5, 1.5],
                       colors="#333333", linewidths=0.8, alpha=0.55)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())

            for class_idx, color in enumerate(iris_colors):
                mask = y_iris == class_idx
                ax.scatter(x[mask], y_feat[mask], s=22, color=color,
                           edgecolors="white", linewidths=0.35,
                           alpha=0.95)
                missed = mask & (iris_full_pred != y_iris)
                if np.any(missed):
                    ax.scatter(x[missed], y_feat[missed], s=50,
                               facecolors="none", edgecolors="#111111",
                               linewidths=0.8, alpha=0.85)

        if row == n_features - 1:
            ax.set_xlabel(iris_feature_names[col], fontsize=9)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(iris_feature_names[row], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8, length=3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

handles = [
    plt.Line2D([0], [0], marker="o", color="w", label=name,
               markerfacecolor=color, markeredgecolor="white", markersize=7)
    for name, color in zip(iris.target_names, iris_colors)
]
fig.legend(handles=handles, title="target", loc="center right",
           bbox_to_anchor=(1.02, 0.52), frameon=False)
fig.suptitle("Iris Multiclass Decision Boundaries by Feature Pair",
             fontsize=15, y=0.995)
plt.tight_layout(rect=[0, 0, 0.93, 0.97])
plt.savefig(os.path.join(IMG_DIR, "classifier_iris_probabilities.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  classifier_iris_probabilities.png")

cm_bg = ListedColormap(["#FFAAAA", "#AAAAFF"])
cm_pts = ListedColormap(["#FF0000", "#0000FF"])
h = 0.02

# --- Datasets ---
X_moons, y_moons = make_moons(n_samples=300, noise=0.25, random_state=0)
X_circles, y_circles = make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=1)
X_lin, y_lin = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                    n_informative=2, random_state=2,
                                    n_clusters_per_class=1)
rng2 = np.random.RandomState(2)
X_lin += 2 * rng2.uniform(size=X_lin.shape)

datasets = [
    ("Moons", X_moons, y_moons),
    ("Circles", X_circles, y_circles),
    ("Linear", X_lin, y_lin),
]

classifiers = [
    ("FuzzyDTree (d=5)",  FuzzyTreeClassifier(max_depth=5)),
    ("FuzzyDTree (d=10)", FuzzyTreeClassifier(max_depth=10)),
    ("DecisionTree",     DecisionTreeClassifier(max_depth=5)),
    ("Random Forest",    RandomForestClassifier(max_depth=5, n_estimators=50, random_state=0)),
    ("kNN (k=5)",        KNeighborsClassifier(n_neighbors=5)),
    ("SVM (RBF)",        SVC(gamma="auto", probability=True)),
]

# 10. Decision boundaries grid
n_cls = len(classifiers)
n_ds = len(datasets)
fig, axes = plt.subplots(n_cls, n_ds, figsize=(4 * n_ds, 3 * n_cls))

for col, (ds_name, X_raw, y_raw) in enumerate(datasets):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_raw, test_size=0.3, random_state=42)
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh_pts = np.c_[xx.ravel(), yy.ravel()]

    for row, (clf_name, clf_template) in enumerate(classifiers):
        ax = axes[row, col]
        try:
            clf = clone(clf_template)
        except Exception:
            clf = clf_template.__class__(**clf_template.get_params())
        clf.fit(X_tr, y_tr)

        if hasattr(clf, "predict_proba"):
            Z = clf.predict_proba(mesh_pts)[:, 1]
        else:
            Z = clf.predict(mesh_pts).astype(float)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.8)
        ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5, linestyles="--")
        ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=cm_pts,
                   edgecolors="k", s=15, linewidths=0.5, alpha=0.6)
        ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap=cm_pts,
                   edgecolors="k", s=25, linewidths=0.8, marker="D", alpha=0.9)

        score = clf.score(X_te, y_te)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(x_max - 0.1, y_min + 0.1, f"{score:.0%}",
                fontsize=11, fontweight="bold", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
        if row == 0:
            ax.set_title(ds_name, fontsize=13, fontweight="bold")
        if col == 0:
            ax.set_ylabel(clf_name, fontsize=11)

fig.suptitle("Decision Boundaries — FuzzyTreeClassifier vs Standard Classifiers",
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "classifier_decision_boundaries.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  classifier_decision_boundaries.png")

# 11. Probability heatmap (Moons)
scaler = StandardScaler()
X_m = scaler.fit_transform(X_moons)
X_tr, X_te, y_tr, y_te = train_test_split(X_m, y_moons, test_size=0.3, random_state=42)

x_min, x_max = X_m[:, 0].min() - 0.5, X_m[:, 0].max() + 0.5
y_min, y_max = X_m[:, 1].min() - 0.5, X_m[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
mesh = np.c_[xx.ravel(), yy.ravel()]

fuzzy_clf = FuzzyTreeClassifier(max_depth=5)
fuzzy_clf.fit(X_tr, y_tr)
hard_clf = DecisionTreeClassifier(max_depth=5)
hard_clf.fit(X_tr, y_tr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for ax, clf, title in [(ax1, fuzzy_clf, "FuzzyTreeClassifier"),
                        (ax2, hard_clf, "DecisionTreeClassifier")]:
    Z = clf.predict_proba(mesh)[:, 1].reshape(xx.shape)
    im = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 51), cmap="RdBu", alpha=0.9)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=cm_pts,
              edgecolors="k", s=20, linewidths=0.5)
    score = clf.score(X_te, y_te)
    ax.set_title(f"{title}  (acc={score:.1%})", fontsize=12)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

fig.colorbar(im, ax=[ax1, ax2], label="P(class = 1)", shrink=0.8)
fig.suptitle("Probability heatmap — Moons dataset", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "classifier_proba_heatmap.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  classifier_proba_heatmap.png")

# 12. Multiclass
X_blobs, y_blobs = make_blobs(n_samples=400, centers=3, n_features=2,
                              cluster_std=1.5, random_state=42)
scaler = StandardScaler()
X_b = scaler.fit_transform(X_blobs)
X_tr, X_te, y_tr, y_te = train_test_split(X_b, y_blobs, test_size=0.3, random_state=42)

x_min, x_max = X_b[:, 0].min() - 0.5, X_b[:, 0].max() + 0.5
y_min, y_max = X_b[:, 1].min() - 0.5, X_b[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
mesh = np.c_[xx.ravel(), yy.ravel()]
cm3 = ListedColormap(["#FF4444", "#44FF44", "#4444FF"])

multi_clfs = [
    ("FuzzyTreeClassifier",    FuzzyTreeClassifier(max_depth=6)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(max_depth=6)),
    ("RandomForest",           RandomForestClassifier(max_depth=6, n_estimators=50, random_state=0)),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, clf) in zip(axes, multi_clfs):
    clf.fit(X_tr, y_tr)
    Z = clf.predict(mesh).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm3, alpha=0.3)
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=cm3,
              edgecolors="k", s=20, linewidths=0.5, alpha=0.6)
    ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap=cm3,
              edgecolors="k", s=30, linewidths=0.8, marker="D")
    score = clf.score(X_te, y_te)
    ax.set_title(f"{name}\nacc={score:.1%}", fontsize=11)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Multiclass Decision Regions (3 classes)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "classifier_multiclass.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("  classifier_multiclass.png")

print("\nDone. All images saved to docs/img/")
