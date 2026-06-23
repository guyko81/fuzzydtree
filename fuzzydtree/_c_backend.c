#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_MSC_VER)
#define FDT_RESTRICT __restrict
#else
#define FDT_RESTRICT __restrict__
#endif

static inline double
membership_value(double z)
{
    if (z <= -1.0) {
        return 1.0;
    }
    if (z <= 0.0) {
        double t = z + 1.0;
        return 1.0 - 0.5 * t * t;
    }
    if (z < 1.0) {
        double t = z - 1.0;
        return 0.5 * t * t;
    }
    return 0.0;
}

static inline int
centers_are_sorted(const double *centers, int n)
{
    for (int i = 1; i < n; ++i) {
        if (centers[i] < centers[i - 1]) return 0;
    }
    return 1;
}

static inline int
lower_bound_double(const double *values, int n, double target)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (values[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline int
upper_bound_double(const double *values, int n, double target)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (values[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline void
build_prefix_sums(
    const double *FDT_RESTRICT bin_w,
    const double *FDT_RESTRICT bin_wy,
    const double *FDT_RESTRICT bin_wy2,
    int B,
    double *FDT_RESTRICT prefix_w,
    double *FDT_RESTRICT prefix_wy,
    double *FDT_RESTRICT prefix_wy2)
{
    prefix_w[0] = 0.0;
    prefix_wy[0] = 0.0;
    prefix_wy2[0] = 0.0;
    for (int b = 0; b < B; ++b) {
        prefix_w[b + 1] = prefix_w[b] + bin_w[b];
        prefix_wy[b + 1] = prefix_wy[b] + bin_wy[b];
        prefix_wy2[b + 1] = prefix_wy2[b] + bin_wy2[b];
    }
}

typedef struct {
    int feature;
    double threshold;
    double margin;
    int left;
    int right;
    double value;
    double n_samples;
    double impurity_reduction;
    int nan_go_left;
} CNode;

typedef struct {
    double red;
    int feature;
    int threshold_idx;
    double threshold;
    int nan_go_left;
} LAThresholdCandidate;

typedef struct {
    double imp;
    int feature;
    int threshold_idx;
    int margin_idx;
    double threshold;
    double margin;
    int nan_go_left;
} SplitResult;

/* Pre-allocated per-depth workspace — eliminates per-node malloc. */
typedef struct {
    npy_intp *left_idx;
    npy_intp *right_idx;
    double   *left_w;
    double   *right_w;
} DepthWork;

typedef struct {
    npy_intp *left_idx;
    npy_intp *right_idx;
    double   *left_w;
    double   *right_w;
    double   *val_left_w;
    double   *val_right_w;
} LookaheadDepthWork;

typedef struct {
    double *left_w;
    double *right_w;
} LookaheadRolloutWork;

typedef struct {
    const double *X;
    const int *X_bins;
    const double *y;
    const double *Xv;
    const double *yv;
    npy_intp n_features;
    npy_intp n_val;
    const double *thresholds_flat;
    const double *centers_flat;
    const int *n_thresholds;
    int max_thresholds;
    int margin_grid_size;
    int include_hard_splits;
    double margin_min_scale;
    double margin_max_scale;
    double margin_depth_decay;
    double min_samples_leaf;
    double min_train_weight_fraction;
    int lookahead_horizon;
    int lookahead_candidates;
    double lookahead_min_val;
    int margin_cv_folds;
    int margin_cv_repeats;
    uint64_t cv_rng_state;
    int n_classes;
    int split_criterion;

    double *prefix_buf;
    double *margin_buf;
    double *hist_buf;
    double *nan_buf;
    double *xstat_buf;
    double *class_hist_buf;
    double *nan_class_buf;
    double *class_left_buf;
    double *class_total_buf;
    int *int_scratch;
    LAThresholdCandidate *shortlist;
    double *candidate_left_w;
    double *candidate_right_w;
    LookaheadRolloutWork *rollout_work;
    CNode *temp_nodes;
    double *temp_leaf_probs;
    double *temp_pred_probs;
    npy_int64 *temp_stack_nodes;
    double *temp_stack_weights;
    int max_rollout_nodes;

    npy_intp *cv_perm;
    int *cv_fold_ids;
    double *cv_mu_buf;
    double *cv_losses;
    unsigned char *cv_valid;
    double *cv_class_left;
    double *cv_class_right;
} LookaheadContext;

/* ---------- helpers -------------------------------------------------- */

/*
 * One-pass histogram accumulation for several selected features at once.
 *
 * Scans the active row list a SINGLE time, updating every candidate feature's
 * (bin_w, bin_wy, bin_wy2) histogram plus NaN stats.  This replaces N_feature
 * separate active-row scans with one, cutting y[] gathers and exploiting the
 * row-major contiguity of X_bins[row*n_features + f] across the feature axis.
 *
 * Per-feature summation order is identical to accumulate_feature_bins (rows are
 * visited in idx order), so results are bit-for-bit the same.
 *
 * Layout (caller-allocated, zeroed here):
 *   hist[c]   -> bin_w/bin_wy/bin_wy2 at hist + (c*3+s)*B_stride
 *   nan[c]    -> nan_w/nan_wy/nan_wy2 at nan_buf + c*3
 *   has_nan[c], has_two[c]            -> flags per candidate
 */
static void
accumulate_selected_features_bins(
    const int *FDT_RESTRICT X_bins,
    const double *FDT_RESTRICT y,
    npy_intp n_features,
    const npy_intp *FDT_RESTRICT idx,
    const double *FDT_RESTRICT weights,
    npy_intp n,
    const int *FDT_RESTRICT cand, int n_cand,
    const int *FDT_RESTRICT cand_B,
    int B_stride,
    double *FDT_RESTRICT hist,
    double *FDT_RESTRICT nan_buf,
    int *FDT_RESTRICT first_buf,
    int *FDT_RESTRICT has_two,
    int *FDT_RESTRICT has_nan)
{
    memset(hist, 0,
           (size_t)n_cand * 3 * (size_t)B_stride * sizeof(double));
    memset(nan_buf, 0, (size_t)n_cand * 3 * sizeof(double));
    for (int c = 0; c < n_cand; ++c) {
        first_buf[c] = -2;
        has_two[c]   = 0;
        has_nan[c]   = 0;
    }

    for (npy_intp i = 0; i < n; ++i) {
        npy_intp row = idx[i];
        double wi  = weights[i];
        double yi  = y[row];
        double wyi = wi * yi;
        double wy2i = wyi * yi;
        const int *row_bins = X_bins + row * n_features;
        for (int c = 0; c < n_cand; ++c) {
            int b = row_bins[cand[c]];
            if (b < 0) {
                double *nan_c = nan_buf + c * 3;
                nan_c[0] += wi;
                nan_c[1] += wyi;
                nan_c[2] += wy2i;
                has_nan[c] = 1;
            } else if (b < cand_B[c]) {
                double *h = hist + (size_t)c * 3 * B_stride;
                h[b]                += wi;
                h[B_stride + b]     += wyi;
                h[2 * B_stride + b] += wy2i;
                if (!has_two[c]) {
                    if (first_buf[c] < 0) first_buf[c] = b;
                    else if (b != first_buf[c]) has_two[c] = 1;
                }
            }
        }
    }
}

static void
accumulate_selected_features_bins_la(
    const double *FDT_RESTRICT X,
    const int *FDT_RESTRICT X_bins,
    const double *FDT_RESTRICT y,
    npy_intp n_features,
    const npy_intp *FDT_RESTRICT idx,
    const double *FDT_RESTRICT weights,
    npy_intp n,
    const int *FDT_RESTRICT cand, int n_cand,
    const int *FDT_RESTRICT cand_B,
    int B_stride,
    double *FDT_RESTRICT hist,
    double *FDT_RESTRICT nan_buf,
    double *FDT_RESTRICT xstat_buf,
    int *FDT_RESTRICT first_buf,
    int *FDT_RESTRICT has_two,
    int *FDT_RESTRICT has_nan)
{
    memset(hist, 0,
           (size_t)n_cand * 3 * (size_t)B_stride * sizeof(double));
    memset(nan_buf, 0, (size_t)n_cand * 3 * sizeof(double));
    memset(xstat_buf, 0, (size_t)n_cand * 3 * sizeof(double));
    for (int c = 0; c < n_cand; ++c) {
        first_buf[c] = -2;
        has_two[c]   = 0;
        has_nan[c]   = 0;
    }

    for (npy_intp i = 0; i < n; ++i) {
        npy_intp row = idx[i];
        double wi  = weights[i];
        double yi  = y[row];
        double wyi = wi * yi;
        double wy2i = wyi * yi;
        const int *row_bins = X_bins + row * n_features;
        const double *row_x = X + row * n_features;
        for (int c = 0; c < n_cand; ++c) {
            int feature = cand[c];
            int b = row_bins[feature];
            if (b < 0) {
                double *nan_c = nan_buf + c * 3;
                nan_c[0] += wi;
                nan_c[1] += wyi;
                nan_c[2] += wy2i;
                has_nan[c] = 1;
            } else if (b < cand_B[c]) {
                double *h = hist + (size_t)c * 3 * B_stride;
                h[b]                += wi;
                h[B_stride + b]     += wyi;
                h[2 * B_stride + b] += wy2i;

                double x = row_x[feature];
                double *xs = xstat_buf + c * 3;
                xs[0] += wi;
                xs[1] += wi * x;
                xs[2] += wi * x * x;

                if (wi > 1e-12 && !has_two[c]) {
                    if (first_buf[c] < 0) first_buf[c] = b;
                    else if (b != first_buf[c]) has_two[c] = 1;
                }
            }
        }
    }
}

static double
std_from_weighted_sums(double sw, double sx, double sx2)
{
    if (sw < 1e-12) return 0.0;
    double mean = sx / sw;
    double var = sx2 / sw - mean * mean;
    return var > 0.0 ? sqrt(var) : 0.0;
}

static int
append_unique_margin(double *margins, int n, double value)
{
    for (int i = 0; i < n; ++i) {
        if (margins[i] == value) return n;
    }
    margins[n] = value;
    return n + 1;
}

static int
build_margin_grid_la(double feat_std, int depth, int include_hard_splits,
                     double margin_min_scale, double margin_max_scale,
                     int margin_grid_size, double margin_depth_decay,
                     double *margins)
{
    int n = 0;
    if (include_hard_splits) n = append_unique_margin(margins, n, 0.0);

    if (feat_std < 1e-12) {
        n = append_unique_margin(margins, n, 1e-12);
        return n;
    }

    double lo = feat_std * margin_min_scale;
    if (lo < 1e-12) lo = 1e-12;
    double hi = feat_std * margin_max_scale;
    hi *= pow(margin_depth_decay, depth);
    if (hi < lo) hi = lo;

    if (margin_grid_size <= 1) {
        n = append_unique_margin(margins, n, lo);
    } else {
        double ratio = hi / lo;
        for (int mi = 0; mi < margin_grid_size; ++mi) {
            double value = lo * pow(
                ratio, (double)mi / (double)(margin_grid_size - 1));
            n = append_unique_margin(margins, n, value);
        }
    }
    return n;
}

static double
weighted_mean_indexed(
    const double *FDT_RESTRICT y,
    const npy_intp *FDT_RESTRICT idx,
    const double *FDT_RESTRICT weights,
    npy_intp n,
    double *eff_out,
    double *max_weight_out)
{
    double sw = 0.0, swy = 0.0, mw = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        double wi = weights[i];
        sw += wi;
        swy += wi * y[idx[i]];
        if (wi > mw) mw = wi;
    }
    if (eff_out) *eff_out = sw;
    if (max_weight_out) *max_weight_out = mw;
    return sw > 1e-15 ? swy / sw : 0.0;
}

static double
weighted_std_feature_raw(
    const double *FDT_RESTRICT X,
    npy_intp n_features,
    const npy_intp *FDT_RESTRICT idx,
    const double *FDT_RESTRICT weights,
    npy_intp n,
    int feature)
{
    double sw = 0.0, sx = 0.0, sx2 = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        double x = X[idx[i] * n_features + feature];
        if (isnan(x)) continue;
        double wi = weights[i];
        sw += wi;
        sx += wi * x;
        sx2 += wi * x * x;
    }
    return std_from_weighted_sums(sw, sx, sx2);
}

static double
mse_reduction_from_bins(
    const double *FDT_RESTRICT bin_w,
    const double *FDT_RESTRICT bin_wy,
    const double *FDT_RESTRICT bin_wy2,
    double nan_w, double nan_wy, double nan_wy2,
    double nan_mu,
    double w_total, double wy_total, double wy2_total,
    double parent_loss,
    int B,
    const double *centers,
    double threshold,
    double margin,
    double min_samples_leaf)
{
    double inv_margin = margin > 1e-12 ? 1.0 / margin : 0.0;
    double sl = nan_mu * nan_w;
    double wy_l = nan_mu * nan_wy;
    double wy2_l = nan_mu * nan_wy2;

    for (int b = 0; b < B; ++b) {
        double mu;
        if (inv_margin == 0.0) {
            mu = centers[b] <= threshold ? 1.0 : 0.0;
        } else {
            mu = membership_value((centers[b] - threshold) * inv_margin);
        }
        sl += mu * bin_w[b];
        wy_l += mu * bin_wy[b];
        wy2_l += mu * bin_wy2[b];
    }

    double sr = w_total - sl;
    if (sl < min_samples_leaf || sr < min_samples_leaf) return -INFINITY;

    double mean_l = wy_l / sl;
    double mean_r = (wy_total - wy_l) / sr;
    double var_l = wy2_l / sl - mean_l * mean_l;
    double var_r = (wy2_total - wy2_l) / sr - mean_r * mean_r;
    if (var_l < 0.0) var_l = 0.0;
    if (var_r < 0.0) var_r = 0.0;
    double child_loss = (sl * var_l + sr * var_r) / w_total;
    return parent_loss - child_loss;
}

static void
insert_la_candidate(LAThresholdCandidate *shortlist, int *count, int max_count,
                    double red, int feature, int threshold_idx,
                    double threshold, int nan_go_left)
{
    if (max_count <= 0) return;
    int n = *count;
    if (n >= max_count && red <= shortlist[n - 1].red) return;
    int pos = n < max_count ? n : max_count - 1;
    if (n < max_count) *count = n + 1;
    while (pos > 0 && red > shortlist[pos - 1].red) {
        shortlist[pos] = shortlist[pos - 1];
        --pos;
    }
    shortlist[pos].red = red;
    shortlist[pos].feature = feature;
    shortlist[pos].threshold_idx = threshold_idx;
    shortlist[pos].threshold = threshold;
    shortlist[pos].nan_go_left = nan_go_left;
}

static int
lookahead_scan_regression(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    int depth,
    int want_shortlist,
    SplitResult *best)
{
    double eff = 0.0;
    for (npy_intp i = 0; i < n; ++i) eff += weights[i];
    if (eff < 2.0 * ctx->min_samples_leaf) return 0;

    int n_features = (int)ctx->n_features;
    int B_stride = ctx->max_thresholds + 1;
    int *cand      = ctx->int_scratch;
    int *cand_B    = ctx->int_scratch + n_features;
    int *first_buf = ctx->int_scratch + 2 * n_features;
    int *has_two_b = ctx->int_scratch + 3 * n_features;
    int *has_nan_b = ctx->int_scratch + 4 * n_features;

    int n_cand = 0;
    for (int feature = 0; feature < n_features; ++feature) {
        int T = ctx->n_thresholds[feature];
        if (T <= 0) continue;
        cand[n_cand] = feature;
        cand_B[n_cand] = T + 1;
        ++n_cand;
    }
    if (n_cand == 0) return 0;

    accumulate_selected_features_bins_la(
        ctx->X, ctx->X_bins, ctx->y, ctx->n_features,
        idx, weights, n, cand, n_cand, cand_B, B_stride,
        ctx->hist_buf, ctx->nan_buf, ctx->xstat_buf,
        first_buf, has_two_b, has_nan_b);

    if (want_shortlist) {
        for (int i = 0; i < ctx->lookahead_candidates; ++i)
            ctx->shortlist[i].red = -INFINITY;
    } else {
        best->imp = 1e-12;
        best->feature = -1;
        best->threshold_idx = -1;
        best->margin_idx = -1;
    }

    int shortlist_count = 0;
    for (int c = 0; c < n_cand; ++c) {
        if (!has_two_b[c]) continue;

        int feature = cand[c];
        int B = cand_B[c];
        int T = B - 1;
        double *bin_w   = ctx->hist_buf + (size_t)c * 3 * B_stride;
        double *bin_wy  = bin_w + B_stride;
        double *bin_wy2 = bin_w + 2 * B_stride;
        double nan_w  = ctx->nan_buf[c * 3];
        double nan_wy = ctx->nan_buf[c * 3 + 1];
        double nan_wy2 = ctx->nan_buf[c * 3 + 2];
        double *xs = ctx->xstat_buf + c * 3;
        double feat_std = std_from_weighted_sums(xs[0], xs[1], xs[2]);
        int n_margins = build_margin_grid_la(
            feat_std, depth, ctx->include_hard_splits,
            ctx->margin_min_scale, ctx->margin_max_scale,
            ctx->margin_grid_size, ctx->margin_depth_decay,
            ctx->margin_buf);

        double w_total = nan_w, wy_total = nan_wy, wy2_total = nan_wy2;
        for (int b = 0; b < B; ++b) {
            w_total += bin_w[b];
            wy_total += bin_wy[b];
            wy2_total += bin_wy2[b];
        }
        if (w_total < 1e-15) continue;

        double parent_mean = wy_total / w_total;
        double parent_loss = wy2_total / w_total - parent_mean * parent_mean;
        if (parent_loss < 0.0) parent_loss = 0.0;

        int nan_left = has_nan_b[c] ? (nan_w <= 0.5 * w_total) : 1;
        double nan_mu = nan_left ? 1.0 : 0.0;
        const double *thresholds =
            ctx->thresholds_flat + feature * ctx->max_thresholds;
        const double *centers =
            ctx->centers_flat + feature * (ctx->max_thresholds + 1);

        for (int ti = 0; ti < T; ++ti) {
            double threshold = thresholds[ti];
            double t_best = -INFINITY;
            int t_best_margin = -1;
            double t_best_margin_value = 0.0;
            for (int mi = 0; mi < n_margins; ++mi) {
                double margin = ctx->margin_buf[mi];
                double red = mse_reduction_from_bins(
                    bin_w, bin_wy, bin_wy2,
                    nan_w, nan_wy, nan_wy2, nan_mu,
                    w_total, wy_total, wy2_total, parent_loss,
                    B, centers, threshold, margin,
                    ctx->min_samples_leaf);
                if (want_shortlist) {
                    if (red > t_best) {
                        t_best = red;
                        t_best_margin = mi;
                        t_best_margin_value = margin;
                    }
                } else if (red > best->imp) {
                    best->imp = red;
                    best->feature = feature;
                    best->threshold_idx = ti;
                    best->margin_idx = mi;
                    best->threshold = threshold;
                    best->margin = margin;
                    best->nan_go_left = nan_left;
                }
            }
            (void)t_best_margin;
            (void)t_best_margin_value;
            if (want_shortlist && t_best > 0.0) {
                insert_la_candidate(
                    ctx->shortlist, &shortlist_count,
                    ctx->lookahead_candidates, t_best, feature, ti,
                    threshold, nan_left);
            }
        }
    }

    return want_shortlist ? shortlist_count : (best->feature >= 0);
}

static void
accumulate_selected_features_class_bins_la(
    const double *FDT_RESTRICT X,
    const int *FDT_RESTRICT X_bins,
    const double *FDT_RESTRICT y,
    npy_intp n_features,
    const npy_intp *FDT_RESTRICT idx,
    const double *FDT_RESTRICT weights,
    npy_intp n,
    const int *FDT_RESTRICT cand, int n_cand,
    const int *FDT_RESTRICT cand_B,
    int B_stride,
    int n_classes,
    double *FDT_RESTRICT class_hist,
    double *FDT_RESTRICT nan_class,
    double *FDT_RESTRICT xstat_buf,
    int *FDT_RESTRICT first_buf,
    int *FDT_RESTRICT has_two,
    int *FDT_RESTRICT has_nan)
{
    memset(class_hist, 0,
           (size_t)n_cand * (size_t)B_stride * (size_t)n_classes
           * sizeof(double));
    memset(nan_class, 0,
           (size_t)n_cand * (size_t)n_classes * sizeof(double));
    memset(xstat_buf, 0, (size_t)n_cand * 3 * sizeof(double));
    for (int c = 0; c < n_cand; ++c) {
        first_buf[c] = -2;
        has_two[c] = 0;
        has_nan[c] = 0;
    }

    for (npy_intp i = 0; i < n; ++i) {
        npy_intp row = idx[i];
        double wi = weights[i];
        int cls = (int)y[row];
        if (cls < 0 || cls >= n_classes) continue;
        const int *row_bins = X_bins + row * n_features;
        const double *row_x = X + row * n_features;
        for (int c = 0; c < n_cand; ++c) {
            int feature = cand[c];
            int b = row_bins[feature];
            if (b < 0) {
                nan_class[c * n_classes + cls] += wi;
                has_nan[c] = 1;
            } else if (b < cand_B[c]) {
                class_hist[((size_t)c * (size_t)B_stride + (size_t)b)
                           * (size_t)n_classes + (size_t)cls] += wi;

                double x = row_x[feature];
                double *xs = xstat_buf + c * 3;
                xs[0] += wi;
                xs[1] += wi * x;
                xs[2] += wi * x * x;

                if (wi > 1e-12 && !has_two[c]) {
                    if (first_buf[c] < 0) first_buf[c] = b;
                    else if (b != first_buf[c]) has_two[c] = 1;
                }
            }
        }
    }
}

static double
class_impurity_from_counts(const double *counts, int n_classes,
                           double total, int split_criterion)
{
    if (total < 1e-15) return 0.0;
    if (split_criterion == 1) {
        double entropy = 0.0;
        for (int c = 0; c < n_classes; ++c) {
            double p = counts[c] / total;
            if (p > 1e-15) entropy -= p * log(p);
        }
        return entropy;
    }

    double sum_sq = 0.0;
    for (int c = 0; c < n_classes; ++c) {
        double p = counts[c] / total;
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

static double
classifier_reduction_from_bins(
    const double *FDT_RESTRICT class_hist,
    const double *FDT_RESTRICT nan_class,
    double nan_mu,
    const double *FDT_RESTRICT total_counts,
    double parent_impurity,
    double w_total,
    int B,
    int n_classes,
    const double *centers,
    double threshold,
    double margin,
    double min_samples_leaf,
    int split_criterion,
    double *FDT_RESTRICT left_counts)
{
    double inv_margin = margin > 1e-12 ? 1.0 / margin : 0.0;
    double sl = 0.0;
    for (int c = 0; c < n_classes; ++c) {
        double v = nan_mu * nan_class[c];
        left_counts[c] = v;
        sl += v;
    }

    for (int b = 0; b < B; ++b) {
        double mu;
        if (inv_margin == 0.0) {
            mu = centers[b] <= threshold ? 1.0 : 0.0;
        } else {
            mu = membership_value((centers[b] - threshold) * inv_margin);
        }
        const double *bin = class_hist + (size_t)b * (size_t)n_classes;
        for (int c = 0; c < n_classes; ++c) {
            double v = mu * bin[c];
            left_counts[c] += v;
            sl += v;
        }
    }

    double sr = w_total - sl;
    if (sl < min_samples_leaf || sr < min_samples_leaf) return -INFINITY;

    double left_imp = class_impurity_from_counts(
        left_counts, n_classes, sl, split_criterion);
    double right_imp;
    {
        double right_total = 0.0;
        for (int c = 0; c < n_classes; ++c) {
            left_counts[c] = total_counts[c] - left_counts[c];
            right_total += left_counts[c];
        }
        right_imp = class_impurity_from_counts(
            left_counts, n_classes, right_total, split_criterion);
    }

    double child_impurity = (sl * left_imp + sr * right_imp) / w_total;
    return parent_impurity - child_impurity;
}

static int
lookahead_scan_classifier(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    int depth,
    int want_shortlist,
    SplitResult *best)
{
    double eff = 0.0;
    for (npy_intp i = 0; i < n; ++i) eff += weights[i];
    if (eff < 2.0 * ctx->min_samples_leaf) return 0;

    int n_features = (int)ctx->n_features;
    int n_classes = ctx->n_classes;
    int B_stride = ctx->max_thresholds + 1;
    int *cand      = ctx->int_scratch;
    int *cand_B    = ctx->int_scratch + n_features;
    int *first_buf = ctx->int_scratch + 2 * n_features;
    int *has_two_b = ctx->int_scratch + 3 * n_features;
    int *has_nan_b = ctx->int_scratch + 4 * n_features;

    int n_cand = 0;
    for (int feature = 0; feature < n_features; ++feature) {
        int T = ctx->n_thresholds[feature];
        if (T <= 0) continue;
        cand[n_cand] = feature;
        cand_B[n_cand] = T + 1;
        ++n_cand;
    }
    if (n_cand == 0) return 0;

    accumulate_selected_features_class_bins_la(
        ctx->X, ctx->X_bins, ctx->y, ctx->n_features,
        idx, weights, n, cand, n_cand, cand_B, B_stride,
        n_classes, ctx->class_hist_buf, ctx->nan_class_buf,
        ctx->xstat_buf, first_buf, has_two_b, has_nan_b);

    if (want_shortlist) {
        for (int i = 0; i < ctx->lookahead_candidates; ++i)
            ctx->shortlist[i].red = -INFINITY;
    } else {
        best->imp = 1e-12;
        best->feature = -1;
        best->threshold_idx = -1;
        best->margin_idx = -1;
    }

    int shortlist_count = 0;
    for (int c = 0; c < n_cand; ++c) {
        if (!has_two_b[c]) continue;

        int feature = cand[c];
        int B = cand_B[c];
        int T = B - 1;
        double *class_hist = ctx->class_hist_buf
            + (size_t)c * (size_t)B_stride * (size_t)n_classes;
        double *nan_class = ctx->nan_class_buf + c * n_classes;
        double *xs = ctx->xstat_buf + c * 3;
        double feat_std = std_from_weighted_sums(xs[0], xs[1], xs[2]);
        int n_margins = build_margin_grid_la(
            feat_std, depth, ctx->include_hard_splits,
            ctx->margin_min_scale, ctx->margin_max_scale,
            ctx->margin_grid_size, ctx->margin_depth_decay,
            ctx->margin_buf);

        double w_total = 0.0;
        double nan_w = 0.0;
        for (int k = 0; k < n_classes; ++k) {
            double total = nan_class[k];
            nan_w += nan_class[k];
            for (int b = 0; b < B; ++b) {
                total += class_hist[((size_t)b * (size_t)n_classes)
                                    + (size_t)k];
            }
            ctx->class_total_buf[k] = total;
            w_total += total;
        }
        if (w_total < 1e-15) continue;

        double parent_impurity = class_impurity_from_counts(
            ctx->class_total_buf, n_classes, w_total,
            ctx->split_criterion);

        int nan_left = has_nan_b[c] ? (nan_w <= 0.5 * w_total) : 1;
        double nan_mu = nan_left ? 1.0 : 0.0;
        const double *thresholds =
            ctx->thresholds_flat + feature * ctx->max_thresholds;
        const double *centers =
            ctx->centers_flat + feature * (ctx->max_thresholds + 1);

        for (int ti = 0; ti < T; ++ti) {
            double threshold = thresholds[ti];
            double t_best = -INFINITY;
            for (int mi = 0; mi < n_margins; ++mi) {
                double margin = ctx->margin_buf[mi];
                double red = classifier_reduction_from_bins(
                    class_hist, nan_class, nan_mu, ctx->class_total_buf,
                    parent_impurity, w_total, B, n_classes, centers,
                    threshold, margin, ctx->min_samples_leaf,
                    ctx->split_criterion, ctx->class_left_buf);
                if (want_shortlist) {
                    if (red > t_best) t_best = red;
                } else if (red > best->imp) {
                    best->imp = red;
                    best->feature = feature;
                    best->threshold_idx = ti;
                    best->margin_idx = mi;
                    best->threshold = threshold;
                    best->margin = margin;
                    best->nan_go_left = nan_left;
                }
            }
            if (want_shortlist && t_best > 0.0) {
                insert_la_candidate(
                    ctx->shortlist, &shortlist_count,
                    ctx->lookahead_candidates, t_best, feature, ti,
                    threshold, nan_left);
            }
        }
    }

    return want_shortlist ? shortlist_count : (best->feature >= 0);
}

/* Approximate feature std from pre-accumulated bin stats (O(B)). */
static double
std_from_bins(const double *bin_w, const double *centers, int B)
{
    double sw = 0.0, sx = 0.0, sx2 = 0.0;
    for (int b = 0; b < B; ++b) {
        double bw = bin_w[b];
        if (bw > 0.0) {
            double c = centers[b];
            sw  += bw;
            sx  += bw * c;
            sx2 += bw * c * c;
        }
    }
    if (sw < 1e-12) return 0.0;
    double mean = sx / sw;
    double var  = sx2 / sw - mean * mean;
    return var > 0.0 ? sqrt(var) : 0.0;
}

/*
 * Evaluate all (threshold, margin) split candidates for one feature.
 * Takes pre-accumulated bin stats — no O(N) work inside.
 * Prefix sums let fully-left/right bins be handled in O(1); only the fuzzy
 * transition window needs membership_value calls.
 */
static void
eval_feature_splits(
    const double *FDT_RESTRICT bin_w,
    const double *FDT_RESTRICT bin_wy,
    const double *FDT_RESTRICT bin_wy2,
    const double *FDT_RESTRICT prefix_w,
    const double *FDT_RESTRICT prefix_wy,
    const double *FDT_RESTRICT prefix_wy2,
    double nan_w, double nan_wy, double nan_wy2,
    double nan_mu,
    double w_total, double wy_total, double wy2_total,
    int B,
    const double *thresholds, const double *centers, int n_thresholds,
    const double *margins, int n_margins,
    double min_samples_leaf, int optimize_split_gain, double split_gain_l2,
    int feature, int accept_first_candidate, SplitResult *best)
{
    if (w_total < 1e-15) return;

    double parent_mean = wy_total / w_total;
    double parent_loss = wy2_total / w_total - parent_mean * parent_mean;
    int sorted_centers = centers_are_sorted(centers, B);

    for (int ti = 0; ti < n_thresholds; ++ti) {
        double threshold = thresholds[ti];
        for (int mi = 0; mi < n_margins; ++mi) {
            double margin    = margins[mi];
            double inv_margin = margin > 1e-12 ? 1.0 / margin : 0.0;
            double om_nan = 1.0 - nan_mu;

            double sl, wy_l, wy2_l;
            double aa, ab, bb, rhs_a, rhs_b;

            if (sorted_centers) {
                int lo, hi;
                if (inv_margin == 0.0) {
                    lo = upper_bound_double(centers, B, threshold);
                    hi = lo;
                } else {
                    double left_edge = threshold - margin;
                    double right_edge = threshold + margin;
                    lo = lower_bound_double(centers, B, left_edge);
                    hi = upper_bound_double(centers, B, right_edge);
                    if (hi < lo) hi = lo;
                }

                sl    = nan_mu * nan_w + prefix_w[lo];
                wy_l  = nan_mu * nan_wy + prefix_wy[lo];
                wy2_l = nan_mu * nan_wy2 + prefix_wy2[lo];

                aa    = nan_mu * nan_mu * nan_w + prefix_w[lo];
                ab    = nan_mu * om_nan * nan_w;
                bb    = om_nan * om_nan * nan_w;
                rhs_a = nan_mu * nan_wy + prefix_wy[lo];
                rhs_b = om_nan * nan_wy;

                if (inv_margin != 0.0) {
                    for (int b = lo; b < hi; ++b) {
                        double mu = membership_value(
                            (centers[b] - threshold) * inv_margin);
                        double bw = bin_w[b];
                        double bwy = bin_wy[b];
                        double om = 1.0 - mu;
                        sl    += mu * bw;
                        wy_l  += mu * bwy;
                        wy2_l += mu * bin_wy2[b];
                        aa    += mu * mu * bw;
                        ab    += mu * om * bw;
                        bb    += om * om * bw;
                        rhs_a += mu * bwy;
                        rhs_b += om * bwy;
                    }
                }

                if (optimize_split_gain) {
                    bb    += prefix_w[B] - prefix_w[hi];
                    rhs_b += prefix_wy[B] - prefix_wy[hi];
                }
            } else {
                sl    = nan_mu * nan_w;
                wy_l  = nan_mu * nan_wy;
                wy2_l = nan_mu * nan_wy2;

                aa    = nan_mu * nan_mu * nan_w;
                ab    = nan_mu * om_nan * nan_w;
                bb    = om_nan * om_nan * nan_w;
                rhs_a = nan_mu * nan_wy;
                rhs_b = om_nan * nan_wy;

                for (int b = 0; b < B; ++b) {
                    double mu;
                    if (inv_margin == 0.0) {
                        mu = centers[b] <= threshold ? 1.0 : 0.0;
                    } else {
                        mu = membership_value(
                            (centers[b] - threshold) * inv_margin);
                    }
                    double bw = bin_w[b];
                    double bwy = bin_wy[b];
                    double om = 1.0 - mu;
                    sl    += mu * bw;
                    wy_l  += mu * bwy;
                    wy2_l += mu * bin_wy2[b];
                    aa    += mu * mu * bw;
                    ab    += mu * om * bw;
                    bb    += om * om * bw;
                    rhs_a += mu * bwy;
                    rhs_b += om * bwy;
                }
            }

            double sr = w_total - sl;
            if (sl < min_samples_leaf || sr < min_samples_leaf) continue;

            double child_loss;
            if (optimize_split_gain) {
                double ridge = split_gain_l2 + 1e-12 * w_total;
                aa += ridge;
                bb += ridge;
                double det = aa * bb - ab * ab;
                if (det <= 1e-18) continue;
                double gain_fit = (bb * rhs_a * rhs_a
                                   - 2.0 * ab * rhs_a * rhs_b
                                   + aa * rhs_b * rhs_b) / det;
                child_loss = (wy2_total - gain_fit) / w_total;
                if (child_loss < 0.0) child_loss = 0.0;
            } else {
                double mean_l = wy_l / sl;
                double mean_r = (wy_total - wy_l) / sr;
                double var_l = wy2_l / sl - mean_l * mean_l;
                double var_r = (wy2_total - wy2_l) / sr - mean_r * mean_r;
                if (var_l < 0.0) var_l = 0.0;
                if (var_r < 0.0) var_r = 0.0;
                child_loss = (sl * var_l + sr * var_r) / w_total;
            }

            double imp = parent_loss - child_loss;
            if ((accept_first_candidate && best->threshold_idx < 0)
                    || imp > best->imp) {
                best->imp         = imp;
                best->feature     = feature;
                best->threshold_idx = ti;
                best->margin_idx  = mi;
                best->threshold   = threshold;
                best->margin      = margin;
                best->nan_go_left = nan_mu == 1.0 ? 1 : 0;
            }
        }
    }
}

/*
 * Find the best split across all candidate features.
 *
 * bin_buf:    pre-allocated, size 6 * (max_thresholds + 2) doubles.
 *             Partitioned as three bin-stat arrays and three prefix arrays.
 * margin_buf: pre-allocated, size margin_grid_size doubles.
 */
static int
find_best_split_numeric(
    const double *X, const int *X_bins, const double *y,
    npy_intp n_features,
    const npy_intp *idx, const double *weights, npy_intp n,
    int depth, const double *thresholds_flat,
    const double *centers_flat, const int *n_thresholds,
    int max_thresholds, int margin_grid_size,
    double margin_depth_decay, double min_samples_leaf,
    int optimize_split_gain, double split_gain_l2,
    const int *feature_choices, const int *feature_counts,
    int feature_choice_width, int *feature_choice_cursor,
    double *prefix_ws, double *margin_ws,
    double *hist_buf, double *nan_buf, int *int_scratch,
    SplitResult *cand_best,
    SplitResult *best)
{
    best->imp     = 0.0;
    best->feature = -1;
    best->threshold_idx = -1;
    best->margin_idx = -1;

    int choice_row    = *feature_choice_cursor;
    *feature_choice_cursor += 1;
    int n_candidates = feature_counts[choice_row];
    int buf_stride = max_thresholds + 2;
    int B_stride   = max_thresholds + 1;

    /* int_scratch partitioned into five n_features-sized regions. */
    int *cand      = int_scratch;
    int *cand_B    = int_scratch + (int)n_features;
    int *first_buf = int_scratch + 2 * (int)n_features;
    int *has_two_b = int_scratch + 3 * (int)n_features;
    int *has_nan_b = int_scratch + 4 * (int)n_features;

    /* Build the compact list of valid candidate features for this node. */
    int n_cand = 0;
    for (int choice_i = 0; choice_i < n_candidates; ++choice_i) {
        int feature = feature_choices[choice_row * feature_choice_width + choice_i];
        if (feature < 0 || feature >= n_features) continue;
        int T = n_thresholds[feature];
        if (T <= 0) continue;
        cand[n_cand]   = feature;
        cand_B[n_cand] = T + 1;
        ++n_cand;
    }
    if (n_cand == 0) return 0;

    /* Single active-row pass builds every candidate's histogram. */
    accumulate_selected_features_bins(
        X_bins, y, n_features, idx, weights, n,
        cand, n_cand, cand_B, B_stride,
        hist_buf, nan_buf, first_buf, has_two_b, has_nan_b);

    /* Each candidate writes only its own slot, so the parallel loop has no
       shared writes; the merge afterwards is serial and order-deterministic,
       making the result identical to the single-threaded build. */
    for (int c = 0; c < n_cand; ++c) {
        cand_best[c].imp = 0.0;
        cand_best[c].feature = -1;
        cand_best[c].threshold_idx = -1;
        cand_best[c].margin_idx = -1;
    }

    int c;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(n_cand > 1)
#endif
    for (c = 0; c < n_cand; ++c) {
        if (!has_two_b[c]) continue;

        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        int feature = cand[c];
        int B = cand_B[c];
        int T = B - 1;
        double *bin_w   = hist_buf + (size_t)c * 3 * B_stride;
        double *bin_wy  = bin_w + B_stride;
        double *bin_wy2 = bin_w + 2 * B_stride;
        double *prefix_w   = prefix_ws + (size_t)tid * 3 * buf_stride;
        double *prefix_wy  = prefix_w + buf_stride;
        double *prefix_wy2 = prefix_w + 2 * buf_stride;
        double *margin_buf = margin_ws + (size_t)tid * margin_grid_size;

        double nan_w  = nan_buf[c * 3];
        double nan_wy = nan_buf[c * 3 + 1];
        double nan_wy2 = nan_buf[c * 3 + 2];
        int has_nan = has_nan_b[c];

        /* Compute std from bin centers (O(B)) — replaces weighted_std_feature. */
        const double *centers = centers_flat + feature * (max_thresholds + 1);
        double feat_std = std_from_bins(bin_w, centers, B);

        /* Build margin grid. */
        if (feat_std < 1e-12) {
            for (int mi = 0; mi < margin_grid_size; ++mi)
                margin_buf[mi] = 1e-12;
        } else {
            double lo    = feat_std * 0.4;
            double hi    = feat_std * 20.0 * pow(margin_depth_decay, depth);
            if (hi < lo) hi = lo;
            if (margin_grid_size <= 1) {
                margin_buf[0] = lo;
            } else {
                double ratio = hi / lo;
                for (int mi = 0; mi < margin_grid_size; ++mi) {
                    margin_buf[mi] = lo * pow(
                        ratio, (double)mi / (double)(margin_grid_size - 1));
                }
            }
        }

        /* Totals from bin stats + NaN stats. */
        double w_total   = nan_w;
        double wy_total  = nan_wy;
        double wy2_total = nan_wy2;
        for (int b = 0; b < B; ++b) {
            w_total   += bin_w[b];
            wy_total  += bin_wy[b];
            wy2_total += bin_wy2[b];
        }
        build_prefix_sums(
            bin_w, bin_wy, bin_wy2, B, prefix_w, prefix_wy, prefix_wy2);

        const double *thresholds = thresholds_flat + feature * max_thresholds;

        /* nan goes left */
        eval_feature_splits(
            bin_w, bin_wy, bin_wy2, prefix_w, prefix_wy, prefix_wy2,
            nan_w, nan_wy, nan_wy2, 1.0,
            w_total, wy_total, wy2_total, B,
            thresholds, centers, T,
            margin_buf, margin_grid_size,
            min_samples_leaf, optimize_split_gain, split_gain_l2,
            feature, 0, &cand_best[c]);

        if (has_nan) {
            /* nan goes right */
            eval_feature_splits(
                bin_w, bin_wy, bin_wy2, prefix_w, prefix_wy, prefix_wy2,
                nan_w, nan_wy, nan_wy2, 0.0,
                w_total, wy_total, wy2_total, B,
                thresholds, centers, T,
                margin_buf, margin_grid_size,
                min_samples_leaf, optimize_split_gain, split_gain_l2,
                feature, 0, &cand_best[c]);
        }
    }

    /* Deterministic serial merge in candidate order: strict '>' keeps the
       lowest-index candidate on ties, matching the serial split search. */
    for (int c = 0; c < n_cand; ++c) {
        if (cand_best[c].threshold_idx >= 0 &&
            (best->threshold_idx < 0 || cand_best[c].imp > best->imp)) {
            *best = cand_best[c];
        }
    }
    return 0;
}

static inline double
mu_left_numeric(double x, double threshold, double margin, int nan_go_left)
{
    if (isnan(x)) return nan_go_left ? 1.0 : 0.0;
    if (margin < 1e-12) return x <= threshold ? 1.0 : 0.0;
    return membership_value((x - threshold) / margin);
}

/*
 * Depth-first fuzzy regression tree builder.
 *
 * Uses pre-allocated DepthWork[depth] to store left/right child index arrays,
 * eliminating per-node malloc.  bin_buf and margin_buf are shared across all
 * nodes (single allocation per tree).
 */
static int
build_depth_first_numeric(
    const double *X, const int *X_bins,
    const double *y, npy_intp n_features,
    const npy_intp *idx, const double *weights,
    npy_intp n,
    double summary_eff, double summary_swy, double summary_mw,
    int has_summary,
    int depth, int max_depth,
    const double *thresholds_flat, const double *centers_flat,
    const int *n_thresholds, int max_thresholds,
    int margin_grid_size, double margin_depth_decay,
    double min_samples_leaf, double min_train_weight_fraction,
    int optimize_split_gain, double split_gain_l2,
    const int *feature_choices, const int *feature_counts,
    int feature_choice_width, int *feature_choice_cursor,
    DepthWork *dw, double *prefix_ws, double *margin_ws,
    double *hist_buf, double *nan_buf, int *int_scratch,
    SplitResult *cand_best,
    CNode *nodes, int *node_count, int max_nodes)
{
    if (*node_count >= max_nodes) return -1;

    int node_id = *node_count;
    *node_count += 1;

    double eff, swy, mw;
    if (has_summary) {
        eff = summary_eff;
        swy = summary_swy;
        mw = summary_mw;
    } else {
        eff = 0.0;
        swy = 0.0;
        mw = 0.0;
        for (npy_intp i = 0; i < n; ++i) {
            double wi = weights[i];
            eff += wi;
            swy += wi * y[idx[i]];
            if (wi > mw) mw = wi;
        }
    }
    double value = eff > 1e-15 ? swy / eff : 0.0;

    nodes[node_id].feature           = -1;
    nodes[node_id].threshold         = 0.0;
    nodes[node_id].margin            = 0.0;
    nodes[node_id].left              = -1;
    nodes[node_id].right             = -1;
    nodes[node_id].value             = value;
    nodes[node_id].n_samples         = eff;
    nodes[node_id].impurity_reduction = 0.0;
    nodes[node_id].nan_go_left       = 1;

    if (depth >= max_depth || n < 2 || eff < 2.0 * min_samples_leaf)
        return node_id;

    SplitResult best;
    if (find_best_split_numeric(
            X, X_bins, y, n_features, idx, weights, n, depth,
            thresholds_flat, centers_flat, n_thresholds, max_thresholds,
            margin_grid_size, margin_depth_decay, min_samples_leaf,
            optimize_split_gain, split_gain_l2,
            feature_choices, feature_counts, feature_choice_width,
            feature_choice_cursor, prefix_ws, margin_ws,
            hist_buf, nan_buf, int_scratch, cand_best, &best) != 0)
        return -1;

    if (best.feature < 0) return node_id;

    /* Use pre-allocated arrays for this depth level — no malloc. */
    npy_intp *left_idx  = dw[depth].left_idx;
    npy_intp *right_idx = dw[depth].right_idx;
    double   *left_w    = dw[depth].left_w;
    double   *right_w   = dw[depth].right_w;

    double eps = mw > 0.0 ? 1e-6 * mw : 1e-10;
    if (min_train_weight_fraction > 0.0) {
        double frac_eps = min_train_weight_fraction * mw;
        if (frac_eps > eps) eps = frac_eps;
    }

    npy_intp nl = 0, nr = 0;
    double left_eff = 0.0, left_swy = 0.0, left_mw = 0.0;
    double right_eff = 0.0, right_swy = 0.0, right_mw = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        npy_intp row = idx[i];
        double x     = X[row * n_features + best.feature];
        double yi    = y[row];
        double mu    = mu_left_numeric(x, best.threshold, best.margin,
                                       best.nan_go_left);
        double wl_i  = weights[i] * mu;
        double wr_i  = weights[i] * (1.0 - mu);
        if (wl_i > eps) {
            left_idx[nl] = row;
            left_w[nl] = wl_i;
            left_eff += wl_i;
            left_swy += wl_i * yi;
            if (wl_i > left_mw) left_mw = wl_i;
            ++nl;
        }
        if (wr_i > eps) {
            right_idx[nr] = row;
            right_w[nr] = wr_i;
            right_eff += wr_i;
            right_swy += wr_i * yi;
            if (wr_i > right_mw) right_mw = wr_i;
            ++nr;
        }
    }

    if (nl < 1 || nr < 1) return node_id;

    nodes[node_id].feature           = best.feature;
    nodes[node_id].threshold         = best.threshold;
    nodes[node_id].margin            = best.margin;
    nodes[node_id].impurity_reduction = best.imp * eff;
    nodes[node_id].nan_go_left       = best.nan_go_left;

    int left_id = build_depth_first_numeric(
        X, X_bins, y, n_features, left_idx, left_w, nl,
        left_eff, left_swy, left_mw, 1, depth + 1,
        max_depth, thresholds_flat, centers_flat, n_thresholds,
        max_thresholds, margin_grid_size, margin_depth_decay,
        min_samples_leaf, min_train_weight_fraction, optimize_split_gain,
        split_gain_l2, feature_choices, feature_counts,
        feature_choice_width, feature_choice_cursor,
        dw, prefix_ws, margin_ws, hist_buf, nan_buf, int_scratch,
        cand_best, nodes, node_count, max_nodes);

    int right_id = build_depth_first_numeric(
        X, X_bins, y, n_features, right_idx, right_w, nr,
        right_eff, right_swy, right_mw, 1, depth + 1,
        max_depth, thresholds_flat, centers_flat, n_thresholds,
        max_thresholds, margin_grid_size, margin_depth_decay,
        min_samples_leaf, min_train_weight_fraction, optimize_split_gain,
        split_gain_l2, feature_choices, feature_counts,
        feature_choice_width, feature_choice_cursor,
        dw, prefix_ws, margin_ws, hist_buf, nan_buf, int_scratch,
        cand_best, nodes, node_count, max_nodes);

    if (left_id < 0 || right_id < 0) return -1;

    nodes[node_id].left  = left_id;
    nodes[node_id].right = right_id;
    return node_id;
}

static void
split_weights_raw(
    const double *FDT_RESTRICT X,
    npy_intp n_features,
    const npy_intp *FDT_RESTRICT idx,
    const double *FDT_RESTRICT weights,
    npy_intp n,
    int feature,
    double threshold,
    double margin,
    int nan_go_left,
    double *FDT_RESTRICT left_w,
    double *FDT_RESTRICT right_w,
    double *left_sum,
    double *right_sum)
{
    double sl = 0.0, sr = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        double x = X[idx[i] * n_features + feature];
        double mu = mu_left_numeric(x, threshold, margin, nan_go_left);
        double wl = weights[i] * mu;
        double wr = weights[i] * (1.0 - mu);
        left_w[i] = wl;
        right_w[i] = wr;
        sl += wl;
        sr += wr;
    }
    if (left_sum) *left_sum = sl;
    if (right_sum) *right_sum = sr;
}

static double
split_child_loss_raw(
    const double *FDT_RESTRICT y,
    const npy_intp *FDT_RESTRICT idx,
    const double *FDT_RESTRICT left_w,
    const double *FDT_RESTRICT right_w,
    npy_intp n,
    double parent_eff)
{
    double sl = 0.0, swyl = 0.0, swy2l = 0.0;
    double sr = 0.0, swyr = 0.0, swy2r = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        double yi = y[idx[i]];
        double wl = left_w[i];
        double wr = right_w[i];
        sl += wl;
        swyl += wl * yi;
        swy2l += wl * yi * yi;
        sr += wr;
        swyr += wr * yi;
        swy2r += wr * yi * yi;
    }
    if (sl < 1e-15 || sr < 1e-15 || parent_eff < 1e-15) return 0.0;
    double ml = swyl / sl;
    double mr = swyr / sr;
    double vl = swy2l / sl - ml * ml;
    double vr = swy2r / sr - mr * mr;
    if (vl < 0.0) vl = 0.0;
    if (vr < 0.0) vr = 0.0;
    return (sl * vl + sr * vr) / parent_eff;
}

static uint64_t
cv_next_u64(LookaheadContext *ctx)
{
    uint64_t x = ctx->cv_rng_state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    ctx->cv_rng_state = x;
    return x * 2685821657736338717ULL;
}

static void
cv_make_fold_ids(LookaheadContext *ctx, npy_intp n, int folds)
{
    for (npy_intp i = 0; i < n; ++i) ctx->cv_perm[i] = i;
    for (npy_intp i = n - 1; i > 0; --i) {
        npy_intp j = (npy_intp)(
            cv_next_u64(ctx) % (uint64_t)(i + 1));
        npy_intp tmp = ctx->cv_perm[i];
        ctx->cv_perm[i] = ctx->cv_perm[j];
        ctx->cv_perm[j] = tmp;
    }
    for (npy_intp i = 0; i < n; ++i) {
        ctx->cv_fold_ids[i] = (int)(ctx->cv_perm[i] % folds);
    }
}

static int
build_rollout_regression(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    int depth,
    int *node_count);

static int
build_rollout_classifier(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    int depth,
    int *node_count);

static double
temp_tree_validation_loss_indexed(
    LookaheadContext *ctx,
    int root_id,
    const npy_intp *idx,
    const double *weights,
    const int *fold_ids,
    npy_intp n,
    int fold)
{
    double loss = 0.0;
    double denom = 0.0;
    double unweighted_loss = 0.0;
    npy_intp n_val = 0;

    for (npy_intp i = 0; i < n; ++i) {
        if (fold_ids[i] != fold) continue;
        ++n_val;

        double pred = 0.0;
        npy_intp top = 0;
        ctx->temp_stack_nodes[top] = root_id;
        ctx->temp_stack_weights[top] = 1.0;
        ++top;

        while (top > 0) {
            --top;
            npy_int64 node_idx = ctx->temp_stack_nodes[top];
            double path_w = ctx->temp_stack_weights[top];
            CNode *node = &ctx->temp_nodes[node_idx];
            if (node->left < 0) {
                pred += path_w * node->value;
                continue;
            }

            double x = ctx->X[idx[i] * ctx->n_features + node->feature];
            double mu = mu_left_numeric(
                x, node->threshold, node->margin, node->nan_go_left);
            if (mu > 0.0) {
                ctx->temp_stack_nodes[top] = node->left;
                ctx->temp_stack_weights[top] = path_w * mu;
                ++top;
            }
            if (mu < 1.0) {
                ctx->temp_stack_nodes[top] = node->right;
                ctx->temp_stack_weights[top] = path_w * (1.0 - mu);
                ++top;
            }
        }

        double r = ctx->y[idx[i]] - pred;
        double wi = weights[i];
        loss += wi * r * r;
        denom += wi;
        unweighted_loss += r * r;
    }

    if (denom > 1e-12) return loss / denom;
    return n_val > 0 ? unweighted_loss / (double)n_val : 0.0;
}

static double
temp_tree_validation_logloss_indexed(
    LookaheadContext *ctx,
    int root_id,
    const npy_intp *idx,
    const double *weights,
    const int *fold_ids,
    npy_intp n,
    int fold)
{
    int K = ctx->n_classes;
    double loss = 0.0;
    double denom = 0.0;
    double unweighted_loss = 0.0;
    npy_intp n_val = 0;

    for (npy_intp i = 0; i < n; ++i) {
        if (fold_ids[i] != fold) continue;
        ++n_val;
        for (int c = 0; c < K; ++c) ctx->temp_pred_probs[c] = 0.0;

        npy_intp top = 0;
        ctx->temp_stack_nodes[top] = root_id;
        ctx->temp_stack_weights[top] = 1.0;
        ++top;

        while (top > 0) {
            --top;
            npy_int64 node_idx = ctx->temp_stack_nodes[top];
            double path_w = ctx->temp_stack_weights[top];
            CNode *node = &ctx->temp_nodes[node_idx];
            if (node->left < 0) {
                double *p = ctx->temp_leaf_probs
                    + (size_t)node_idx * (size_t)K;
                for (int c = 0; c < K; ++c)
                    ctx->temp_pred_probs[c] += path_w * p[c];
                continue;
            }

            double x = ctx->X[idx[i] * ctx->n_features + node->feature];
            double mu = mu_left_numeric(
                x, node->threshold, node->margin, node->nan_go_left);
            if (mu > 0.0) {
                ctx->temp_stack_nodes[top] = node->left;
                ctx->temp_stack_weights[top] = path_w * mu;
                ++top;
            }
            if (mu < 1.0) {
                ctx->temp_stack_nodes[top] = node->right;
                ctx->temp_stack_weights[top] = path_w * (1.0 - mu);
                ++top;
            }
        }

        int cls = (int)ctx->y[idx[i]];
        double p = (cls >= 0 && cls < K) ? ctx->temp_pred_probs[cls] : 0.0;
        if (p < 1e-12) p = 1e-12;
        if (p > 1.0) p = 1.0;
        double li = -log(p);
        double wi = weights[i];
        loss += wi * li;
        denom += wi;
        unweighted_loss += li;
    }

    if (denom > 1e-12) return loss / denom;
    return n_val > 0 ? unweighted_loss / (double)n_val : 0.0;
}

static int
cv_select_margin_regression(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    int feature,
    double threshold,
    int nan_go_left,
    int depth,
    double *selected_margin)
{
    int folds = ctx->margin_cv_folds;
    int repeats = ctx->margin_cv_repeats > 1 ? ctx->margin_cv_repeats : 1;
    if (folds < 2 || n < (npy_intp)2 * (npy_intp)folds) return 0;
    if (!ctx->cv_perm || !ctx->cv_fold_ids || !ctx->cv_mu_buf ||
            !ctx->cv_losses || !ctx->cv_valid) {
        return -1;
    }

    int max_margins = ctx->margin_grid_size + (
        ctx->include_hard_splits ? 1 : 0) + 1;
    if (max_margins < 1) max_margins = 1;
    double *margins = (double *)malloc((size_t)max_margins * sizeof(double));
    if (!margins) return -1;

    double feat_std = weighted_std_feature_raw(
        ctx->X, ctx->n_features, idx, weights, n, feature);
    int n_margins = build_margin_grid_la(
        feat_std, depth, ctx->include_hard_splits,
        ctx->margin_min_scale, ctx->margin_max_scale,
        ctx->margin_grid_size, ctx->margin_depth_decay,
        margins);
    if (n_margins <= 1) {
        free(margins);
        return 0;
    }

    for (int mi = 0; mi < n_margins; ++mi) {
        ctx->cv_losses[mi] = 0.0;
        ctx->cv_valid[mi] = 1;
    }

    double *train_w = ctx->cv_mu_buf;
    for (int rep = 0; rep < repeats; ++rep) {
        cv_make_fold_ids(ctx, n, folds);
        for (int fold = 0; fold < folds; ++fold) {
            npy_intp n_val = 0, n_train = 0;
            for (npy_intp i = 0; i < n; ++i) {
                if (ctx->cv_fold_ids[i] == fold) ++n_val;
                else ++n_train;
            }
            if (n_val == 0 || n_train == 0) continue;

            for (int mi = 0; mi < n_margins; ++mi) {
                if (!ctx->cv_valid[mi]) continue;

                for (npy_intp i = 0; i < n; ++i) {
                    train_w[i] = ctx->cv_fold_ids[i] == fold
                        ? 0.0 : weights[i];
                }

                double sl = 0.0, sr = 0.0;
                double margin = margins[mi];
                split_weights_raw(
                    ctx->X, ctx->n_features, idx, train_w, n,
                    feature, threshold, margin, nan_go_left,
                    ctx->candidate_left_w, ctx->candidate_right_w,
                    &sl, &sr);
                if (sl < ctx->min_samples_leaf ||
                        sr < ctx->min_samples_leaf) {
                    ctx->cv_valid[mi] = 0;
                    continue;
                }

                int node_count = 1;
                CNode *root = &ctx->temp_nodes[0];
                root->feature = feature;
                root->threshold = threshold;
                root->margin = margin;
                root->left = -1;
                root->right = -1;
                root->value = 0.0;
                root->n_samples = sl + sr;
                root->impurity_reduction = 0.0;
                root->nan_go_left = nan_go_left;

                int left_id = build_rollout_regression(
                    ctx, idx, ctx->candidate_left_w, n, 1, &node_count);
                if (left_id < 0) {
                    free(margins);
                    return -1;
                }
                int right_id = build_rollout_regression(
                    ctx, idx, ctx->candidate_right_w, n, 1, &node_count);
                if (right_id < 0) {
                    free(margins);
                    return -1;
                }
                root->left = left_id;
                root->right = right_id;

                ctx->cv_losses[mi] += temp_tree_validation_loss_indexed(
                    ctx, 0, idx, weights, ctx->cv_fold_ids, n, fold);
            }
        }
    }

    int best = -1;
    double best_loss = INFINITY;
    for (int mi = 0; mi < n_margins; ++mi) {
        if (ctx->cv_valid[mi] && ctx->cv_losses[mi] < best_loss) {
            best_loss = ctx->cv_losses[mi];
            best = mi;
        }
    }
    if (best < 0 || !isfinite(best_loss)) {
        free(margins);
        return 0;
    }
    *selected_margin = margins[best];
    free(margins);
    return 1;
}

static int
cv_select_margin_classifier(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    int feature,
    double threshold,
    int nan_go_left,
    int depth,
    double *selected_margin)
{
    int folds = ctx->margin_cv_folds;
    int repeats = ctx->margin_cv_repeats > 1 ? ctx->margin_cv_repeats : 1;
    if (folds < 2 || n < (npy_intp)2 * (npy_intp)folds) return 0;
    if (!ctx->cv_perm || !ctx->cv_fold_ids || !ctx->cv_mu_buf ||
            !ctx->cv_losses || !ctx->cv_valid ||
            !ctx->cv_class_left || !ctx->cv_class_right) {
        return -1;
    }

    int max_margins = ctx->margin_grid_size + (
        ctx->include_hard_splits ? 1 : 0) + 1;
    if (max_margins < 1) max_margins = 1;
    double *margins = (double *)malloc((size_t)max_margins * sizeof(double));
    if (!margins) return -1;

    double feat_std = weighted_std_feature_raw(
        ctx->X, ctx->n_features, idx, weights, n, feature);
    int n_margins = build_margin_grid_la(
        feat_std, depth, ctx->include_hard_splits,
        ctx->margin_min_scale, ctx->margin_max_scale,
        ctx->margin_grid_size, ctx->margin_depth_decay,
        margins);
    if (n_margins <= 1) {
        free(margins);
        return 0;
    }

    for (int mi = 0; mi < n_margins; ++mi) {
        ctx->cv_losses[mi] = 0.0;
        ctx->cv_valid[mi] = 1;
    }

    double *train_w = ctx->cv_mu_buf;
    for (int rep = 0; rep < repeats; ++rep) {
        cv_make_fold_ids(ctx, n, folds);
        for (int fold = 0; fold < folds; ++fold) {
            npy_intp n_val = 0, n_train = 0;
            for (npy_intp i = 0; i < n; ++i) {
                if (ctx->cv_fold_ids[i] == fold) ++n_val;
                else ++n_train;
            }
            if (n_val == 0 || n_train == 0) continue;

            for (int mi = 0; mi < n_margins; ++mi) {
                if (!ctx->cv_valid[mi]) continue;

                for (npy_intp i = 0; i < n; ++i) {
                    train_w[i] = ctx->cv_fold_ids[i] == fold
                        ? 0.0 : weights[i];
                }

                double sl = 0.0, sr = 0.0;
                double margin = margins[mi];
                split_weights_raw(
                    ctx->X, ctx->n_features, idx, train_w, n,
                    feature, threshold, margin, nan_go_left,
                    ctx->candidate_left_w, ctx->candidate_right_w,
                    &sl, &sr);
                if (sl < ctx->min_samples_leaf ||
                        sr < ctx->min_samples_leaf) {
                    ctx->cv_valid[mi] = 0;
                    continue;
                }

                int node_count = 1;
                CNode *root = &ctx->temp_nodes[0];
                root->feature = feature;
                root->threshold = threshold;
                root->margin = margin;
                root->left = -1;
                root->right = -1;
                root->value = 0.0;
                root->n_samples = sl + sr;
                root->impurity_reduction = 0.0;
                root->nan_go_left = nan_go_left;

                int left_id = build_rollout_classifier(
                    ctx, idx, ctx->candidate_left_w, n, 1, &node_count);
                if (left_id < 0) {
                    free(margins);
                    return -1;
                }
                int right_id = build_rollout_classifier(
                    ctx, idx, ctx->candidate_right_w, n, 1, &node_count);
                if (right_id < 0) {
                    free(margins);
                    return -1;
                }
                root->left = left_id;
                root->right = right_id;

                ctx->cv_losses[mi] += temp_tree_validation_logloss_indexed(
                    ctx, 0, idx, weights, ctx->cv_fold_ids, n, fold);
            }
        }
    }

    int best = -1;
    double best_loss = INFINITY;
    for (int mi = 0; mi < n_margins; ++mi) {
        if (ctx->cv_valid[mi] && ctx->cv_losses[mi] < best_loss) {
            best_loss = ctx->cv_losses[mi];
            best = mi;
        }
    }
    if (best < 0 || !isfinite(best_loss)) {
        free(margins);
        return 0;
    }
    *selected_margin = margins[best];
    free(margins);
    return 1;
}

static int
build_rollout_regression(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    int depth,
    int *node_count)
{
    if (*node_count >= ctx->max_rollout_nodes) return -1;

    int node_id = *node_count;
    *node_count += 1;

    double eff = 0.0, max_w = 0.0;
    double value = weighted_mean_indexed(
        ctx->y, idx, weights, n, &eff, &max_w);
    (void)max_w;

    CNode *node = &ctx->temp_nodes[node_id];
    node->feature = -1;
    node->threshold = 0.0;
    node->margin = 0.0;
    node->left = -1;
    node->right = -1;
    node->value = value;
    node->n_samples = eff;
    node->impurity_reduction = 0.0;
    node->nan_go_left = 1;

    if (depth >= ctx->lookahead_horizon ||
        eff < 2.0 * ctx->min_samples_leaf) {
        return node_id;
    }

    SplitResult best;
    if (!lookahead_scan_regression(ctx, idx, weights, n, depth, 0, &best)) {
        return node_id;
    }

    double sl = 0.0, sr = 0.0;
    double *left_w = ctx->rollout_work[depth].left_w;
    double *right_w = ctx->rollout_work[depth].right_w;
    split_weights_raw(
        ctx->X, ctx->n_features, idx, weights, n,
        best.feature, best.threshold, best.margin, best.nan_go_left,
        left_w, right_w, &sl, &sr);
    if (sl < ctx->min_samples_leaf || sr < ctx->min_samples_leaf) {
        return node_id;
    }

    node->feature = best.feature;
    node->threshold = best.threshold;
    node->margin = best.margin;
    node->nan_go_left = best.nan_go_left;

    int left_id = build_rollout_regression(
        ctx, idx, left_w, n, depth + 1, node_count);
    if (left_id < 0) return -1;
    int right_id = build_rollout_regression(
        ctx, idx, right_w, n, depth + 1, node_count);
    if (right_id < 0) return -1;
    node->left = left_id;
    node->right = right_id;
    return node_id;
}

static double
temp_tree_validation_loss(
    LookaheadContext *ctx,
    int root_id,
    const double *val_weights)
{
    double loss = 0.0;
    double denom = 0.0;
    double unweighted_loss = 0.0;

    for (npy_intp i = 0; i < ctx->n_val; ++i) {
        double pred = 0.0;
        npy_intp top = 0;
        ctx->temp_stack_nodes[top] = root_id;
        ctx->temp_stack_weights[top] = 1.0;
        ++top;

        while (top > 0) {
            --top;
            npy_int64 node_idx = ctx->temp_stack_nodes[top];
            double path_w = ctx->temp_stack_weights[top];
            CNode *node = &ctx->temp_nodes[node_idx];
            if (node->left < 0) {
                pred += path_w * node->value;
                continue;
            }

            double x = ctx->Xv[i * ctx->n_features + node->feature];
            double mu = mu_left_numeric(
                x, node->threshold, node->margin, node->nan_go_left);
            if (mu > 0.0) {
                ctx->temp_stack_nodes[top] = node->left;
                ctx->temp_stack_weights[top] = path_w * mu;
                ++top;
            }
            if (mu < 1.0) {
                ctx->temp_stack_nodes[top] = node->right;
                ctx->temp_stack_weights[top] = path_w * (1.0 - mu);
                ++top;
            }
        }

        double r = ctx->yv[i] - pred;
        double wi = val_weights[i];
        loss += wi * r * r;
        denom += wi;
        unweighted_loss += r * r;
    }

    if (denom > 1e-12) return loss / denom;
    return ctx->n_val > 0 ? unweighted_loss / (double)ctx->n_val : 0.0;
}

static int
find_lookahead_split_regression(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    const double *val_weights,
    int depth,
    SplitResult *best)
{
    int n_short = lookahead_scan_regression(
        ctx, idx, weights, n, depth, 1, best);
    if (n_short <= 0) {
        best->feature = -1;
        return 0;
    }

    best->feature = -1;
    best->threshold_idx = -1;
    best->margin_idx = -1;
    best->imp = INFINITY;  /* validation loss while choosing */

    int max_margins = ctx->margin_grid_size + (ctx->include_hard_splits ? 1 : 0);
    if (max_margins < 1) max_margins = 1;
    double *margins = (double *)malloc((size_t)max_margins * sizeof(double));
    if (!margins) return -1;

    for (int ci = 0; ci < n_short; ++ci) {
        LAThresholdCandidate cand = ctx->shortlist[ci];
        double feat_std = weighted_std_feature_raw(
            ctx->X, ctx->n_features, idx, weights, n, cand.feature);
        int n_margins = build_margin_grid_la(
            feat_std, depth, ctx->include_hard_splits,
            ctx->margin_min_scale, ctx->margin_max_scale,
            ctx->margin_grid_size, ctx->margin_depth_decay,
            margins);

        for (int mi = 0; mi < n_margins; ++mi) {
            double sl = 0.0, sr = 0.0;
            double margin = margins[mi];
            split_weights_raw(
                ctx->X, ctx->n_features, idx, weights, n,
                cand.feature, cand.threshold, margin, cand.nan_go_left,
                ctx->candidate_left_w, ctx->candidate_right_w,
                &sl, &sr);
            if (sl < ctx->min_samples_leaf || sr < ctx->min_samples_leaf)
                continue;

            int node_count = 1;
            CNode *root = &ctx->temp_nodes[0];
            root->feature = cand.feature;
            root->threshold = cand.threshold;
            root->margin = margin;
            root->left = -1;
            root->right = -1;
            root->value = 0.0;
            root->n_samples = sl + sr;
            root->impurity_reduction = 0.0;
            root->nan_go_left = cand.nan_go_left;

            int left_id = build_rollout_regression(
                ctx, idx, ctx->candidate_left_w, n, 1, &node_count);
            if (left_id < 0) {
                free(margins);
                return -1;
            }
            int right_id = build_rollout_regression(
                ctx, idx, ctx->candidate_right_w, n, 1, &node_count);
            if (right_id < 0) {
                free(margins);
                return -1;
            }
            root->left = left_id;
            root->right = right_id;

            double loss = temp_tree_validation_loss(ctx, 0, val_weights);
            if (loss < best->imp) {
                best->imp = loss;
                best->feature = cand.feature;
                best->threshold_idx = cand.threshold_idx;
                best->margin_idx = mi;
                best->threshold = cand.threshold;
                best->margin = margin;
                best->nan_go_left = cand.nan_go_left;
            }
        }
    }

    free(margins);
    return best->feature >= 0 ? 1 : 0;
}

static int
build_lookahead_regression_numeric(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    const double *val_weights,
    int depth,
    int max_depth,
    LookaheadDepthWork *dw,
    CNode *nodes,
    int *node_count,
    int max_nodes)
{
    if (*node_count >= max_nodes) return -1;

    int node_id = *node_count;
    *node_count += 1;

    double eff = 0.0, max_w = 0.0;
    double value = weighted_mean_indexed(
        ctx->y, idx, weights, n, &eff, &max_w);

    nodes[node_id].feature = -1;
    nodes[node_id].threshold = 0.0;
    nodes[node_id].margin = 0.0;
    nodes[node_id].left = -1;
    nodes[node_id].right = -1;
    nodes[node_id].value = value;
    nodes[node_id].n_samples = eff;
    nodes[node_id].impurity_reduction = 0.0;
    nodes[node_id].nan_go_left = 1;

    if (depth >= max_depth || n < 2 ||
        eff < 2.0 * ctx->min_samples_leaf) {
        return node_id;
    }

    double val_eff = 0.0;
    for (npy_intp i = 0; i < ctx->n_val; ++i) val_eff += val_weights[i];

    SplitResult best;
    int found;
    if (val_eff < ctx->lookahead_min_val) {
        found = lookahead_scan_regression(
            ctx, idx, weights, n, depth, 0, &best);
    } else {
        found = find_lookahead_split_regression(
            ctx, idx, weights, n, val_weights, depth, &best);
        if (found < 0) return -1;
    }
    if (!found || best.feature < 0) return node_id;

    if (ctx->margin_cv_folds >= 2) {
        double cv_margin = best.margin;
        int cv_status = cv_select_margin_regression(
            ctx, idx, weights, n, best.feature, best.threshold,
            best.nan_go_left, depth, &cv_margin);
        if (cv_status < 0) return -1;
        if (cv_status > 0) best.margin = cv_margin;
    }

    LookaheadDepthWork *work = &dw[depth];
    double sl = 0.0, sr = 0.0;
    split_weights_raw(
        ctx->X, ctx->n_features, idx, weights, n,
        best.feature, best.threshold, best.margin, best.nan_go_left,
        work->left_w, work->right_w, &sl, &sr);

    double parent_mean = value;
    double parent_loss = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        double r = ctx->y[idx[i]] - parent_mean;
        parent_loss += weights[i] * r * r;
    }
    parent_loss = eff > 1e-15 ? parent_loss / eff : 0.0;
    double child_loss = split_child_loss_raw(
        ctx->y, idx, work->left_w, work->right_w, n, eff);

    double eps = max_w > 0.0 ? 1e-6 * max_w : 1e-10;
    if (ctx->min_train_weight_fraction > 0.0) {
        double frac_eps = ctx->min_train_weight_fraction * max_w;
        if (frac_eps > eps) eps = frac_eps;
    }

    npy_intp nl = 0, nr = 0;
    for (npy_intp i = 0; i < n; ++i) {
        if (work->left_w[i] > eps) {
            work->left_idx[nl] = idx[i];
            work->left_w[nl] = work->left_w[i];
            ++nl;
        }
        if (work->right_w[i] > eps) {
            work->right_idx[nr] = idx[i];
            work->right_w[nr] = work->right_w[i];
            ++nr;
        }
    }
    if (nl < 1 || nr < 1) return node_id;

    for (npy_intp i = 0; i < ctx->n_val; ++i) {
        double x = ctx->Xv[i * ctx->n_features + best.feature];
        double mu = mu_left_numeric(
            x, best.threshold, best.margin, best.nan_go_left);
        work->val_left_w[i] = val_weights[i] * mu;
        work->val_right_w[i] = val_weights[i] * (1.0 - mu);
    }

    nodes[node_id].feature = best.feature;
    nodes[node_id].threshold = best.threshold;
    nodes[node_id].margin = best.margin;
    nodes[node_id].impurity_reduction = (parent_loss - child_loss) * eff;
    nodes[node_id].nan_go_left = best.nan_go_left;

    int left_id = build_lookahead_regression_numeric(
        ctx, work->left_idx, work->left_w, nl, work->val_left_w,
        depth + 1, max_depth, dw, nodes, node_count, max_nodes);
    if (left_id < 0) return -1;
    int right_id = build_lookahead_regression_numeric(
        ctx, work->right_idx, work->right_w, nr, work->val_right_w,
        depth + 1, max_depth, dw, nodes, node_count, max_nodes);
    if (right_id < 0) return -1;

    nodes[node_id].left = left_id;
    nodes[node_id].right = right_id;
    return node_id;
}

static double
fill_class_node_summary(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    int node_id,
    double *eff_out,
    double *max_weight_out)
{
    int K = ctx->n_classes;
    for (int c = 0; c < K; ++c) ctx->class_left_buf[c] = 0.0;

    double eff = 0.0, max_w = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        double wi = weights[i];
        int cls = (int)ctx->y[idx[i]];
        if (cls >= 0 && cls < K) ctx->class_left_buf[cls] += wi;
        eff += wi;
        if (wi > max_w) max_w = wi;
    }

    int best_class = 0;
    double best_weight = K > 0 ? ctx->class_left_buf[0] : 0.0;
    for (int c = 1; c < K; ++c) {
        if (ctx->class_left_buf[c] > best_weight) {
            best_weight = ctx->class_left_buf[c];
            best_class = c;
        }
    }

    if (ctx->temp_leaf_probs && node_id >= 0 &&
            node_id < ctx->max_rollout_nodes) {
        double *p = ctx->temp_leaf_probs + (size_t)node_id * (size_t)K;
        if (eff > 1e-12) {
            for (int c = 0; c < K; ++c) p[c] = ctx->class_left_buf[c] / eff;
        } else {
            double uniform = K > 0 ? 1.0 / (double)K : 0.0;
            for (int c = 0; c < K; ++c) p[c] = uniform;
        }
    }

    if (eff_out) *eff_out = eff;
    if (max_weight_out) *max_weight_out = max_w;
    return (double)best_class;
}

static double
split_child_impurity_classifier_raw(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *left_w,
    const double *right_w,
    npy_intp n,
    double parent_eff)
{
    int K = ctx->n_classes;
    for (int c = 0; c < K; ++c) {
        ctx->class_left_buf[c] = 0.0;
        ctx->class_total_buf[c] = 0.0;
    }

    double sl = 0.0, sr = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        int cls = (int)ctx->y[idx[i]];
        if (cls < 0 || cls >= K) continue;
        double wl = left_w[i];
        double wr = right_w[i];
        ctx->class_left_buf[cls] += wl;
        ctx->class_total_buf[cls] += wl + wr;
        sl += wl;
        sr += wr;
    }
    if (sl < 1e-15 || sr < 1e-15 || parent_eff < 1e-15) return 0.0;

    double left_imp = class_impurity_from_counts(
        ctx->class_left_buf, K, sl, ctx->split_criterion);
    for (int c = 0; c < K; ++c) {
        ctx->class_total_buf[c] -= ctx->class_left_buf[c];
    }
    double right_imp = class_impurity_from_counts(
        ctx->class_total_buf, K, sr, ctx->split_criterion);
    return (sl * left_imp + sr * right_imp) / parent_eff;
}

static int
build_rollout_classifier(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    int depth,
    int *node_count)
{
    if (*node_count >= ctx->max_rollout_nodes) return -1;

    int node_id = *node_count;
    *node_count += 1;

    double eff = 0.0, max_w = 0.0;
    double value = fill_class_node_summary(
        ctx, idx, weights, n, node_id, &eff, &max_w);
    (void)max_w;

    CNode *node = &ctx->temp_nodes[node_id];
    node->feature = -1;
    node->threshold = 0.0;
    node->margin = 0.0;
    node->left = -1;
    node->right = -1;
    node->value = value;
    node->n_samples = eff;
    node->impurity_reduction = 0.0;
    node->nan_go_left = 1;

    if (depth >= ctx->lookahead_horizon ||
        eff < 2.0 * ctx->min_samples_leaf) {
        return node_id;
    }

    SplitResult best;
    if (!lookahead_scan_classifier(ctx, idx, weights, n, depth, 0, &best)) {
        return node_id;
    }

    double sl = 0.0, sr = 0.0;
    double *left_w = ctx->rollout_work[depth].left_w;
    double *right_w = ctx->rollout_work[depth].right_w;
    split_weights_raw(
        ctx->X, ctx->n_features, idx, weights, n,
        best.feature, best.threshold, best.margin, best.nan_go_left,
        left_w, right_w, &sl, &sr);
    if (sl < ctx->min_samples_leaf || sr < ctx->min_samples_leaf) {
        return node_id;
    }

    node->feature = best.feature;
    node->threshold = best.threshold;
    node->margin = best.margin;
    node->nan_go_left = best.nan_go_left;

    int left_id = build_rollout_classifier(
        ctx, idx, left_w, n, depth + 1, node_count);
    if (left_id < 0) return -1;
    int right_id = build_rollout_classifier(
        ctx, idx, right_w, n, depth + 1, node_count);
    if (right_id < 0) return -1;
    node->left = left_id;
    node->right = right_id;
    return node_id;
}

static double
temp_tree_validation_logloss(
    LookaheadContext *ctx,
    int root_id,
    const double *val_weights)
{
    int K = ctx->n_classes;
    double loss = 0.0;
    double denom = 0.0;
    double unweighted_loss = 0.0;

    for (npy_intp i = 0; i < ctx->n_val; ++i) {
        for (int c = 0; c < K; ++c) ctx->temp_pred_probs[c] = 0.0;

        npy_intp top = 0;
        ctx->temp_stack_nodes[top] = root_id;
        ctx->temp_stack_weights[top] = 1.0;
        ++top;

        while (top > 0) {
            --top;
            npy_int64 node_idx = ctx->temp_stack_nodes[top];
            double path_w = ctx->temp_stack_weights[top];
            CNode *node = &ctx->temp_nodes[node_idx];
            if (node->left < 0) {
                double *p = ctx->temp_leaf_probs
                    + (size_t)node_idx * (size_t)K;
                for (int c = 0; c < K; ++c)
                    ctx->temp_pred_probs[c] += path_w * p[c];
                continue;
            }

            double x = ctx->Xv[i * ctx->n_features + node->feature];
            double mu = mu_left_numeric(
                x, node->threshold, node->margin, node->nan_go_left);
            if (mu > 0.0) {
                ctx->temp_stack_nodes[top] = node->left;
                ctx->temp_stack_weights[top] = path_w * mu;
                ++top;
            }
            if (mu < 1.0) {
                ctx->temp_stack_nodes[top] = node->right;
                ctx->temp_stack_weights[top] = path_w * (1.0 - mu);
                ++top;
            }
        }

        int cls = (int)ctx->yv[i];
        double p = (cls >= 0 && cls < K) ? ctx->temp_pred_probs[cls] : 0.0;
        if (p < 1e-12) p = 1e-12;
        if (p > 1.0) p = 1.0;
        double li = -log(p);
        double wi = val_weights[i];
        loss += wi * li;
        denom += wi;
        unweighted_loss += li;
    }

    if (denom > 1e-12) return loss / denom;
    return ctx->n_val > 0 ? unweighted_loss / (double)ctx->n_val : 0.0;
}

static int
find_lookahead_split_classifier(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    const double *val_weights,
    int depth,
    SplitResult *best)
{
    int n_short = lookahead_scan_classifier(
        ctx, idx, weights, n, depth, 1, best);
    if (n_short <= 0) {
        best->feature = -1;
        return 0;
    }

    best->feature = -1;
    best->threshold_idx = -1;
    best->margin_idx = -1;
    best->imp = INFINITY;

    int max_margins = ctx->margin_grid_size + (ctx->include_hard_splits ? 1 : 0);
    if (max_margins < 1) max_margins = 1;
    double *margins = (double *)malloc((size_t)max_margins * sizeof(double));
    if (!margins) return -1;

    for (int ci = 0; ci < n_short; ++ci) {
        LAThresholdCandidate cand = ctx->shortlist[ci];
        double feat_std = weighted_std_feature_raw(
            ctx->X, ctx->n_features, idx, weights, n, cand.feature);
        int n_margins = build_margin_grid_la(
            feat_std, depth, ctx->include_hard_splits,
            ctx->margin_min_scale, ctx->margin_max_scale,
            ctx->margin_grid_size, ctx->margin_depth_decay,
            margins);

        for (int mi = 0; mi < n_margins; ++mi) {
            double sl = 0.0, sr = 0.0;
            double margin = margins[mi];
            split_weights_raw(
                ctx->X, ctx->n_features, idx, weights, n,
                cand.feature, cand.threshold, margin, cand.nan_go_left,
                ctx->candidate_left_w, ctx->candidate_right_w,
                &sl, &sr);
            if (sl < ctx->min_samples_leaf || sr < ctx->min_samples_leaf)
                continue;

            int node_count = 1;
            CNode *root = &ctx->temp_nodes[0];
            root->feature = cand.feature;
            root->threshold = cand.threshold;
            root->margin = margin;
            root->left = -1;
            root->right = -1;
            root->value = 0.0;
            root->n_samples = sl + sr;
            root->impurity_reduction = 0.0;
            root->nan_go_left = cand.nan_go_left;

            int left_id = build_rollout_classifier(
                ctx, idx, ctx->candidate_left_w, n, 1, &node_count);
            if (left_id < 0) {
                free(margins);
                return -1;
            }
            int right_id = build_rollout_classifier(
                ctx, idx, ctx->candidate_right_w, n, 1, &node_count);
            if (right_id < 0) {
                free(margins);
                return -1;
            }
            root->left = left_id;
            root->right = right_id;

            double loss = temp_tree_validation_logloss(ctx, 0, val_weights);
            if (loss < best->imp) {
                best->imp = loss;
                best->feature = cand.feature;
                best->threshold_idx = cand.threshold_idx;
                best->margin_idx = mi;
                best->threshold = cand.threshold;
                best->margin = margin;
                best->nan_go_left = cand.nan_go_left;
            }
        }
    }

    free(margins);
    return best->feature >= 0 ? 1 : 0;
}

static int
build_lookahead_classifier_numeric(
    LookaheadContext *ctx,
    const npy_intp *idx,
    const double *weights,
    npy_intp n,
    const double *val_weights,
    int depth,
    int max_depth,
    LookaheadDepthWork *dw,
    CNode *nodes,
    int *node_count,
    int max_nodes)
{
    if (*node_count >= max_nodes) return -1;

    int node_id = *node_count;
    *node_count += 1;

    double eff = 0.0, max_w = 0.0;
    double value = fill_class_node_summary(
        ctx, idx, weights, n, node_id, &eff, &max_w);

    nodes[node_id].feature = -1;
    nodes[node_id].threshold = 0.0;
    nodes[node_id].margin = 0.0;
    nodes[node_id].left = -1;
    nodes[node_id].right = -1;
    nodes[node_id].value = value;
    nodes[node_id].n_samples = eff;
    nodes[node_id].impurity_reduction = 0.0;
    nodes[node_id].nan_go_left = 1;

    if (depth >= max_depth || n < 2 ||
        eff < 2.0 * ctx->min_samples_leaf) {
        return node_id;
    }

    double val_eff = 0.0;
    for (npy_intp i = 0; i < ctx->n_val; ++i) val_eff += val_weights[i];

    SplitResult best;
    int found;
    if (val_eff < ctx->lookahead_min_val) {
        found = lookahead_scan_classifier(
            ctx, idx, weights, n, depth, 0, &best);
    } else {
        found = find_lookahead_split_classifier(
            ctx, idx, weights, n, val_weights, depth, &best);
        if (found < 0) return -1;
    }
    if (!found || best.feature < 0) return node_id;

    if (ctx->margin_cv_folds >= 2) {
        double cv_margin = best.margin;
        int cv_status = cv_select_margin_classifier(
            ctx, idx, weights, n, best.feature, best.threshold,
            best.nan_go_left, depth, &cv_margin);
        if (cv_status < 0) return -1;
        if (cv_status > 0) best.margin = cv_margin;
    }

    LookaheadDepthWork *work = &dw[depth];
    double sl = 0.0, sr = 0.0;
    split_weights_raw(
        ctx->X, ctx->n_features, idx, weights, n,
        best.feature, best.threshold, best.margin, best.nan_go_left,
        work->left_w, work->right_w, &sl, &sr);

    double parent_impurity;
    {
        for (int c = 0; c < ctx->n_classes; ++c)
            ctx->class_total_buf[c] = 0.0;
        for (npy_intp i = 0; i < n; ++i) {
            int cls = (int)ctx->y[idx[i]];
            if (cls >= 0 && cls < ctx->n_classes)
                ctx->class_total_buf[cls] += weights[i];
        }
        parent_impurity = class_impurity_from_counts(
            ctx->class_total_buf, ctx->n_classes, eff,
            ctx->split_criterion);
    }
    double child_impurity = split_child_impurity_classifier_raw(
        ctx, idx, work->left_w, work->right_w, n, eff);

    double eps = max_w > 0.0 ? 1e-6 * max_w : 1e-10;
    if (ctx->min_train_weight_fraction > 0.0) {
        double frac_eps = ctx->min_train_weight_fraction * max_w;
        if (frac_eps > eps) eps = frac_eps;
    }

    npy_intp nl = 0, nr = 0;
    for (npy_intp i = 0; i < n; ++i) {
        if (work->left_w[i] > eps) {
            work->left_idx[nl] = idx[i];
            work->left_w[nl] = work->left_w[i];
            ++nl;
        }
        if (work->right_w[i] > eps) {
            work->right_idx[nr] = idx[i];
            work->right_w[nr] = work->right_w[i];
            ++nr;
        }
    }
    if (nl < 1 || nr < 1) return node_id;

    for (npy_intp i = 0; i < ctx->n_val; ++i) {
        double x = ctx->Xv[i * ctx->n_features + best.feature];
        double mu = mu_left_numeric(
            x, best.threshold, best.margin, best.nan_go_left);
        work->val_left_w[i] = val_weights[i] * mu;
        work->val_right_w[i] = val_weights[i] * (1.0 - mu);
    }

    nodes[node_id].feature = best.feature;
    nodes[node_id].threshold = best.threshold;
    nodes[node_id].margin = best.margin;
    nodes[node_id].impurity_reduction = (
        parent_impurity - child_impurity) * eff;
    nodes[node_id].nan_go_left = best.nan_go_left;

    int left_id = build_lookahead_classifier_numeric(
        ctx, work->left_idx, work->left_w, nl, work->val_left_w,
        depth + 1, max_depth, dw, nodes, node_count, max_nodes);
    if (left_id < 0) return -1;
    int right_id = build_lookahead_classifier_numeric(
        ctx, work->right_idx, work->right_w, nr, work->val_right_w,
        depth + 1, max_depth, dw, nodes, node_count, max_nodes);
    if (right_id < 0) return -1;

    nodes[node_id].left = left_id;
    nodes[node_id].right = right_id;
    return node_id;
}

/* ------------------------------------------------------------------ */
/* Python-accessible functions                                         */
/* ------------------------------------------------------------------ */

static PyObject *
find_mse_split(PyObject *self, PyObject *args)
{
    PyObject *y_obj, *w_obj, *bin_assign_obj, *bin_centers_obj;
    PyObject *candidates_obj, *margins_obj;
    double min_samples_leaf, nan_mu;
    int optimize_split_gain;
    double split_gain_l2;

    if (!PyArg_ParseTuple(args, "OOOOOOddpd",
                          &y_obj, &w_obj, &bin_assign_obj,
                          &bin_centers_obj, &candidates_obj, &margins_obj,
                          &min_samples_leaf, &nan_mu, &optimize_split_gain,
                          &split_gain_l2))
        return NULL;

    PyArrayObject *y_arr = (PyArrayObject *)PyArray_FROM_OTF(
        y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *bin_assign_arr = (PyArrayObject *)PyArray_FROM_OTF(
        bin_assign_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *bin_centers_arr = (PyArrayObject *)PyArray_FROM_OTF(
        bin_centers_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *candidates_arr = (PyArrayObject *)PyArray_FROM_OTF(
        candidates_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *margins_arr = (PyArrayObject *)PyArray_FROM_OTF(
        margins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!y_arr || !w_arr || !bin_assign_arr ||
        !bin_centers_arr || !candidates_arr || !margins_arr) {
        Py_XDECREF(y_arr); Py_XDECREF(w_arr); Py_XDECREF(bin_assign_arr);
        Py_XDECREF(bin_centers_arr); Py_XDECREF(candidates_arr);
        Py_XDECREF(margins_arr);
        return NULL;
    }

    npy_intp N = PyArray_SIZE(y_arr);
    npy_intp B = PyArray_SIZE(bin_centers_arr);
    npy_intp T = PyArray_SIZE(candidates_arr);
    npy_intp S = PyArray_SIZE(margins_arr);

    if (PyArray_SIZE(w_arr) != N || PyArray_SIZE(bin_assign_arr) != N) {
        PyErr_SetString(PyExc_ValueError,
                        "y, w, and bin_assign must have the same length");
        goto fail;
    }
    if (B > (npy_intp)INT_MAX || T > (npy_intp)INT_MAX ||
            S > (npy_intp)INT_MAX) {
        PyErr_SetString(PyExc_ValueError,
                        "too many bins, candidates, or margins");
        goto fail;
    }

    {
        double *y          = (double *)PyArray_DATA(y_arr);
        double *w          = (double *)PyArray_DATA(w_arr);
        int    *bin_assign = (int    *)PyArray_DATA(bin_assign_arr);
        double *bin_centers= (double *)PyArray_DATA(bin_centers_arr);
        double *candidates = (double *)PyArray_DATA(candidates_arr);
        double *margins    = (double *)PyArray_DATA(margins_arr);

        double *bin_w   = (double *)calloc((size_t)B, sizeof(double));
        double *bin_wy  = (double *)calloc((size_t)B, sizeof(double));
        double *bin_wy2 = (double *)calloc((size_t)B, sizeof(double));
        double *prefix_w = (double *)malloc((size_t)(B + 1) * sizeof(double));
        double *prefix_wy = (double *)malloc((size_t)(B + 1) * sizeof(double));
        double *prefix_wy2 = (double *)malloc((size_t)(B + 1) * sizeof(double));
        if (!bin_w || !bin_wy || !bin_wy2 ||
                !prefix_w || !prefix_wy || !prefix_wy2) {
            PyErr_NoMemory();
            free(bin_w); free(bin_wy); free(bin_wy2);
            free(prefix_w); free(prefix_wy); free(prefix_wy2);
            goto fail;
        }

        SplitResult best;
        best.imp = 0.0;
        best.feature = -1;
        best.threshold_idx = -1;
        best.margin_idx = -1;
        best.threshold = 0.0;
        best.margin = 0.0;
        best.nan_go_left = nan_mu == 1.0 ? 1 : 0;

        Py_BEGIN_ALLOW_THREADS

        double nan_w = 0.0, nan_wy = 0.0, nan_wy2 = 0.0;
        for (npy_intp i = 0; i < N; ++i) {
            int b    = bin_assign[i];
            double wi = w[i], yi = y[i];
            if (b < 0) {
                nan_w += wi; nan_wy += wi*yi; nan_wy2 += wi*yi*yi;
            } else if (b < B) {
                bin_w[b] += wi; bin_wy[b] += wi*yi; bin_wy2[b] += wi*yi*yi;
            }
        }

        double w_total = nan_w, wy_total = nan_wy, wy2_total = nan_wy2;
        for (npy_intp b = 0; b < B; ++b) {
            w_total += bin_w[b]; wy_total += bin_wy[b]; wy2_total += bin_wy2[b];
        }

        if (w_total >= 1e-15) {
            build_prefix_sums(
                bin_w, bin_wy, bin_wy2, (int)B,
                prefix_w, prefix_wy, prefix_wy2);
            eval_feature_splits(
                bin_w, bin_wy, bin_wy2, prefix_w, prefix_wy, prefix_wy2,
                nan_w, nan_wy, nan_wy2, nan_mu,
                w_total, wy_total, wy2_total, (int)B,
                candidates, bin_centers, (int)T,
                margins, (int)S,
                min_samples_leaf, optimize_split_gain, split_gain_l2,
                0, 1, &best);
        }

        Py_END_ALLOW_THREADS

        free(bin_w); free(bin_wy); free(bin_wy2);
        free(prefix_w); free(prefix_wy); free(prefix_wy2);
        Py_DECREF(y_arr); Py_DECREF(w_arr); Py_DECREF(bin_assign_arr);
        Py_DECREF(bin_centers_arr); Py_DECREF(candidates_arr);
        Py_DECREF(margins_arr);

        if (best.threshold_idx < 0) return Py_BuildValue("iid", -1, -1, 0.0);
        return Py_BuildValue("iid", best.threshold_idx, best.margin_idx, best.imp);
    }

fail:
    Py_XDECREF(y_arr); Py_XDECREF(w_arr); Py_XDECREF(bin_assign_arr);
    Py_XDECREF(bin_centers_arr); Py_XDECREF(candidates_arr);
    Py_XDECREF(margins_arr);
    return NULL;
}

static PyObject *
predict_numeric_tree(PyObject *self, PyObject *args)
{
    PyObject *X_obj, *features_obj, *thresholds_obj, *margins_obj;
    PyObject *lefts_obj, *rights_obj, *values_obj, *nan_go_left_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &X_obj, &features_obj, &thresholds_obj,
                          &margins_obj, &lefts_obj, &rights_obj,
                          &values_obj, &nan_go_left_obj))
        return NULL;

    PyArrayObject *X_arr = (PyArrayObject *)PyArray_FROM_OTF(
        X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_arr = (PyArrayObject *)PyArray_FROM_OTF(
        features_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_arr = (PyArrayObject *)PyArray_FROM_OTF(
        thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *margins_arr = (PyArrayObject *)PyArray_FROM_OTF(
        margins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *lefts_arr = (PyArrayObject *)PyArray_FROM_OTF(
        lefts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *rights_arr = (PyArrayObject *)PyArray_FROM_OTF(
        rights_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_arr = (PyArrayObject *)PyArray_FROM_OTF(
        values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *nan_go_left_arr = (PyArrayObject *)PyArray_FROM_OTF(
        nan_go_left_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);

    if (!X_arr || !features_arr || !thresholds_arr || !margins_arr ||
        !lefts_arr || !rights_arr || !values_arr || !nan_go_left_arr) {
        Py_XDECREF(X_arr); Py_XDECREF(features_arr); Py_XDECREF(thresholds_arr);
        Py_XDECREF(margins_arr); Py_XDECREF(lefts_arr); Py_XDECREF(rights_arr);
        Py_XDECREF(values_arr); Py_XDECREF(nan_go_left_arr);
        return NULL;
    }

    if (PyArray_NDIM(X_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "X must be a 2D float64 array");
        goto predict_fail;
    }

    {
        npy_intp n_samples  = PyArray_DIM(X_arr, 0);
        npy_intp n_features = PyArray_DIM(X_arr, 1);
        npy_intp n_nodes    = PyArray_SIZE(values_arr);

        if (PyArray_SIZE(features_arr) != n_nodes ||
            PyArray_SIZE(thresholds_arr) != n_nodes ||
            PyArray_SIZE(margins_arr) != n_nodes ||
            PyArray_SIZE(lefts_arr) != n_nodes ||
            PyArray_SIZE(rights_arr) != n_nodes ||
            PyArray_SIZE(nan_go_left_arr) != n_nodes) {
            PyErr_SetString(PyExc_ValueError,
                            "tree arrays must all have the same length");
            goto predict_fail;
        }

        npy_intp out_dims[1] = {n_samples};
        PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(
            1, out_dims, NPY_DOUBLE);
        if (!out_arr) goto predict_fail;

        npy_int64 *stack_nodes   = (npy_int64 *)malloc(
            (size_t)n_nodes * sizeof(npy_int64));
        double    *stack_weights = (double    *)malloc(
            (size_t)n_nodes * sizeof(double));
        if (!stack_nodes || !stack_weights) {
            PyErr_NoMemory();
            free(stack_nodes); free(stack_weights);
            Py_DECREF(out_arr);
            goto predict_fail;
        }

        double    *X          = (double    *)PyArray_DATA(X_arr);
        npy_int64 *features   = (npy_int64 *)PyArray_DATA(features_arr);
        double    *thresholds = (double    *)PyArray_DATA(thresholds_arr);
        double    *margins    = (double    *)PyArray_DATA(margins_arr);
        npy_int64 *lefts      = (npy_int64 *)PyArray_DATA(lefts_arr);
        npy_int64 *rights     = (npy_int64 *)PyArray_DATA(rights_arr);
        double    *values     = (double    *)PyArray_DATA(values_arr);
        npy_bool  *nan_go_left= (npy_bool  *)PyArray_DATA(nan_go_left_arr);
        double    *out        = (double    *)PyArray_DATA(out_arr);

        Py_BEGIN_ALLOW_THREADS

        for (npy_intp i = 0; i < n_samples; ++i) {
            double pred = 0.0;
            npy_intp top = 0;
            stack_nodes[top]   = 0;
            stack_weights[top] = 1.0;
            ++top;

            while (top > 0) {
                --top;
                npy_int64 nidx   = stack_nodes[top];
                double    weight = stack_weights[top];

                npy_int64 left_idx = lefts[nidx];
                if (left_idx < 0) { pred += weight * values[nidx]; continue; }

                double x  = X[i * n_features + features[nidx]];
                double mu_l;
                if (isnan(x)) {
                    mu_l = nan_go_left[nidx] ? 1.0 : 0.0;
                } else if (margins[nidx] < 1e-12) {
                    mu_l = x <= thresholds[nidx] ? 1.0 : 0.0;
                } else {
                    mu_l = membership_value(
                        (x - thresholds[nidx]) / margins[nidx]);
                }
                if (mu_l > 0.0) {
                    stack_nodes[top]   = left_idx;
                    stack_weights[top] = weight * mu_l;
                    ++top;
                }
                if (mu_l < 1.0) {
                    stack_nodes[top]   = rights[nidx];
                    stack_weights[top] = weight * (1.0 - mu_l);
                    ++top;
                }
            }
            out[i] = pred;
        }

        Py_END_ALLOW_THREADS

        free(stack_nodes); free(stack_weights);
        Py_DECREF(X_arr); Py_DECREF(features_arr); Py_DECREF(thresholds_arr);
        Py_DECREF(margins_arr); Py_DECREF(lefts_arr); Py_DECREF(rights_arr);
        Py_DECREF(values_arr); Py_DECREF(nan_go_left_arr);
        return (PyObject *)out_arr;
    }

predict_fail:
    Py_XDECREF(X_arr); Py_XDECREF(features_arr); Py_XDECREF(thresholds_arr);
    Py_XDECREF(margins_arr); Py_XDECREF(lefts_arr); Py_XDECREF(rights_arr);
    Py_XDECREF(values_arr); Py_XDECREF(nan_go_left_arr);
    return NULL;
}

static PyObject *
grow_depth_first_regression(PyObject *self, PyObject *args)
{
    PyObject *X_obj, *X_bins_obj, *y_obj, *w_obj;
    PyObject *thresholds_obj, *centers_obj, *n_thresholds_obj;
    PyObject *feature_choices_obj, *feature_counts_obj;
    int max_depth;
    double min_samples_leaf;
    int margin_grid_size;
    double margin_depth_decay, min_train_weight_fraction;
    int optimize_split_gain;
    double split_gain_l2;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOididdpd",
                          &X_obj, &X_bins_obj, &y_obj, &w_obj,
                          &thresholds_obj, &centers_obj, &n_thresholds_obj,
                          &feature_choices_obj, &feature_counts_obj,
                          &max_depth, &min_samples_leaf, &margin_grid_size,
                          &margin_depth_decay, &min_train_weight_fraction,
                          &optimize_split_gain, &split_gain_l2))
        return NULL;

    PyArrayObject *X_arr = (PyArrayObject *)PyArray_FROM_OTF(
        X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_bins_arr = (PyArrayObject *)PyArray_FROM_OTF(
        X_bins_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_arr = (PyArrayObject *)PyArray_FROM_OTF(
        y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_arr = (PyArrayObject *)PyArray_FROM_OTF(
        thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *centers_arr = (PyArrayObject *)PyArray_FROM_OTF(
        centers_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *n_thresholds_arr = (PyArrayObject *)PyArray_FROM_OTF(
        n_thresholds_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *feature_choices_arr = (PyArrayObject *)PyArray_FROM_OTF(
        feature_choices_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *feature_counts_arr = (PyArrayObject *)PyArray_FROM_OTF(
        feature_counts_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!X_arr || !X_bins_arr || !y_arr || !w_arr || !thresholds_arr ||
        !centers_arr || !n_thresholds_arr || !feature_choices_arr ||
        !feature_counts_arr) {
        Py_XDECREF(X_arr); Py_XDECREF(X_bins_arr); Py_XDECREF(y_arr);
        Py_XDECREF(w_arr); Py_XDECREF(thresholds_arr); Py_XDECREF(centers_arr);
        Py_XDECREF(n_thresholds_arr); Py_XDECREF(feature_choices_arr);
        Py_XDECREF(feature_counts_arr);
        return NULL;
    }

    if (PyArray_NDIM(X_arr) != 2 || PyArray_NDIM(X_bins_arr) != 2 ||
        PyArray_NDIM(thresholds_arr) != 2 || PyArray_NDIM(centers_arr) != 2 ||
        PyArray_NDIM(feature_choices_arr) != 2) {
        PyErr_SetString(PyExc_ValueError,
            "X, X_bins, thresholds, centers, and feature_choices must be 2D");
        goto grow_fail;
    }

    {
        npy_intp n_samples  = PyArray_DIM(X_arr, 0);
        npy_intp n_features = PyArray_DIM(X_arr, 1);

        if (PyArray_DIM(X_bins_arr, 0) != n_samples ||
            PyArray_DIM(X_bins_arr, 1) != n_features ||
            PyArray_SIZE(y_arr) != n_samples ||
            PyArray_SIZE(w_arr) != n_samples ||
            PyArray_DIM(thresholds_arr, 0) != n_features ||
            PyArray_DIM(centers_arr, 0) != n_features ||
            PyArray_SIZE(n_thresholds_arr) != n_features ||
            PyArray_SIZE(feature_counts_arr) != PyArray_DIM(feature_choices_arr, 0)) {
            PyErr_SetString(PyExc_ValueError,
                            "input array dimensions do not match");
            goto grow_fail;
        }

        int max_thresholds      = (int)PyArray_DIM(thresholds_arr, 1);
        int feature_choice_width= (int)PyArray_DIM(feature_choices_arr, 1);

        if (PyArray_DIM(centers_arr, 1) != max_thresholds + 1) {
            PyErr_SetString(PyExc_ValueError,
                            "centers width must equal thresholds width + 1");
            goto grow_fail;
        }
        if (max_depth < 0 || max_depth > 30 || margin_grid_size < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid max_depth or margin_grid_size");
            goto grow_fail;
        }

        int max_nodes = (1 << (max_depth + 1)) - 1;
        CNode *nodes = (CNode *)calloc((size_t)max_nodes, sizeof(CNode));

        /* Initial index array. */
        npy_intp *idx = (npy_intp *)malloc((size_t)n_samples * sizeof(npy_intp));

        /* Pre-allocate depth workspace — one level per depth (0..max_depth-1).
           Leaves at max_depth don't split, so we need max_depth levels. */
        int n_dw = max_depth > 0 ? max_depth : 1;
        DepthWork *dw = (DepthWork *)malloc((size_t)n_dw * sizeof(DepthWork));

        /* Per-thread workspaces for the parallel candidate-split search. */
        int n_threads = 1;
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
        if (n_threads < 1) n_threads = 1;
#endif
        /* Prefix sums (3 arrays of max_thresholds+2) per thread. */
        double *prefix_ws = (double *)malloc(
            (size_t)n_threads * 3 * (size_t)(max_thresholds + 2)
            * sizeof(double));
        /* Margin grid per thread. */
        double *margin_ws = (double *)malloc(
            (size_t)n_threads * (size_t)margin_grid_size * sizeof(double));
        /* One per-candidate best-split slot (written without contention). */
        SplitResult *cand_best = (SplitResult *)malloc(
            (size_t)n_features * sizeof(SplitResult));

        /* One-pass histogram workspace: one (bin_w,bin_wy,bin_wy2) block per
           candidate feature, plus per-feature NaN stats and int scratch. */
        double *hist_buf = (double *)malloc(
            (size_t)n_features * 3 * (size_t)(max_thresholds + 1)
            * sizeof(double));
        double *nan_buf = (double *)malloc(
            (size_t)n_features * 3 * sizeof(double));
        int *int_scratch = (int *)malloc(
            (size_t)n_features * 5 * sizeof(int));

        if (!nodes || !idx || !dw || !prefix_ws || !margin_ws || !cand_best ||
            !hist_buf || !nan_buf || !int_scratch) {
            PyErr_NoMemory();
            free(nodes); free(idx); free(dw);
            free(prefix_ws); free(margin_ws); free(cand_best);
            free(hist_buf); free(nan_buf); free(int_scratch);
            goto grow_fail;
        }

        int alloc_ok = 1;
        for (int d = 0; d < n_dw && alloc_ok; ++d) {
            dw[d].left_idx  = (npy_intp *)malloc((size_t)n_samples * sizeof(npy_intp));
            dw[d].right_idx = (npy_intp *)malloc((size_t)n_samples * sizeof(npy_intp));
            dw[d].left_w    = (double   *)malloc((size_t)n_samples * sizeof(double));
            dw[d].right_w   = (double   *)malloc((size_t)n_samples * sizeof(double));
            if (!dw[d].left_idx || !dw[d].right_idx ||
                !dw[d].left_w   || !dw[d].right_w)
                alloc_ok = 0;
        }
        if (!alloc_ok) {
            PyErr_NoMemory();
            for (int d = 0; d < n_dw; ++d) {
                free(dw[d].left_idx); free(dw[d].right_idx);
                free(dw[d].left_w);   free(dw[d].right_w);
            }
            free(nodes); free(idx); free(dw);
            free(prefix_ws); free(margin_ws); free(cand_best);
            free(hist_buf); free(nan_buf); free(int_scratch);
            goto grow_fail;
        }

        for (npy_intp i = 0; i < n_samples; ++i) idx[i] = i;

        int node_count = 0, status = 0;
        double *X      = (double *)PyArray_DATA(X_arr);
        int    *X_bins = (int    *)PyArray_DATA(X_bins_arr);
        double *y      = (double *)PyArray_DATA(y_arr);
        double *w      = (double *)PyArray_DATA(w_arr);
        double *thresholds = (double *)PyArray_DATA(thresholds_arr);
        double *centers    = (double *)PyArray_DATA(centers_arr);
        int    *n_thresholds    = (int *)PyArray_DATA(n_thresholds_arr);
        int    *feature_choices = (int *)PyArray_DATA(feature_choices_arr);
        int    *feature_counts  = (int *)PyArray_DATA(feature_counts_arr);
        int feature_choice_cursor = 0;

        Py_BEGIN_ALLOW_THREADS
        status = build_depth_first_numeric(
            X, X_bins, y, n_features, idx, w, n_samples,
            0.0, 0.0, 0.0, 0, 0, max_depth,
            thresholds, centers, n_thresholds, max_thresholds,
            margin_grid_size, margin_depth_decay, min_samples_leaf,
            min_train_weight_fraction, optimize_split_gain, split_gain_l2,
            feature_choices, feature_counts, feature_choice_width,
            &feature_choice_cursor, dw, prefix_ws, margin_ws,
            hist_buf, nan_buf, int_scratch, cand_best,
            nodes, &node_count, max_nodes);
        Py_END_ALLOW_THREADS

        free(idx); free(prefix_ws); free(margin_ws); free(cand_best);
        free(hist_buf); free(nan_buf); free(int_scratch);
        for (int d = 0; d < n_dw; ++d) {
            free(dw[d].left_idx); free(dw[d].right_idx);
            free(dw[d].left_w);   free(dw[d].right_w);
        }
        free(dw);

        if (status < 0) {
            free(nodes);
            PyErr_SetString(PyExc_RuntimeError, "C tree grower failed");
            goto grow_fail;
        }

        npy_intp dims[1] = {node_count};
        PyArrayObject *features_arr2   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        PyArrayObject *node_thr_arr    = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *margins_arr2    = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *lefts_arr2      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        PyArrayObject *rights_arr2     = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        PyArrayObject *values_arr2     = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *n_samples_arr2  = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *imps_arr2       = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *nan_left_arr2   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_BOOL);

        if (!features_arr2 || !node_thr_arr || !margins_arr2 ||
            !lefts_arr2    || !rights_arr2  || !values_arr2  ||
            !n_samples_arr2|| !imps_arr2    || !nan_left_arr2) {
            Py_XDECREF(features_arr2); Py_XDECREF(node_thr_arr);
            Py_XDECREF(margins_arr2);  Py_XDECREF(lefts_arr2);
            Py_XDECREF(rights_arr2);   Py_XDECREF(values_arr2);
            Py_XDECREF(n_samples_arr2);Py_XDECREF(imps_arr2);
            Py_XDECREF(nan_left_arr2);
            free(nodes);
            goto grow_fail;
        }

        npy_int64 *features_out  = (npy_int64 *)PyArray_DATA(features_arr2);
        double    *thr_out       = (double    *)PyArray_DATA(node_thr_arr);
        double    *margins_out   = (double    *)PyArray_DATA(margins_arr2);
        npy_int64 *lefts_out     = (npy_int64 *)PyArray_DATA(lefts_arr2);
        npy_int64 *rights_out    = (npy_int64 *)PyArray_DATA(rights_arr2);
        double    *values_out    = (double    *)PyArray_DATA(values_arr2);
        double    *samples_out   = (double    *)PyArray_DATA(n_samples_arr2);
        double    *imps_out      = (double    *)PyArray_DATA(imps_arr2);
        npy_bool  *nan_left_out  = (npy_bool  *)PyArray_DATA(nan_left_arr2);

        for (int i = 0; i < node_count; ++i) {
            features_out[i] = nodes[i].feature;
            thr_out[i]      = nodes[i].threshold;
            margins_out[i]  = nodes[i].margin;
            lefts_out[i]    = nodes[i].left;
            rights_out[i]   = nodes[i].right;
            values_out[i]   = nodes[i].value;
            samples_out[i]  = nodes[i].n_samples;
            imps_out[i]     = nodes[i].impurity_reduction;
            nan_left_out[i] = nodes[i].nan_go_left ? 1 : 0;
        }
        free(nodes);

        Py_DECREF(X_arr); Py_DECREF(X_bins_arr); Py_DECREF(y_arr);
        Py_DECREF(w_arr); Py_DECREF(thresholds_arr); Py_DECREF(centers_arr);
        Py_DECREF(n_thresholds_arr); Py_DECREF(feature_choices_arr);
        Py_DECREF(feature_counts_arr);

        return Py_BuildValue("NNNNNNNNN",
            (PyObject *)features_arr2, (PyObject *)node_thr_arr,
            (PyObject *)margins_arr2,  (PyObject *)lefts_arr2,
            (PyObject *)rights_arr2,   (PyObject *)values_arr2,
            (PyObject *)n_samples_arr2,(PyObject *)imps_arr2,
            (PyObject *)nan_left_arr2);
    }

grow_fail:
    Py_XDECREF(X_arr); Py_XDECREF(X_bins_arr); Py_XDECREF(y_arr);
    Py_XDECREF(w_arr); Py_XDECREF(thresholds_arr); Py_XDECREF(centers_arr);
    Py_XDECREF(n_thresholds_arr); Py_XDECREF(feature_choices_arr);
    Py_XDECREF(feature_counts_arr);
    return NULL;
}

static PyObject *
grow_lookahead_regression(PyObject *self, PyObject *args)
{
    PyObject *X_obj, *X_bins_obj, *y_obj, *w_obj;
    PyObject *Xv_obj, *yv_obj, *wv_obj;
    PyObject *thresholds_obj, *centers_obj, *n_thresholds_obj;
    int max_depth, margin_grid_size, include_hard_splits;
    int lookahead_horizon, lookahead_candidates;
    int margin_cv_folds, margin_cv_repeats;
    unsigned long cv_seed;
    double min_samples_leaf, margin_min_scale, margin_max_scale;
    double margin_depth_decay, min_train_weight_fraction, lookahead_min_val;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOididdpddiidiik",
                          &X_obj, &X_bins_obj, &y_obj, &w_obj,
                          &Xv_obj, &yv_obj, &wv_obj,
                          &thresholds_obj, &centers_obj, &n_thresholds_obj,
                          &max_depth, &min_samples_leaf, &margin_grid_size,
                          &margin_min_scale, &margin_max_scale,
                          &include_hard_splits, &margin_depth_decay,
                          &min_train_weight_fraction, &lookahead_horizon,
                          &lookahead_candidates, &lookahead_min_val,
                          &margin_cv_folds, &margin_cv_repeats, &cv_seed))
        return NULL;

    PyArrayObject *X_arr = (PyArrayObject *)PyArray_FROM_OTF(
        X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_bins_arr = (PyArrayObject *)PyArray_FROM_OTF(
        X_bins_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_arr = (PyArrayObject *)PyArray_FROM_OTF(
        y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Xv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Xv_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        yv_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *wv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        wv_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_arr = (PyArrayObject *)PyArray_FROM_OTF(
        thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *centers_arr = (PyArrayObject *)PyArray_FROM_OTF(
        centers_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *n_thresholds_arr = (PyArrayObject *)PyArray_FROM_OTF(
        n_thresholds_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!X_arr || !X_bins_arr || !y_arr || !w_arr || !Xv_arr ||
        !yv_arr || !wv_arr || !thresholds_arr || !centers_arr ||
        !n_thresholds_arr) {
        Py_XDECREF(X_arr); Py_XDECREF(X_bins_arr); Py_XDECREF(y_arr);
        Py_XDECREF(w_arr); Py_XDECREF(Xv_arr); Py_XDECREF(yv_arr);
        Py_XDECREF(wv_arr); Py_XDECREF(thresholds_arr);
        Py_XDECREF(centers_arr); Py_XDECREF(n_thresholds_arr);
        return NULL;
    }

    if (PyArray_NDIM(X_arr) != 2 || PyArray_NDIM(X_bins_arr) != 2 ||
        PyArray_NDIM(Xv_arr) != 2 ||
        PyArray_NDIM(thresholds_arr) != 2 ||
        PyArray_NDIM(centers_arr) != 2) {
        PyErr_SetString(PyExc_ValueError,
            "X, X_bins, X_val, thresholds, and centers must be 2D");
        goto la_grow_fail;
    }

    {
        npy_intp n_train = PyArray_DIM(X_arr, 0);
        npy_intp n_features = PyArray_DIM(X_arr, 1);
        npy_intp n_val = PyArray_DIM(Xv_arr, 0);

        if (PyArray_DIM(X_bins_arr, 0) != n_train ||
            PyArray_DIM(X_bins_arr, 1) != n_features ||
            PyArray_DIM(Xv_arr, 1) != n_features ||
            PyArray_SIZE(y_arr) != n_train ||
            PyArray_SIZE(w_arr) != n_train ||
            PyArray_SIZE(yv_arr) != n_val ||
            PyArray_SIZE(wv_arr) != n_val ||
            PyArray_DIM(thresholds_arr, 0) != n_features ||
            PyArray_DIM(centers_arr, 0) != n_features ||
            PyArray_SIZE(n_thresholds_arr) != n_features) {
            PyErr_SetString(PyExc_ValueError,
                            "input array dimensions do not match");
            goto la_grow_fail;
        }

        int max_thresholds = (int)PyArray_DIM(thresholds_arr, 1);
        if (PyArray_DIM(centers_arr, 1) != max_thresholds + 1) {
            PyErr_SetString(PyExc_ValueError,
                            "centers width must equal thresholds width + 1");
            goto la_grow_fail;
        }
        if (max_depth < 0 || max_depth > 20 ||
            lookahead_horizon < 1 || lookahead_horizon > 16 ||
            margin_grid_size < 1 || lookahead_candidates < 1 ||
            (margin_cv_folds != 0 && margin_cv_folds < 2) ||
            margin_cv_repeats < 1 ||
            n_features > (npy_intp)INT_MAX ||
            max_thresholds > INT_MAX - 2) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid lookahead grower parameter");
            goto la_grow_fail;
        }

        int max_nodes = (1 << (max_depth + 1)) - 1;
        int max_rollout_nodes = (1 << (lookahead_horizon + 1)) - 1;
        int n_dw = max_depth > 0 ? max_depth : 1;
        int n_roll = lookahead_horizon + 1;
        int B_stride = max_thresholds + 1;
        int max_margins = margin_grid_size + (include_hard_splits ? 1 : 0) + 1;

        CNode *nodes = (CNode *)calloc((size_t)max_nodes, sizeof(CNode));
        npy_intp *idx = (npy_intp *)malloc(
            (size_t)n_train * sizeof(npy_intp));
        LookaheadDepthWork *dw = (LookaheadDepthWork *)calloc(
            (size_t)n_dw, sizeof(LookaheadDepthWork));
        LookaheadRolloutWork *rollout_work = (LookaheadRolloutWork *)calloc(
            (size_t)n_roll, sizeof(LookaheadRolloutWork));

        double *margin_buf = (double *)malloc(
            (size_t)max_margins * sizeof(double));
        double *hist_buf = (double *)malloc(
            (size_t)n_features * 3 * (size_t)B_stride * sizeof(double));
        double *nan_buf = (double *)malloc(
            (size_t)n_features * 3 * sizeof(double));
        double *xstat_buf = (double *)malloc(
            (size_t)n_features * 3 * sizeof(double));
        int *int_scratch = (int *)malloc(
            (size_t)n_features * 5 * sizeof(int));
        LAThresholdCandidate *shortlist = (LAThresholdCandidate *)malloc(
            (size_t)lookahead_candidates * sizeof(LAThresholdCandidate));
        double *candidate_left_w = (double *)malloc(
            (size_t)n_train * sizeof(double));
        double *candidate_right_w = (double *)malloc(
            (size_t)n_train * sizeof(double));
        CNode *temp_nodes = (CNode *)calloc(
            (size_t)max_rollout_nodes, sizeof(CNode));
        npy_int64 *temp_stack_nodes = (npy_int64 *)malloc(
            (size_t)max_rollout_nodes * sizeof(npy_int64));
        double *temp_stack_weights = (double *)malloc(
            (size_t)max_rollout_nodes * sizeof(double));
        npy_intp *cv_perm = (npy_intp *)malloc(
            (size_t)n_train * sizeof(npy_intp));
        int *cv_fold_ids = (int *)malloc(
            (size_t)n_train * sizeof(int));
        double *cv_mu_buf = (double *)malloc(
            (size_t)max_margins * (size_t)n_train * sizeof(double));
        double *cv_losses = (double *)malloc(
            (size_t)max_margins * sizeof(double));
        unsigned char *cv_valid = (unsigned char *)malloc(
            (size_t)max_margins * sizeof(unsigned char));

        if (!nodes || !idx || !dw || !rollout_work || !margin_buf ||
            !hist_buf || !nan_buf || !xstat_buf || !int_scratch ||
            !shortlist || !candidate_left_w || !candidate_right_w ||
            !temp_nodes || !temp_stack_nodes || !temp_stack_weights ||
            !cv_perm || !cv_fold_ids || !cv_mu_buf ||
            !cv_losses || !cv_valid) {
            PyErr_NoMemory();
            free(nodes); free(idx); free(dw); free(rollout_work);
            free(margin_buf); free(hist_buf); free(nan_buf);
            free(xstat_buf); free(int_scratch); free(shortlist);
            free(candidate_left_w); free(candidate_right_w);
            free(temp_nodes); free(temp_stack_nodes); free(temp_stack_weights);
            free(cv_perm); free(cv_fold_ids); free(cv_mu_buf);
            free(cv_losses); free(cv_valid);
            goto la_grow_fail;
        }

        int alloc_ok = 1;
        for (int d = 0; d < n_dw && alloc_ok; ++d) {
            dw[d].left_idx = (npy_intp *)malloc(
                (size_t)n_train * sizeof(npy_intp));
            dw[d].right_idx = (npy_intp *)malloc(
                (size_t)n_train * sizeof(npy_intp));
            dw[d].left_w = (double *)malloc(
                (size_t)n_train * sizeof(double));
            dw[d].right_w = (double *)malloc(
                (size_t)n_train * sizeof(double));
            dw[d].val_left_w = (double *)malloc(
                (size_t)n_val * sizeof(double));
            dw[d].val_right_w = (double *)malloc(
                (size_t)n_val * sizeof(double));
            if (!dw[d].left_idx || !dw[d].right_idx ||
                !dw[d].left_w || !dw[d].right_w ||
                !dw[d].val_left_w || !dw[d].val_right_w)
                alloc_ok = 0;
        }
        for (int d = 0; d < n_roll && alloc_ok; ++d) {
            rollout_work[d].left_w = (double *)malloc(
                (size_t)n_train * sizeof(double));
            rollout_work[d].right_w = (double *)malloc(
                (size_t)n_train * sizeof(double));
            if (!rollout_work[d].left_w || !rollout_work[d].right_w)
                alloc_ok = 0;
        }
        if (!alloc_ok) {
            PyErr_NoMemory();
            for (int d = 0; d < n_dw; ++d) {
                free(dw[d].left_idx); free(dw[d].right_idx);
                free(dw[d].left_w); free(dw[d].right_w);
                free(dw[d].val_left_w); free(dw[d].val_right_w);
            }
            for (int d = 0; d < n_roll; ++d) {
                free(rollout_work[d].left_w);
                free(rollout_work[d].right_w);
            }
            free(nodes); free(idx); free(dw); free(rollout_work);
            free(margin_buf); free(hist_buf); free(nan_buf);
            free(xstat_buf); free(int_scratch); free(shortlist);
            free(candidate_left_w); free(candidate_right_w);
            free(temp_nodes); free(temp_stack_nodes); free(temp_stack_weights);
            free(cv_perm); free(cv_fold_ids); free(cv_mu_buf);
            free(cv_losses); free(cv_valid);
            goto la_grow_fail;
        }

        for (npy_intp i = 0; i < n_train; ++i) idx[i] = i;

        LookaheadContext ctx;
        ctx.X = (double *)PyArray_DATA(X_arr);
        ctx.X_bins = (int *)PyArray_DATA(X_bins_arr);
        ctx.y = (double *)PyArray_DATA(y_arr);
        ctx.Xv = (double *)PyArray_DATA(Xv_arr);
        ctx.yv = (double *)PyArray_DATA(yv_arr);
        ctx.n_features = n_features;
        ctx.n_val = n_val;
        ctx.thresholds_flat = (double *)PyArray_DATA(thresholds_arr);
        ctx.centers_flat = (double *)PyArray_DATA(centers_arr);
        ctx.n_thresholds = (int *)PyArray_DATA(n_thresholds_arr);
        ctx.max_thresholds = max_thresholds;
        ctx.margin_grid_size = margin_grid_size;
        ctx.include_hard_splits = include_hard_splits ? 1 : 0;
        ctx.margin_min_scale = margin_min_scale;
        ctx.margin_max_scale = margin_max_scale;
        ctx.margin_depth_decay = margin_depth_decay;
        ctx.min_samples_leaf = min_samples_leaf;
        ctx.min_train_weight_fraction = min_train_weight_fraction;
        ctx.lookahead_horizon = lookahead_horizon;
        ctx.lookahead_candidates = lookahead_candidates;
        ctx.lookahead_min_val = lookahead_min_val;
        ctx.margin_cv_folds = margin_cv_folds;
        ctx.margin_cv_repeats = margin_cv_repeats;
        ctx.cv_rng_state = ((uint64_t)cv_seed + 1ULL)
            ^ 0x9e3779b97f4a7c15ULL;
        ctx.prefix_buf = NULL;
        ctx.margin_buf = margin_buf;
        ctx.hist_buf = hist_buf;
        ctx.nan_buf = nan_buf;
        ctx.xstat_buf = xstat_buf;
        ctx.int_scratch = int_scratch;
        ctx.shortlist = shortlist;
        ctx.candidate_left_w = candidate_left_w;
        ctx.candidate_right_w = candidate_right_w;
        ctx.rollout_work = rollout_work;
        ctx.temp_nodes = temp_nodes;
        ctx.temp_stack_nodes = temp_stack_nodes;
        ctx.temp_stack_weights = temp_stack_weights;
        ctx.max_rollout_nodes = max_rollout_nodes;
        ctx.cv_perm = cv_perm;
        ctx.cv_fold_ids = cv_fold_ids;
        ctx.cv_mu_buf = cv_mu_buf;
        ctx.cv_losses = cv_losses;
        ctx.cv_valid = cv_valid;
        ctx.cv_class_left = NULL;
        ctx.cv_class_right = NULL;

        int node_count = 0;
        int status = 0;
        Py_BEGIN_ALLOW_THREADS
        status = build_lookahead_regression_numeric(
            &ctx, idx, (double *)PyArray_DATA(w_arr), n_train,
            (double *)PyArray_DATA(wv_arr), 0, max_depth, dw,
            nodes, &node_count, max_nodes);
        Py_END_ALLOW_THREADS

        for (int d = 0; d < n_dw; ++d) {
            free(dw[d].left_idx); free(dw[d].right_idx);
            free(dw[d].left_w); free(dw[d].right_w);
            free(dw[d].val_left_w); free(dw[d].val_right_w);
        }
        for (int d = 0; d < n_roll; ++d) {
            free(rollout_work[d].left_w);
            free(rollout_work[d].right_w);
        }
        free(idx); free(dw); free(rollout_work);
        free(margin_buf); free(hist_buf); free(nan_buf);
        free(xstat_buf); free(int_scratch); free(shortlist);
        free(candidate_left_w); free(candidate_right_w);
        free(temp_nodes); free(temp_stack_nodes); free(temp_stack_weights);
        free(cv_perm); free(cv_fold_ids); free(cv_mu_buf);
        free(cv_losses); free(cv_valid);

        if (status < 0) {
            free(nodes);
            PyErr_SetString(PyExc_RuntimeError,
                            "C lookahead tree grower failed");
            goto la_grow_fail;
        }

        npy_intp dims[1] = {node_count};
        PyArrayObject *features_arr2   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        PyArrayObject *node_thr_arr    = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *margins_arr2    = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *lefts_arr2      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        PyArrayObject *rights_arr2     = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        PyArrayObject *values_arr2     = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *n_samples_arr2  = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *imps_arr2       = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *nan_left_arr2   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_BOOL);

        if (!features_arr2 || !node_thr_arr || !margins_arr2 ||
            !lefts_arr2    || !rights_arr2  || !values_arr2  ||
            !n_samples_arr2|| !imps_arr2    || !nan_left_arr2) {
            Py_XDECREF(features_arr2); Py_XDECREF(node_thr_arr);
            Py_XDECREF(margins_arr2);  Py_XDECREF(lefts_arr2);
            Py_XDECREF(rights_arr2);   Py_XDECREF(values_arr2);
            Py_XDECREF(n_samples_arr2);Py_XDECREF(imps_arr2);
            Py_XDECREF(nan_left_arr2);
            free(nodes);
            goto la_grow_fail;
        }

        npy_int64 *features_out  = (npy_int64 *)PyArray_DATA(features_arr2);
        double    *thr_out       = (double    *)PyArray_DATA(node_thr_arr);
        double    *margins_out   = (double    *)PyArray_DATA(margins_arr2);
        npy_int64 *lefts_out     = (npy_int64 *)PyArray_DATA(lefts_arr2);
        npy_int64 *rights_out    = (npy_int64 *)PyArray_DATA(rights_arr2);
        double    *values_out    = (double    *)PyArray_DATA(values_arr2);
        double    *samples_out   = (double    *)PyArray_DATA(n_samples_arr2);
        double    *imps_out      = (double    *)PyArray_DATA(imps_arr2);
        npy_bool  *nan_left_out  = (npy_bool  *)PyArray_DATA(nan_left_arr2);

        for (int i = 0; i < node_count; ++i) {
            features_out[i] = nodes[i].feature;
            thr_out[i]      = nodes[i].threshold;
            margins_out[i]  = nodes[i].margin;
            lefts_out[i]    = nodes[i].left;
            rights_out[i]   = nodes[i].right;
            values_out[i]   = nodes[i].value;
            samples_out[i]  = nodes[i].n_samples;
            imps_out[i]     = nodes[i].impurity_reduction;
            nan_left_out[i] = nodes[i].nan_go_left ? 1 : 0;
        }
        free(nodes);

        Py_DECREF(X_arr); Py_DECREF(X_bins_arr); Py_DECREF(y_arr);
        Py_DECREF(w_arr); Py_DECREF(Xv_arr); Py_DECREF(yv_arr);
        Py_DECREF(wv_arr); Py_DECREF(thresholds_arr);
        Py_DECREF(centers_arr); Py_DECREF(n_thresholds_arr);

        return Py_BuildValue("NNNNNNNNN",
            (PyObject *)features_arr2, (PyObject *)node_thr_arr,
            (PyObject *)margins_arr2,  (PyObject *)lefts_arr2,
            (PyObject *)rights_arr2,   (PyObject *)values_arr2,
            (PyObject *)n_samples_arr2,(PyObject *)imps_arr2,
            (PyObject *)nan_left_arr2);
    }

la_grow_fail:
    Py_XDECREF(X_arr); Py_XDECREF(X_bins_arr); Py_XDECREF(y_arr);
    Py_XDECREF(w_arr); Py_XDECREF(Xv_arr); Py_XDECREF(yv_arr);
    Py_XDECREF(wv_arr); Py_XDECREF(thresholds_arr);
    Py_XDECREF(centers_arr); Py_XDECREF(n_thresholds_arr);
    return NULL;
}

static PyObject *
grow_lookahead_classifier(PyObject *self, PyObject *args)
{
    PyObject *X_obj, *X_bins_obj, *y_obj, *w_obj;
    PyObject *Xv_obj, *yv_obj, *wv_obj;
    PyObject *thresholds_obj, *centers_obj, *n_thresholds_obj;
    int max_depth, margin_grid_size, include_hard_splits;
    int lookahead_horizon, lookahead_candidates;
    int margin_cv_folds, margin_cv_repeats;
    int n_classes, split_criterion;
    unsigned long cv_seed;
    double min_samples_leaf, margin_min_scale, margin_max_scale;
    double margin_depth_decay, min_train_weight_fraction, lookahead_min_val;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOididdpddiidiikii",
                          &X_obj, &X_bins_obj, &y_obj, &w_obj,
                          &Xv_obj, &yv_obj, &wv_obj,
                          &thresholds_obj, &centers_obj, &n_thresholds_obj,
                          &max_depth, &min_samples_leaf, &margin_grid_size,
                          &margin_min_scale, &margin_max_scale,
                          &include_hard_splits, &margin_depth_decay,
                          &min_train_weight_fraction, &lookahead_horizon,
                          &lookahead_candidates, &lookahead_min_val,
                          &margin_cv_folds, &margin_cv_repeats, &cv_seed,
                          &n_classes, &split_criterion))
        return NULL;

    PyArrayObject *X_arr = (PyArrayObject *)PyArray_FROM_OTF(
        X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_bins_arr = (PyArrayObject *)PyArray_FROM_OTF(
        X_bins_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_arr = (PyArrayObject *)PyArray_FROM_OTF(
        y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Xv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        Xv_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        yv_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *wv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        wv_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_arr = (PyArrayObject *)PyArray_FROM_OTF(
        thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *centers_arr = (PyArrayObject *)PyArray_FROM_OTF(
        centers_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *n_thresholds_arr = (PyArrayObject *)PyArray_FROM_OTF(
        n_thresholds_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!X_arr || !X_bins_arr || !y_arr || !w_arr || !Xv_arr ||
        !yv_arr || !wv_arr || !thresholds_arr || !centers_arr ||
        !n_thresholds_arr) {
        Py_XDECREF(X_arr); Py_XDECREF(X_bins_arr); Py_XDECREF(y_arr);
        Py_XDECREF(w_arr); Py_XDECREF(Xv_arr); Py_XDECREF(yv_arr);
        Py_XDECREF(wv_arr); Py_XDECREF(thresholds_arr);
        Py_XDECREF(centers_arr); Py_XDECREF(n_thresholds_arr);
        return NULL;
    }

    if (PyArray_NDIM(X_arr) != 2 || PyArray_NDIM(X_bins_arr) != 2 ||
        PyArray_NDIM(Xv_arr) != 2 ||
        PyArray_NDIM(thresholds_arr) != 2 ||
        PyArray_NDIM(centers_arr) != 2) {
        PyErr_SetString(PyExc_ValueError,
            "X, X_bins, X_val, thresholds, and centers must be 2D");
        goto cla_grow_fail;
    }

    {
        npy_intp n_train = PyArray_DIM(X_arr, 0);
        npy_intp n_features = PyArray_DIM(X_arr, 1);
        npy_intp n_val = PyArray_DIM(Xv_arr, 0);

        if (PyArray_DIM(X_bins_arr, 0) != n_train ||
            PyArray_DIM(X_bins_arr, 1) != n_features ||
            PyArray_DIM(Xv_arr, 1) != n_features ||
            PyArray_SIZE(y_arr) != n_train ||
            PyArray_SIZE(w_arr) != n_train ||
            PyArray_SIZE(yv_arr) != n_val ||
            PyArray_SIZE(wv_arr) != n_val ||
            PyArray_DIM(thresholds_arr, 0) != n_features ||
            PyArray_DIM(centers_arr, 0) != n_features ||
            PyArray_SIZE(n_thresholds_arr) != n_features) {
            PyErr_SetString(PyExc_ValueError,
                            "input array dimensions do not match");
            goto cla_grow_fail;
        }

        int max_thresholds = (int)PyArray_DIM(thresholds_arr, 1);
        if (PyArray_DIM(centers_arr, 1) != max_thresholds + 1) {
            PyErr_SetString(PyExc_ValueError,
                            "centers width must equal thresholds width + 1");
            goto cla_grow_fail;
        }
        if (max_depth < 0 || max_depth > 20 ||
            lookahead_horizon < 1 || lookahead_horizon > 16 ||
            margin_grid_size < 1 || lookahead_candidates < 1 ||
            (margin_cv_folds != 0 && margin_cv_folds < 2) ||
            margin_cv_repeats < 1 ||
            n_features > (npy_intp)INT_MAX ||
            max_thresholds > INT_MAX - 2 ||
            n_classes < 2 || n_classes > 4096 ||
            (split_criterion != 0 && split_criterion != 1)) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid classifier lookahead grower parameter");
            goto cla_grow_fail;
        }

        int max_nodes = (1 << (max_depth + 1)) - 1;
        int max_rollout_nodes = (1 << (lookahead_horizon + 1)) - 1;
        int n_dw = max_depth > 0 ? max_depth : 1;
        int n_roll = lookahead_horizon + 1;
        int B_stride = max_thresholds + 1;
        int max_margins = margin_grid_size + (include_hard_splits ? 1 : 0) + 1;

        CNode *nodes = (CNode *)calloc((size_t)max_nodes, sizeof(CNode));
        npy_intp *idx = (npy_intp *)malloc(
            (size_t)n_train * sizeof(npy_intp));
        LookaheadDepthWork *dw = (LookaheadDepthWork *)calloc(
            (size_t)n_dw, sizeof(LookaheadDepthWork));
        LookaheadRolloutWork *rollout_work = (LookaheadRolloutWork *)calloc(
            (size_t)n_roll, sizeof(LookaheadRolloutWork));

        double *margin_buf = (double *)malloc(
            (size_t)max_margins * sizeof(double));
        double *class_hist_buf = (double *)malloc(
            (size_t)n_features * (size_t)B_stride * (size_t)n_classes
            * sizeof(double));
        double *nan_class_buf = (double *)malloc(
            (size_t)n_features * (size_t)n_classes * sizeof(double));
        double *xstat_buf = (double *)malloc(
            (size_t)n_features * 3 * sizeof(double));
        double *class_left_buf = (double *)malloc(
            (size_t)n_classes * sizeof(double));
        double *class_total_buf = (double *)malloc(
            (size_t)n_classes * sizeof(double));
        int *int_scratch = (int *)malloc(
            (size_t)n_features * 5 * sizeof(int));
        LAThresholdCandidate *shortlist = (LAThresholdCandidate *)malloc(
            (size_t)lookahead_candidates * sizeof(LAThresholdCandidate));
        double *candidate_left_w = (double *)malloc(
            (size_t)n_train * sizeof(double));
        double *candidate_right_w = (double *)malloc(
            (size_t)n_train * sizeof(double));
        CNode *temp_nodes = (CNode *)calloc(
            (size_t)max_rollout_nodes, sizeof(CNode));
        double *temp_leaf_probs = (double *)calloc(
            (size_t)max_rollout_nodes * (size_t)n_classes,
            sizeof(double));
        double *temp_pred_probs = (double *)malloc(
            (size_t)n_classes * sizeof(double));
        npy_int64 *temp_stack_nodes = (npy_int64 *)malloc(
            (size_t)max_rollout_nodes * sizeof(npy_int64));
        double *temp_stack_weights = (double *)malloc(
            (size_t)max_rollout_nodes * sizeof(double));
        npy_intp *cv_perm = (npy_intp *)malloc(
            (size_t)n_train * sizeof(npy_intp));
        int *cv_fold_ids = (int *)malloc(
            (size_t)n_train * sizeof(int));
        double *cv_mu_buf = (double *)malloc(
            (size_t)max_margins * (size_t)n_train * sizeof(double));
        double *cv_losses = (double *)malloc(
            (size_t)max_margins * sizeof(double));
        unsigned char *cv_valid = (unsigned char *)malloc(
            (size_t)max_margins * sizeof(unsigned char));
        double *cv_class_left = (double *)malloc(
            (size_t)n_classes * sizeof(double));
        double *cv_class_right = (double *)malloc(
            (size_t)n_classes * sizeof(double));

        if (!nodes || !idx || !dw || !rollout_work || !margin_buf ||
            !class_hist_buf || !nan_class_buf || !xstat_buf ||
            !class_left_buf || !class_total_buf || !int_scratch ||
            !shortlist || !candidate_left_w || !candidate_right_w ||
            !temp_nodes || !temp_leaf_probs || !temp_pred_probs ||
            !temp_stack_nodes || !temp_stack_weights ||
            !cv_perm || !cv_fold_ids || !cv_mu_buf || !cv_losses ||
            !cv_valid || !cv_class_left || !cv_class_right) {
            PyErr_NoMemory();
            free(nodes); free(idx); free(dw); free(rollout_work);
            free(margin_buf); free(class_hist_buf); free(nan_class_buf);
            free(xstat_buf); free(class_left_buf); free(class_total_buf);
            free(int_scratch); free(shortlist); free(candidate_left_w);
            free(candidate_right_w); free(temp_nodes);
            free(temp_leaf_probs); free(temp_pred_probs);
            free(temp_stack_nodes); free(temp_stack_weights);
            free(cv_perm); free(cv_fold_ids); free(cv_mu_buf);
            free(cv_losses); free(cv_valid);
            free(cv_class_left); free(cv_class_right);
            goto cla_grow_fail;
        }

        int alloc_ok = 1;
        for (int d = 0; d < n_dw && alloc_ok; ++d) {
            dw[d].left_idx = (npy_intp *)malloc(
                (size_t)n_train * sizeof(npy_intp));
            dw[d].right_idx = (npy_intp *)malloc(
                (size_t)n_train * sizeof(npy_intp));
            dw[d].left_w = (double *)malloc(
                (size_t)n_train * sizeof(double));
            dw[d].right_w = (double *)malloc(
                (size_t)n_train * sizeof(double));
            dw[d].val_left_w = (double *)malloc(
                (size_t)n_val * sizeof(double));
            dw[d].val_right_w = (double *)malloc(
                (size_t)n_val * sizeof(double));
            if (!dw[d].left_idx || !dw[d].right_idx ||
                !dw[d].left_w || !dw[d].right_w ||
                !dw[d].val_left_w || !dw[d].val_right_w)
                alloc_ok = 0;
        }
        for (int d = 0; d < n_roll && alloc_ok; ++d) {
            rollout_work[d].left_w = (double *)malloc(
                (size_t)n_train * sizeof(double));
            rollout_work[d].right_w = (double *)malloc(
                (size_t)n_train * sizeof(double));
            if (!rollout_work[d].left_w || !rollout_work[d].right_w)
                alloc_ok = 0;
        }
        if (!alloc_ok) {
            PyErr_NoMemory();
            for (int d = 0; d < n_dw; ++d) {
                free(dw[d].left_idx); free(dw[d].right_idx);
                free(dw[d].left_w); free(dw[d].right_w);
                free(dw[d].val_left_w); free(dw[d].val_right_w);
            }
            for (int d = 0; d < n_roll; ++d) {
                free(rollout_work[d].left_w);
                free(rollout_work[d].right_w);
            }
            free(nodes); free(idx); free(dw); free(rollout_work);
            free(margin_buf); free(class_hist_buf); free(nan_class_buf);
            free(xstat_buf); free(class_left_buf); free(class_total_buf);
            free(int_scratch); free(shortlist); free(candidate_left_w);
            free(candidate_right_w); free(temp_nodes);
            free(temp_leaf_probs); free(temp_pred_probs);
            free(temp_stack_nodes); free(temp_stack_weights);
            free(cv_perm); free(cv_fold_ids); free(cv_mu_buf);
            free(cv_losses); free(cv_valid);
            free(cv_class_left); free(cv_class_right);
            goto cla_grow_fail;
        }

        for (npy_intp i = 0; i < n_train; ++i) idx[i] = i;

        LookaheadContext ctx;
        ctx.X = (double *)PyArray_DATA(X_arr);
        ctx.X_bins = (int *)PyArray_DATA(X_bins_arr);
        ctx.y = (double *)PyArray_DATA(y_arr);
        ctx.Xv = (double *)PyArray_DATA(Xv_arr);
        ctx.yv = (double *)PyArray_DATA(yv_arr);
        ctx.n_features = n_features;
        ctx.n_val = n_val;
        ctx.thresholds_flat = (double *)PyArray_DATA(thresholds_arr);
        ctx.centers_flat = (double *)PyArray_DATA(centers_arr);
        ctx.n_thresholds = (int *)PyArray_DATA(n_thresholds_arr);
        ctx.max_thresholds = max_thresholds;
        ctx.margin_grid_size = margin_grid_size;
        ctx.include_hard_splits = include_hard_splits ? 1 : 0;
        ctx.margin_min_scale = margin_min_scale;
        ctx.margin_max_scale = margin_max_scale;
        ctx.margin_depth_decay = margin_depth_decay;
        ctx.min_samples_leaf = min_samples_leaf;
        ctx.min_train_weight_fraction = min_train_weight_fraction;
        ctx.lookahead_horizon = lookahead_horizon;
        ctx.lookahead_candidates = lookahead_candidates;
        ctx.lookahead_min_val = lookahead_min_val;
        ctx.margin_cv_folds = margin_cv_folds;
        ctx.margin_cv_repeats = margin_cv_repeats;
        ctx.cv_rng_state = ((uint64_t)cv_seed + 1ULL)
            ^ 0x9e3779b97f4a7c15ULL;
        ctx.n_classes = n_classes;
        ctx.split_criterion = split_criterion;
        ctx.prefix_buf = NULL;
        ctx.margin_buf = margin_buf;
        ctx.hist_buf = NULL;
        ctx.nan_buf = NULL;
        ctx.xstat_buf = xstat_buf;
        ctx.class_hist_buf = class_hist_buf;
        ctx.nan_class_buf = nan_class_buf;
        ctx.class_left_buf = class_left_buf;
        ctx.class_total_buf = class_total_buf;
        ctx.int_scratch = int_scratch;
        ctx.shortlist = shortlist;
        ctx.candidate_left_w = candidate_left_w;
        ctx.candidate_right_w = candidate_right_w;
        ctx.rollout_work = rollout_work;
        ctx.temp_nodes = temp_nodes;
        ctx.temp_leaf_probs = temp_leaf_probs;
        ctx.temp_pred_probs = temp_pred_probs;
        ctx.temp_stack_nodes = temp_stack_nodes;
        ctx.temp_stack_weights = temp_stack_weights;
        ctx.max_rollout_nodes = max_rollout_nodes;
        ctx.cv_perm = cv_perm;
        ctx.cv_fold_ids = cv_fold_ids;
        ctx.cv_mu_buf = cv_mu_buf;
        ctx.cv_losses = cv_losses;
        ctx.cv_valid = cv_valid;
        ctx.cv_class_left = cv_class_left;
        ctx.cv_class_right = cv_class_right;

        int node_count = 0;
        int status = 0;
        Py_BEGIN_ALLOW_THREADS
        status = build_lookahead_classifier_numeric(
            &ctx, idx, (double *)PyArray_DATA(w_arr), n_train,
            (double *)PyArray_DATA(wv_arr), 0, max_depth, dw,
            nodes, &node_count, max_nodes);
        Py_END_ALLOW_THREADS

        for (int d = 0; d < n_dw; ++d) {
            free(dw[d].left_idx); free(dw[d].right_idx);
            free(dw[d].left_w); free(dw[d].right_w);
            free(dw[d].val_left_w); free(dw[d].val_right_w);
        }
        for (int d = 0; d < n_roll; ++d) {
            free(rollout_work[d].left_w);
            free(rollout_work[d].right_w);
        }
        free(idx); free(dw); free(rollout_work);
        free(margin_buf); free(class_hist_buf); free(nan_class_buf);
        free(xstat_buf); free(class_left_buf); free(class_total_buf);
        free(int_scratch); free(shortlist); free(candidate_left_w);
        free(candidate_right_w); free(temp_nodes);
        free(temp_leaf_probs); free(temp_pred_probs);
        free(temp_stack_nodes); free(temp_stack_weights);
        free(cv_perm); free(cv_fold_ids); free(cv_mu_buf);
        free(cv_losses); free(cv_valid);
        free(cv_class_left); free(cv_class_right);

        if (status < 0) {
            free(nodes);
            PyErr_SetString(PyExc_RuntimeError,
                            "C classifier lookahead tree grower failed");
            goto cla_grow_fail;
        }

        npy_intp dims[1] = {node_count};
        PyArrayObject *features_arr2   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        PyArrayObject *node_thr_arr    = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *margins_arr2    = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *lefts_arr2      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        PyArrayObject *rights_arr2     = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        PyArrayObject *values_arr2     = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *n_samples_arr2  = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *imps_arr2       = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyArrayObject *nan_left_arr2   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_BOOL);

        if (!features_arr2 || !node_thr_arr || !margins_arr2 ||
            !lefts_arr2    || !rights_arr2  || !values_arr2  ||
            !n_samples_arr2|| !imps_arr2    || !nan_left_arr2) {
            Py_XDECREF(features_arr2); Py_XDECREF(node_thr_arr);
            Py_XDECREF(margins_arr2);  Py_XDECREF(lefts_arr2);
            Py_XDECREF(rights_arr2);   Py_XDECREF(values_arr2);
            Py_XDECREF(n_samples_arr2);Py_XDECREF(imps_arr2);
            Py_XDECREF(nan_left_arr2);
            free(nodes);
            goto cla_grow_fail;
        }

        npy_int64 *features_out  = (npy_int64 *)PyArray_DATA(features_arr2);
        double    *thr_out       = (double    *)PyArray_DATA(node_thr_arr);
        double    *margins_out   = (double    *)PyArray_DATA(margins_arr2);
        npy_int64 *lefts_out     = (npy_int64 *)PyArray_DATA(lefts_arr2);
        npy_int64 *rights_out    = (npy_int64 *)PyArray_DATA(rights_arr2);
        double    *values_out    = (double    *)PyArray_DATA(values_arr2);
        double    *samples_out   = (double    *)PyArray_DATA(n_samples_arr2);
        double    *imps_out      = (double    *)PyArray_DATA(imps_arr2);
        npy_bool  *nan_left_out  = (npy_bool  *)PyArray_DATA(nan_left_arr2);

        for (int i = 0; i < node_count; ++i) {
            features_out[i] = nodes[i].feature;
            thr_out[i]      = nodes[i].threshold;
            margins_out[i]  = nodes[i].margin;
            lefts_out[i]    = nodes[i].left;
            rights_out[i]   = nodes[i].right;
            values_out[i]   = nodes[i].value;
            samples_out[i]  = nodes[i].n_samples;
            imps_out[i]     = nodes[i].impurity_reduction;
            nan_left_out[i] = nodes[i].nan_go_left ? 1 : 0;
        }
        free(nodes);

        Py_DECREF(X_arr); Py_DECREF(X_bins_arr); Py_DECREF(y_arr);
        Py_DECREF(w_arr); Py_DECREF(Xv_arr); Py_DECREF(yv_arr);
        Py_DECREF(wv_arr); Py_DECREF(thresholds_arr);
        Py_DECREF(centers_arr); Py_DECREF(n_thresholds_arr);

        return Py_BuildValue("NNNNNNNNN",
            (PyObject *)features_arr2, (PyObject *)node_thr_arr,
            (PyObject *)margins_arr2,  (PyObject *)lefts_arr2,
            (PyObject *)rights_arr2,   (PyObject *)values_arr2,
            (PyObject *)n_samples_arr2,(PyObject *)imps_arr2,
            (PyObject *)nan_left_arr2);
    }

cla_grow_fail:
    Py_XDECREF(X_arr); Py_XDECREF(X_bins_arr); Py_XDECREF(y_arr);
    Py_XDECREF(w_arr); Py_XDECREF(Xv_arr); Py_XDECREF(yv_arr);
    Py_XDECREF(wv_arr); Py_XDECREF(thresholds_arr);
    Py_XDECREF(centers_arr); Py_XDECREF(n_thresholds_arr);
    return NULL;
}

/*
 * Diagonal leaf-stat accumulation: traverses each sample through the flat
 * numeric tree, accumulating per-leaf numerator and denominator for the
 * diagonal approximation of the weighted leaf-value optimisation.
 *
 *   numerator[k]   += w_i * path_weight * y_i
 *   denominator[k] += w_i * path_weight^2
 *
 * Caller computes: v_k = numerator[k] / (denominator[k] + l2)
 *
 * This is O(N * effective_depth) and never materialises the (N x K) leaf-
 * weight matrix, making it practical for deep trees (depth 8-10, K ~ 100-1024).
 *
 * Input:  X (N,F), features, thresholds, margins, lefts, rights, nan_go_left,
 *         y (N,) — effective targets already shifted by prior,
 *         w (N,)
 * Output: (numerator (K,), denominator (K,))   K = number of leaves
 */
static PyObject *
accumulate_leaf_stats(PyObject *self, PyObject *args)
{
    PyObject *X_obj, *features_obj, *thresholds_obj, *margins_obj;
    PyObject *lefts_obj, *rights_obj, *nan_go_left_obj;
    PyObject *y_obj, *w_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOOOO",
                          &X_obj, &features_obj, &thresholds_obj, &margins_obj,
                          &lefts_obj, &rights_obj, &nan_go_left_obj,
                          &y_obj, &w_obj))
        return NULL;

    PyArrayObject *X_arr = (PyArrayObject *)PyArray_FROM_OTF(
        X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_arr = (PyArrayObject *)PyArray_FROM_OTF(
        features_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_arr = (PyArrayObject *)PyArray_FROM_OTF(
        thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *margins_arr = (PyArrayObject *)PyArray_FROM_OTF(
        margins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *lefts_arr = (PyArrayObject *)PyArray_FROM_OTF(
        lefts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *rights_arr = (PyArrayObject *)PyArray_FROM_OTF(
        rights_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *nan_go_left_arr = (PyArrayObject *)PyArray_FROM_OTF(
        nan_go_left_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_arr = (PyArrayObject *)PyArray_FROM_OTF(
        y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *w_arr = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!X_arr || !features_arr || !thresholds_arr || !margins_arr ||
        !lefts_arr || !rights_arr || !nan_go_left_arr || !y_arr || !w_arr) {
        Py_XDECREF(X_arr); Py_XDECREF(features_arr); Py_XDECREF(thresholds_arr);
        Py_XDECREF(margins_arr); Py_XDECREF(lefts_arr); Py_XDECREF(rights_arr);
        Py_XDECREF(nan_go_left_arr); Py_XDECREF(y_arr); Py_XDECREF(w_arr);
        return NULL;
    }

    if (PyArray_NDIM(X_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "X must be 2-D");
        goto als_fail;
    }

    {
        npy_intp n_samples  = PyArray_DIM(X_arr, 0);
        npy_intp n_features = PyArray_DIM(X_arr, 1);
        npy_intp n_nodes    = PyArray_SIZE(lefts_arr);

        if (PyArray_SIZE(y_arr) != n_samples || PyArray_SIZE(w_arr) != n_samples ||
            PyArray_SIZE(features_arr) != n_nodes ||
            PyArray_SIZE(thresholds_arr) != n_nodes ||
            PyArray_SIZE(margins_arr)   != n_nodes ||
            PyArray_SIZE(rights_arr)    != n_nodes ||
            PyArray_SIZE(nan_go_left_arr) != n_nodes) {
            PyErr_SetString(PyExc_ValueError, "array size mismatch");
            goto als_fail;
        }

        double    *X          = (double    *)PyArray_DATA(X_arr);
        npy_int64 *features   = (npy_int64 *)PyArray_DATA(features_arr);
        double    *thresholds = (double    *)PyArray_DATA(thresholds_arr);
        double    *margins    = (double    *)PyArray_DATA(margins_arr);
        npy_int64 *lefts      = (npy_int64 *)PyArray_DATA(lefts_arr);
        npy_int64 *rights     = (npy_int64 *)PyArray_DATA(rights_arr);
        npy_bool  *nan_go_left= (npy_bool  *)PyArray_DATA(nan_go_left_arr);
        double    *y          = (double    *)PyArray_DATA(y_arr);
        double    *w          = (double    *)PyArray_DATA(w_arr);

        /* Map node index → leaf column index (-1 for internal nodes). */
        int *leaf_col = (int *)malloc((size_t)n_nodes * sizeof(int));
        if (!leaf_col) { PyErr_NoMemory(); goto als_fail; }

        int n_leaves = 0;
        for (npy_intp ni = 0; ni < n_nodes; ++ni) {
            if (lefts[ni] < 0) leaf_col[ni] = n_leaves++;
            else                leaf_col[ni] = -1;
        }

        npy_intp out_dims[1] = {n_leaves};
        PyArrayObject *num_arr  = (PyArrayObject *)PyArray_ZEROS(
            1, out_dims, NPY_DOUBLE, 0);
        PyArrayObject *den_arr  = (PyArrayObject *)PyArray_ZEROS(
            1, out_dims, NPY_DOUBLE, 0);

        if (!num_arr || !den_arr) {
            free(leaf_col);
            Py_XDECREF(num_arr); Py_XDECREF(den_arr);
            PyErr_NoMemory();
            goto als_fail;
        }

        double *numerator   = (double *)PyArray_DATA(num_arr);
        double *denominator = (double *)PyArray_DATA(den_arr);

        /* Per-sample DFS stack (stack depth bounded by n_nodes). */
        npy_int64 *stk_nodes   = (npy_int64 *)malloc(
            (size_t)n_nodes * sizeof(npy_int64));
        double    *stk_weights  = (double    *)malloc(
            (size_t)n_nodes * sizeof(double));
        if (!stk_nodes || !stk_weights) {
            free(leaf_col); free(stk_nodes); free(stk_weights);
            Py_DECREF(num_arr); Py_DECREF(den_arr);
            PyErr_NoMemory();
            goto als_fail;
        }

        Py_BEGIN_ALLOW_THREADS

        for (npy_intp i = 0; i < n_samples; ++i) {
            double wi = w[i], yi = y[i];

            npy_intp top = 0;
            stk_nodes[top]   = 0;
            stk_weights[top] = 1.0;
            ++top;

            while (top > 0) {
                --top;
                npy_int64 nidx   = stk_nodes[top];
                double    pw     = stk_weights[top];

                if (lefts[nidx] < 0) {
                    int col = leaf_col[nidx];
                    numerator[col]   += wi * pw * yi;
                    denominator[col] += wi * pw * pw;
                    continue;
                }

                double x = X[i * n_features + features[nidx]];
                double mu_l;
                if (isnan(x)) {
                    mu_l = nan_go_left[nidx] ? 1.0 : 0.0;
                } else if (margins[nidx] < 1e-12) {
                    mu_l = x <= thresholds[nidx] ? 1.0 : 0.0;
                } else {
                    mu_l = membership_value(
                        (x - thresholds[nidx]) / margins[nidx]);
                }

                if (mu_l > 0.0) {
                    stk_nodes[top]   = lefts[nidx];
                    stk_weights[top] = pw * mu_l;
                    ++top;
                }
                if (mu_l < 1.0) {
                    stk_nodes[top]   = rights[nidx];
                    stk_weights[top] = pw * (1.0 - mu_l);
                    ++top;
                }
            }
        }

        Py_END_ALLOW_THREADS

        free(leaf_col); free(stk_nodes); free(stk_weights);
        Py_DECREF(X_arr); Py_DECREF(features_arr); Py_DECREF(thresholds_arr);
        Py_DECREF(margins_arr); Py_DECREF(lefts_arr); Py_DECREF(rights_arr);
        Py_DECREF(nan_go_left_arr); Py_DECREF(y_arr); Py_DECREF(w_arr);
        return Py_BuildValue("NN", (PyObject *)num_arr, (PyObject *)den_arr);
    }

als_fail:
    Py_XDECREF(X_arr); Py_XDECREF(features_arr); Py_XDECREF(thresholds_arr);
    Py_XDECREF(margins_arr); Py_XDECREF(lefts_arr); Py_XDECREF(rights_arr);
    Py_XDECREF(nan_go_left_arr); Py_XDECREF(y_arr); Py_XDECREF(w_arr);
    return NULL;
}

static PyMethodDef methods[] = {
    {"find_mse_split",             find_mse_split,             METH_VARARGS,
     "Find the best fuzzy MSE split for one numeric feature."},
    {"predict_numeric_tree",       predict_numeric_tree,       METH_VARARGS,
     "Predict a numeric fuzzy tree stored as flat arrays."},
    {"grow_depth_first_regression",grow_depth_first_regression,METH_VARARGS,
     "Grow a full numeric regression tree with depth-first fuzzy splits."},
    {"grow_lookahead_regression",  grow_lookahead_regression,  METH_VARARGS,
     "Grow a numeric regression tree with C-accelerated lookahead splits."},
    {"grow_lookahead_classifier",  grow_lookahead_classifier,  METH_VARARGS,
     "Grow a numeric classifier tree with C-accelerated lookahead splits."},
    {"accumulate_leaf_stats",      accumulate_leaf_stats,      METH_VARARGS,
     "Diagonal leaf-stat accumulation for fast leaf-value optimisation."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_c_backend",
    "Compiled fuzzydecisiontree kernels.", -1, methods
};

PyMODINIT_FUNC
PyInit__c_backend(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    if (!module) return NULL;
    import_array();
    return module;
}
