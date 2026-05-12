import argparse
import json
import math
import itertools
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_RESULTS_DIR = "auto_prompt/preds/Qwen_Qwen2.5-72B-Instruct-AWQ/paper5"
DATASETS = ["acne", "odd", "sdoh", "clip"]
EXCLUDE_FEATURES = {"cancer_family_any"}
ANALYSIS_DIR = "auto_prompt/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Fixed style per optimizer so colors line up across per-dataset and combined figures.
_cmap = plt.colormaps["tab10"]
_markers = ["o", "s", "P", "D", "^", "v", "X", "*", "p", "h"]
ALL_OPTIMIZERS = ["opro", "ours", "etgpo", "ampo", "dspy"]
method_styles = {
    opt: {"color": _cmap(i), "marker": _markers[i]}
    for i, opt in enumerate(ALL_OPTIMIZERS)
}


def _norm(v):
    """Normalize accuracy to 0-1 scale."""
    return v / 100.0 if v > 1.0 else v


def _ph_train(h):
    """Per-iter train accuracy. Same key in old and new format."""
    return h.get("train_accuracy")


def _ph_val(h):
    """Per-iter val accuracy (new) or eval accuracy (old)."""
    v = h.get("val_accuracy")
    return v if v is not None else h.get("eval_accuracy")


def _ph_cand_train(h):
    """Candidate train accuracy. New: candidate_train_accuracy. Old: candidate_accuracy."""
    v = h.get("candidate_train_accuracy")
    return v if v is not None else h.get("candidate_accuracy")


def _ph_cand_val(h):
    """Candidate val accuracy (new) or candidate eval accuracy (old)."""
    v = h.get("candidate_val_accuracy")
    return v if v is not None else h.get("candidate_eval_accuracy")


def _feat_test_combined(fd):
    """Final test accuracy (new) or eval accuracy (old)."""
    v = fd.get("test_combined")
    return v if v is not None else fd.get("eval_combined")


def _feat_best_val(fd):
    """Best val accuracy across iters (new) or best train accuracy (old)."""
    v = fd.get("best_val_accuracy")
    return v if v is not None else fd.get("best_train_accuracy")


def load_dataset(dataset_name, results_dir, skip_incomplete=False):
    """Load all optimizer results for a dataset. Returns {optimizer: data}."""
    base = os.path.join(results_dir, dataset_name)
    results = {}
    if not os.path.isdir(base):
        if skip_incomplete:
            warnings.warn(f"Skipping missing dataset directory: {base}")
            return results
        raise FileNotFoundError(base)
    for ts in sorted(os.listdir(base)):
        path = os.path.join(base, ts, "results.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            results[data["metadata"]["optimizer"]] = data
        except (json.JSONDecodeError, KeyError, OSError) as e:
            if skip_incomplete:
                warnings.warn(f"Skipping {path}: {e}")
                continue
            raise
    return results


def feature_baseline(feature_data, key="train_accuracy"):
    """Iter-0 accuracy for the given logical key. `key` is one of:
       - 'train_accuracy' (same in old + new)
       - 'val_accuracy'   (new) / 'eval_accuracy' (old)
    Returns None if missing or sentinel zero.
    """
    ph = feature_data.get("prompt_history", [])
    if not ph:
        return None
    h0 = ph[0]
    if key == "val_accuracy":
        raw = _ph_val(h0)
    elif key == "eval_accuracy":  # legacy callers
        raw = _ph_val(h0)
    else:
        raw = h0.get(key)
    if raw is None:
        return None
    val = _norm(raw)
    return val if val > 0 else None


def consensus_baselines(results, features, dataset_name="", key="train_accuracy"):
    """Per-feature consensus baseline accuracy. Warns and averages if methods disagree."""
    eps = 1e-9
    prefix = f"[{dataset_name}] " if dataset_name else ""
    out = {}
    for feat in features:
        per_method = {}
        for name, data in results.items():
            if feat not in data["features"]:
                continue
            b = feature_baseline(data["features"][feat], key=key)
            if b is not None:
                per_method[name] = b
        if not per_method:
            continue
        vals = list(per_method.values())
        if max(vals) - min(vals) > eps:
            details = ", ".join(f"{m}={v:.4f}" for m, v in per_method.items())
            warnings.warn(
                f"{prefix}Baseline {key} differs across methods for feature '{feat}': {details}"
            )
        out[feat] = float(np.mean(vals))
    return out


def feature_best_series(feature_data, baseline_acc):
    """[baseline, running max after each non-zero candidate train accuracy].

    Drops zero-baseline candidates (DSPy quirk) and entries lacking a candidate
    train accuracy.
    """
    ph = [h for h in feature_data["prompt_history"] if _ph_cand_train(h) is not None]
    candidates = [_norm(_ph_cand_train(h)) for h in ph]
    candidates = [c for c in candidates if c > 0]
    series = [baseline_acc]
    bsf = baseline_acc
    for c in candidates:
        bsf = max(bsf, c)
        series.append(bsf)
    return series


def feature_paired_eval_series(feature_data, baseline_train, baseline_eval):
    """Val (new) / eval (old) accuracy of the candidate that currently holds the
    train running-max.

    Index-aligned with feature_best_series: index 0 = baseline; subsequent indices
    correspond to each surviving (non-zero candidate train acc) entry. When a
    candidate beats the running-max train acc, switch to its val/eval acc; else
    hold. Returns None if no val/eval data is recorded for this feature.
    """
    ph = [h for h in feature_data["prompt_history"] if _ph_cand_train(h) is not None]
    pairs = [(_norm(_ph_cand_train(h)), _ph_cand_val(h)) for h in ph]
    pairs = [(c, e) for c, e in pairs if c > 0]
    if baseline_eval is None and not any(e is not None for _, e in pairs):
        return None
    series = [baseline_eval if baseline_eval is not None else 0.0]
    best_train = baseline_train
    best_eval = series[0]
    for c_train, c_eval in pairs:
        if c_train > best_train and c_eval is not None:
            best_train = c_train
            best_eval = _norm(c_eval)
        series.append(best_eval)
    return series


def resample_to_11(series):
    """Stretch any non-empty series onto 11 evenly-spaced x positions (0..10)."""
    if len(series) == 1:
        return [series[0]] * 11
    x_old = np.linspace(0, 10, len(series))
    return list(np.interp(np.arange(11), x_old, series))


def avg_curve(results, features, baselines):
    """Per-optimizer average best-accuracy curve. Iter 0 = baseline, then 10 steps."""
    out = {}
    for name, data in results.items():
        series = []
        for feat in features:
            if feat not in data["features"] or feat not in baselines:
                continue
            series.append(feature_best_series(data["features"][feat], baselines[feat]))
        if series:
            out[name] = np.mean([resample_to_11(s) for s in series], axis=0)
    return out


def avg_eval_curve(results, features, train_baselines, eval_baselines):
    """Per-optimizer average eval-acc curve paired with train running-max."""
    out = {}
    for name, data in results.items():
        series = []
        for feat in features:
            if feat not in data["features"] or feat not in train_baselines:
                continue
            s = feature_paired_eval_series(
                data["features"][feat],
                train_baselines[feat],
                eval_baselines.get(feat),
            )
            if s is not None:
                series.append(s)
        if series:
            out[name] = np.mean([resample_to_11(s) for s in series], axis=0)
    return out


def _lighten(color, amount=0.55):
    """Blend an RGBA matplotlib color toward white for the lighter (train) overlay."""
    import matplotlib.colors as mcolors
    r, g, b, a = mcolors.to_rgba(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount, a)


def feature_length_series(feature_data):
    """Active-prompt length per iteration, with iter 0 = baseline prompt length."""
    series = [len(feature_data["baseline_prompt"])]
    ph = [h for h in feature_data["prompt_history"] if "candidate_accuracy" in h]
    candidates = [(_norm(h["candidate_accuracy"]), h) for h in ph]
    candidates = [(c, h) for c, h in candidates if c > 0]
    # Track the active prompt length after each step using the same filter as the
    # accuracy curve: take len(prompt) at each surviving entry.
    for _, h in candidates:
        series.append(len(h["prompt"]))
    return series


def avg_length_curve(results, features):
    """Per-optimizer average prompt-length curve, iter 0 = baseline length."""
    out = {}
    for name, data in results.items():
        series = []
        for feat in features:
            if feat not in data["features"]:
                continue
            series.append(feature_length_series(data["features"][feat]))
        if series:
            out[name] = np.mean([resample_to_11(s) for s in series], axis=0)
    return out


def plot_per_dataset(dataset_name, results, features, baselines, eval_baselines, tag):
    """Per-feature figure with an average subplot at the end. Iter 0 = baseline."""
    n_plots = len(features) + 1
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    for ax, feat in zip(axes, features):
        if feat not in baselines:
            continue
        feat_vals = []
        for name, data in results.items():
            if feat not in data["features"]:
                continue
            best = feature_best_series(data["features"][feat], baselines[feat])
            n_pts = len(best)
            x = list(range(n_pts))  # 0 = baseline, 1..N = each surviving candidate
            style = method_styles[name]
            light = _lighten(style["color"])
            ax.plot(x, best, f"{style['marker']}-",
                    label=f"{name} train", color=light, alpha=0.7, markersize=5)
            feat_vals.extend(best)
            paired_eval = feature_paired_eval_series(
                data["features"][feat], baselines[feat], eval_baselines.get(feat),
            )
            if paired_eval is not None:
                ax.plot(x, paired_eval, f"{style['marker']}-",
                        label=f"{name} eval", color=style["color"], alpha=0.9, markersize=5)
                feat_vals.extend(paired_eval)

        ax.set_title(feat)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy")
        if feat_vals:
            ax.set_ylim(max(0.0, min(feat_vals) - 0.03), 1.0)
        else:
            ax.set_ylim(0, 1.0)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Average subplot
    ax_avg = axes[len(features)]
    train_avg = avg_curve(results, features, baselines)
    eval_avg = avg_eval_curve(results, features, baselines, eval_baselines)
    avg_vals = []
    for name, curve in train_avg.items():
        style = method_styles[name]
        ax_avg.plot(np.arange(11), curve, f"{style['marker']}-",
                    label=f"{name} train", color=_lighten(style["color"]), alpha=0.7, markersize=5)
        avg_vals.extend(curve)
    for name, curve in eval_avg.items():
        style = method_styles[name]
        ax_avg.plot(np.arange(11), curve, f"{style['marker']}-",
                    label=f"{name} eval", color=style["color"], alpha=0.9, markersize=5)
        avg_vals.extend(curve)
    ax_avg.set_title("Average (all features)")
    ax_avg.set_xlabel("Iteration")
    ax_avg.set_ylabel("Accuracy")
    if avg_vals:
        ax_avg.set_ylim(max(0.0, min(avg_vals) - 0.03), 1.0)
    else:
        ax_avg.set_ylim(0, 1.0)
    ax_avg.legend(fontsize=6)
    ax_avg.grid(True, alpha=0.3)

    title_methods = " vs ".join(results.keys())
    fig.suptitle(f"[{dataset_name}] {title_methods}: Accuracy by Iteration", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(ANALYSIS_DIR, f"accuracy_comparison_{tag}_{dataset_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def print_ranking(title, results, features, value_fn, skip_incomplete=False):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    method_final = {}
    feat_values = {}
    for name, data in results.items():
        accs = []
        for f in features:
            if f not in data["features"]:
                continue
            try:
                v = value_fn(data["features"][f])
            except (KeyError, TypeError):
                if skip_incomplete:
                    continue
                raise
            feat_values[(name, f)] = v
            accs.append(v)
        if accs:
            method_final[name] = np.mean(accs)

    ranked = sorted(method_final.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, acc) in enumerate(ranked, 1):
        print(f"  {rank}. {name:20s} {acc:.3f}")

    methods = [n for n, _ in ranked]
    print(f"\n{'Feature':<35s}", end="")
    for name in methods:
        print(f"{name:>12s}", end="")
    print()
    print("-" * (35 + 12 * len(methods)))
    for feat in features:
        print(f"{feat:<35s}", end="")
        for name in methods:
            if (name, feat) in feat_values:
                print(f"{feat_values[(name, feat)]:>12.3f}", end="")
            else:
                print(f"{'N/A':>12s}", end="")
        print()
    print("-" * (35 + 12 * len(methods)))
    print(f"{'Average':<35s}", end="")
    for name in methods:
        print(f"{method_final[name]:>12.3f}", end="")
    print()


def plot_2x2(dataset_curves, ylabel, suptitle, out_name, y_range_fn,
             dataset_curves_light=None):
    """Grid figure: one subplot per dataset + avg-of-avgs. Legend only on the avg.

    If dataset_curves_light is provided, those curves are drawn first in a
    lighter shade of the same per-method color (used to overlay train under eval).
    """
    n_plots = len(DATASETS) + 1  # +1 for avg-of-avgs
    ncols = 2 if n_plots <= 4 else 3
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for ax in flat_axes[n_plots:]:
        ax.set_visible(False)

    common = set.intersection(*[set(c) for c in dataset_curves.values()])
    avg_of_avgs = {
        name: np.stack([dataset_curves[ds][name] for ds in DATASETS]).mean(axis=0)
        for name in common
    }
    light_avg_of_avgs = {}
    if dataset_curves_light is not None:
        light_common = set.intersection(*[set(c) for c in dataset_curves_light.values()])
        light_avg_of_avgs = {
            name: np.stack([dataset_curves_light[ds][name] for ds in DATASETS]).mean(axis=0)
            for name in light_common
        }
    for ax, ds in zip(flat_axes[:len(DATASETS)], DATASETS):
        ax_vals = []
        if dataset_curves_light is not None:
            for name, curve in dataset_curves_light[ds].items():
                style = method_styles[name]
                ax.plot(np.arange(11), curve, f"{style['marker']}-",
                        label=f"{name} train", color=_lighten(style["color"]),
                        alpha=0.6, markersize=5)
                ax_vals.append(curve)
        for name, curve in dataset_curves[ds].items():
            style = method_styles[name]
            label = f"{name} eval" if dataset_curves_light is not None else name
            ax.plot(np.arange(11), curve, f"{style['marker']}-",
                    label=label, color=style["color"], alpha=0.9, markersize=5)
            ax_vals.append(curve)
        ax.set_title(f"{ds} (avg across features)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        if ax_vals:
            ax.set_ylim(*y_range_fn(np.concatenate(ax_vals)))
        ax.grid(True, alpha=0.3)

    ax = flat_axes[len(DATASETS)]
    ax_vals = []
    if dataset_curves_light is not None:
        for name in sorted(light_avg_of_avgs):
            style = method_styles[name]
            ax.plot(np.arange(11), light_avg_of_avgs[name], f"{style['marker']}-",
                    label=f"{name} train", color=_lighten(style["color"]),
                    alpha=0.6, markersize=5)
            ax_vals.append(light_avg_of_avgs[name])
    for name in sorted(common):
        style = method_styles[name]
        label = f"{name} eval" if dataset_curves_light is not None else name
        ax.plot(np.arange(11), avg_of_avgs[name], f"{style['marker']}-",
                label=label, color=style["color"], alpha=0.9, markersize=5)
        ax_vals.append(avg_of_avgs[name])
    ax.set_title(f"Average of averages ({' + '.join(DATASETS)})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    if ax_vals:
        ax.set_ylim(*y_range_fn(np.concatenate(ax_vals)))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(ANALYSIS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined(dataset_eval_curves, dataset_train_curves, tag):
    plot_2x2(
        dataset_eval_curves,
        ylabel="Accuracy",
        suptitle="Per-dataset average accuracy (train light, eval bold) and average-of-averages",
        out_name=f"accuracy_comparison_{tag}_combined.png",
        y_range_fn=lambda vals: (max(0.0, float(vals.min()) - 0.03), 1.0),
        dataset_curves_light=dataset_train_curves,
    )


def plot_length_combined(dataset_length_curves, tag):
    def y_range(vals):
        lo, hi = float(vals.min()), float(vals.max())
        pad = (hi - lo) * 0.05 or 1.0
        return (max(0.0, lo - pad), hi + pad)

    plot_2x2(
        dataset_length_curves,
        ylabel="Prompt length (chars)",
        suptitle="Per-dataset average prompt length and average-of-averages",
        out_name=f"prompt_length_{tag}_combined.png",
        y_range_fn=y_range,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--skip-incomplete", action="store_true",
                        help="Skip datasets/features with missing data instead of crashing.")
    args = parser.parse_args()
    results_dir = args.results_dir
    skip_incomplete = args.skip_incomplete
    tag = os.path.basename(os.path.normpath(results_dir))

    raw_results = {ds: load_dataset(ds, results_dir, skip_incomplete=skip_incomplete)
                   for ds in DATASETS}
    # Restrict to optimizers present in ALL datasets, on every plot and table.
    non_empty = [set(r) for r in raw_results.values() if r]
    if not non_empty:
        if skip_incomplete:
            warnings.warn("No datasets had any results; nothing to do.")
            return
        raise RuntimeError("No datasets had any results.")
    common_opts = set.intersection(*non_empty) if skip_incomplete \
        else set.intersection(*[set(r) for r in raw_results.values()])
    results_by_ds = {
        ds: {opt: data for opt, data in r.items() if opt in common_opts}
        for ds, r in raw_results.items()
    }

    dataset_train_curves = {}
    dataset_eval_curves = {}
    dataset_length_curves = {}
    for ds, results in results_by_ds.items():
        if skip_incomplete and not results:
            warnings.warn(f"Skipping dataset {ds}: no results loaded.")
            dataset_train_curves[ds] = {}
            dataset_eval_curves[ds] = {}
            dataset_length_curves[ds] = {}
            continue

        all_feats = set()
        for data in results.values():
            all_feats.update(data["features"].keys())
        features = sorted(all_feats - EXCLUDE_FEATURES)

        baselines = consensus_baselines(results, features, ds, key="train_accuracy")
        eval_baselines = consensus_baselines(results, features, ds, key="val_accuracy")

        plot_per_dataset(ds, results, features, baselines, eval_baselines, tag)
        print_ranking(f"[{ds}] Best Val Accuracy Ranking", results, features,
                      lambda fd: _norm(_feat_best_val(fd)),
                      skip_incomplete=skip_incomplete)
        print_ranking(f"[{ds}] Final Test Accuracy Ranking", results, features,
                      lambda fd: _feat_test_combined(fd),
                      skip_incomplete=skip_incomplete)

        print("\n" + "=" * 60)
        print(f"[{ds}] Average Prompt Length (chars)")
        print("=" * 60)
        for name, data in results.items():
            lengths = []
            for f in features:
                if f not in data["features"]:
                    continue
                try:
                    lengths.append(len(data["features"][f]["best_prompt"]))
                except (KeyError, TypeError):
                    if skip_incomplete:
                        continue
                    raise
            if lengths:
                print(f"  {name:20s} {np.mean(lengths):>8.0f}")

        dataset_train_curves[ds] = avg_curve(results, features, baselines)
        dataset_eval_curves[ds] = avg_eval_curve(results, features, baselines, eval_baselines)
        dataset_length_curves[ds] = avg_length_curve(results, features)

    if any(dataset_eval_curves.values()) or any(dataset_train_curves.values()):
        plot_combined(dataset_eval_curves, dataset_train_curves, tag)
    elif skip_incomplete:
        warnings.warn("No data available for combined accuracy plot; skipping.")

    if any(dataset_length_curves.values()):
        plot_length_combined(dataset_length_curves, tag)
    elif skip_incomplete:
        warnings.warn("No data available for combined length plot; skipping.")


if __name__ == "__main__":
    main()
