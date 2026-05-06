import argparse
import json
import math
import itertools
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_RESULTS_DIR = "auto_prompt/preds/Qwen_Qwen2.5-72B-Instruct-AWQ/paper3"
DATASETS = ["acne", "odd", "sdoh"]
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


def load_dataset(dataset_name, results_dir):
    """Load all optimizer results for a dataset. Returns {optimizer: data}."""
    base = os.path.join(results_dir, dataset_name)
    results = {}
    for ts in sorted(os.listdir(base)):
        path = os.path.join(base, ts, "results.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            data = json.load(f)
        results[data["metadata"]["optimizer"]] = data
    return results


def feature_baseline(feature_data, key="train_accuracy"):
    """Iter-0 accuracy for the given key. Returns None if missing or sentinel zero."""
    ph = feature_data.get("prompt_history", [])
    if not ph:
        return None
    val = _norm(ph[0].get(key, 0.0))
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
    """[baseline, running max after each accepted/non-zero candidate].

    Drops dspy zero-baseline candidates and entries lacking candidate_accuracy.
    """
    ph = [h for h in feature_data["prompt_history"] if "candidate_accuracy" in h]
    candidates = [_norm(h["candidate_accuracy"]) for h in ph]
    candidates = [c for c in candidates if c > 0]
    series = [baseline_acc]
    bsf = baseline_acc
    for c in candidates:
        bsf = max(bsf, c)
        series.append(bsf)
    return series


def feature_paired_eval_series(feature_data, baseline_train, baseline_eval):
    """Eval accuracy of the candidate that currently holds the train running-max.

    Index-aligned with feature_best_series: index 0 = baseline; subsequent indices
    correspond to each surviving (non-zero candidate_accuracy) entry. When a
    candidate beats the running-max train acc, switch to its eval acc; else hold.
    Returns None if no eval data is recorded for this feature.
    """
    ph = [h for h in feature_data["prompt_history"] if "candidate_accuracy" in h]
    pairs = [(_norm(h["candidate_accuracy"]), h.get("candidate_eval_accuracy")) for h in ph]
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
            paired_eval = feature_paired_eval_series(
                data["features"][feat], baselines[feat], eval_baselines.get(feat),
            )
            if paired_eval is not None:
                ax.plot(x, paired_eval, f"{style['marker']}-",
                        label=f"{name} eval", color=style["color"], alpha=0.9, markersize=5)

        ax.set_title(feat)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Average subplot
    ax_avg = axes[len(features)]
    train_avg = avg_curve(results, features, baselines)
    eval_avg = avg_eval_curve(results, features, baselines, eval_baselines)
    for name, curve in train_avg.items():
        style = method_styles[name]
        ax_avg.plot(np.arange(11), curve, f"{style['marker']}-",
                    label=f"{name} train", color=_lighten(style["color"]), alpha=0.7, markersize=5)
    for name, curve in eval_avg.items():
        style = method_styles[name]
        ax_avg.plot(np.arange(11), curve, f"{style['marker']}-",
                    label=f"{name} eval", color=style["color"], alpha=0.9, markersize=5)
    ax_avg.set_title("Average (all features)")
    ax_avg.set_xlabel("Iteration")
    ax_avg.set_ylabel("Accuracy")
    ax_avg.set_ylim(0, 1.05)
    ax_avg.legend(fontsize=6)
    ax_avg.grid(True, alpha=0.3)

    title_methods = " vs ".join(results.keys())
    fig.suptitle(f"[{dataset_name}] {title_methods}: Accuracy by Iteration", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(ANALYSIS_DIR, f"accuracy_comparison_{tag}_{dataset_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def print_ranking(title, results, features, value_fn):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    method_final = {}
    for name, data in results.items():
        accs = [value_fn(data["features"][f]) for f in features if f in data["features"]]
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
            if feat in results[name]["features"]:
                print(f"{value_fn(results[name]['features'][feat]):>12.3f}", end="")
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
    """2x2 figure: per-dataset curves + avg-of-avgs. Legend only in bottom-right.

    If dataset_curves_light is provided, those curves are drawn first in a
    lighter shade of the same per-method color (used to overlay train under eval).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
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
    all_vals_list = [c for ds in DATASETS for c in dataset_curves[ds].values()]
    all_vals_list += list(avg_of_avgs.values())
    if dataset_curves_light is not None:
        all_vals_list += [c for ds in DATASETS for c in dataset_curves_light[ds].values()]
        all_vals_list += list(light_avg_of_avgs.values())
    all_vals = np.concatenate(all_vals_list)
    y_min, y_max = y_range_fn(all_vals)

    for ax, ds in zip(axes.flatten()[:3], DATASETS):
        if dataset_curves_light is not None:
            for name, curve in dataset_curves_light[ds].items():
                style = method_styles[name]
                ax.plot(np.arange(11), curve, f"{style['marker']}-",
                        label=f"{name} train", color=_lighten(style["color"]),
                        alpha=0.6, markersize=5)
        for name, curve in dataset_curves[ds].items():
            style = method_styles[name]
            label = f"{name} eval" if dataset_curves_light is not None else name
            ax.plot(np.arange(11), curve, f"{style['marker']}-",
                    label=label, color=style["color"], alpha=0.9, markersize=5)
        ax.set_title(f"{ds} (avg across features)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if dataset_curves_light is not None:
        for name in sorted(light_avg_of_avgs):
            style = method_styles[name]
            ax.plot(np.arange(11), light_avg_of_avgs[name], f"{style['marker']}-",
                    label=f"{name} train", color=_lighten(style["color"]),
                    alpha=0.6, markersize=5)
    for name in sorted(common):
        style = method_styles[name]
        label = f"{name} eval" if dataset_curves_light is not None else name
        ax.plot(np.arange(11), avg_of_avgs[name], f"{style['marker']}-",
                label=label, color=style["color"], alpha=0.9, markersize=5)
    ax.set_title("Average of averages (acne + odd + sdoh)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min, y_max)
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
        y_range_fn=lambda vals: (max(0.0, float(vals.min()) - 0.03), 1.05),
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
    args = parser.parse_args()
    results_dir = args.results_dir
    tag = os.path.basename(os.path.normpath(results_dir))

    raw_results = {ds: load_dataset(ds, results_dir) for ds in DATASETS}
    # Restrict to optimizers present in ALL datasets, on every plot and table.
    common_opts = set.intersection(*[set(r) for r in raw_results.values()])
    results_by_ds = {
        ds: {opt: data for opt, data in r.items() if opt in common_opts}
        for ds, r in raw_results.items()
    }

    dataset_train_curves = {}
    dataset_eval_curves = {}
    dataset_length_curves = {}
    for ds, results in results_by_ds.items():
        all_feats = set()
        for data in results.values():
            all_feats.update(data["features"].keys())
        features = sorted(all_feats - EXCLUDE_FEATURES)

        baselines = consensus_baselines(results, features, ds, key="train_accuracy")
        eval_baselines = consensus_baselines(results, features, ds, key="eval_accuracy")

        plot_per_dataset(ds, results, features, baselines, eval_baselines, tag)
        print_ranking(f"[{ds}] Final Train Accuracy Ranking", results, features,
                      lambda fd: _norm(fd["best_train_accuracy"]))
        print_ranking(f"[{ds}] Final Holdout (Eval) Accuracy Ranking", results, features,
                      lambda fd: fd["eval_combined"])

        print("\n" + "=" * 60)
        print(f"[{ds}] Average Prompt Length (chars)")
        print("=" * 60)
        for name, data in results.items():
            lengths = [len(data["features"][f]["best_prompt"])
                       for f in features if f in data["features"]]
            if lengths:
                print(f"  {name:20s} {np.mean(lengths):>8.0f}")

        dataset_train_curves[ds] = avg_curve(results, features, baselines)
        dataset_eval_curves[ds] = avg_eval_curve(results, features, baselines, eval_baselines)
        dataset_length_curves[ds] = avg_length_curve(results, features)

    plot_combined(dataset_eval_curves, dataset_train_curves, tag)
    plot_length_combined(dataset_length_curves, tag)


if __name__ == "__main__":
    main()
