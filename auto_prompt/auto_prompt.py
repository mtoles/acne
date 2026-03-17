"""
Label-2-Prompt: Automatic prompt optimization via iterative rule inference.

Given a labeled dataset of medical record QA pairs and an initial baseline prompt,
this system automatically discovers and incorporates implicit annotator assumptions
into the prompt to maximize accuracy.

Algorithm:
    1. Run LLM with current prompt p on dataset D = {(x_i, y_i)}
    2. Identify error set E = {(x_i, y_i, y_hat_i) | y_hat_i != y_i}
    3. (Optional) Cluster errors by input/output similarity to reduce context length
    4. Prompt a meta-LLM to infer a rule explaining the systematic errors
    5. Create candidate prompt p_next = p_curr + rule
    6. If Acc(p_next, D) > Acc(p_curr, D), accept the rule
    7. Repeat until convergence (no errors or no improving rules)

Label-2-Prompt differs from existing methods in that it extracts implicit user intent
-- the unspoken assumptions experts make during annotation (e.g. answer format, units,
edge case handling) -- rather than optimizing reasoning or instruction phrasing.

Baseline algorithms (implemented in prompt_optimizers.py):

    OPRO (Yang et al., 2023): Meta-LLM sees scored prompt history + errors/correct
        examples and proposes a new prompt. Unique: treats prompt optimization as a
        natural language optimization problem; the meta-prompt contains a trajectory
        of past prompts sorted by score.

    PromptAgent (Wang et al., 2023): Uses Monte Carlo Tree Search over the space of
        prompt rewrites. Each node is a prompt, edges are LLM-proposed edits.
        Unique: strategic exploration/exploitation tradeoff via UCB; plans multiple
        steps ahead rather than greedy single-step refinement.

    DSPy MIPROv2 (Khattab et al., 2023): Optimizes multi-stage LLM pipelines
        end-to-end. Generates candidate instructions from random training subsets,
        bootstraps few-shot demonstrations, and uses Bayesian surrogate models for
        efficient search. Unique: optimizes full pipelines (not just single prompts)
        and jointly tunes instructions + demonstrations.

    ETGPO (arXiv 2602.00997): Error Taxonomy-Guided Prompt Optimization. Collects
        errors and asks the LLM to categorize each into an error taxonomy by root cause.
        Reuses categories across batches. Filters out long-tail errors to focus on the
        most prevalent failure categories. Unique: LLM-based clustering with persistent
        taxonomy that grows over iterations.

    AMPO (arXiv 2410.08696): Automatic Multi-Branched Prompt Optimization. Two-agent
        LLM pipeline: LLM-Analyzer identifies failure causes, LLM-Summarizer groups
        them into patterns with importance scores. Creates multi-branched prompts where
        each branch handles a different failure pattern. Unique: multi-branch prompts
        rather than a single optimized prompt.

    Error-Driven Prompt Optimization (arXiv 2512.13323): Clusters erroneous predictions
        using HDBSCAN (density-based clustering), selects the largest cluster, and
        formulates new prompt rules to address it. Unique: uses algorithmic clustering
        (HDBSCAN) rather than LLM-based categorization.

    SAMMO (Schnabel et al., 2024): Represents prompts as structured ASTs, applies
        typed mutation operators (paraphrase, add/remove sections) to specific nodes,
        and searches via beam search. Not implemented because its contributions are
        framework-level (AST representation, beam search) rather than algorithmic, and
        our prompts are too simple to benefit from AST structure. 

    Matt-opt (ours): for multiple choice questions, use confusion matrix clustering
        (group errors by ground_truth/prediction pairs) to identify systematic failure
        patterns. Address one confusion pattern per iteration with targeted add/remove
        instruction mutations.

Not implemented:

    EvoPrompt (Guo et al., 2023): Maintains a population of prompts, applies
        evolutionary operators (crossover, mutation) to generate candidates, evaluates
        on train set, and selects the fittest. Unique: no meta-LLM needed for the
        optimization step itself; diversity maintained via population. Not implemented
        because it requires an initial set of multiple prompts.

    APE (Zhou et al., 2022): Generates N candidate prompts from input-output examples
        in parallel, evaluates all, selects the best. The foundational "generate then
        select" approach. Not implemented because in our iterative framework it reduces
        to a weaker OPRO: it sees only correct examples (no errors, no trajectory
        history, no current prompt to refine). Its key contribution -- parallel candidate
        generation -- could be added as a variant of OPRO instead.

"""

import pandas as pd
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import yaml
from datetime import datetime
import argparse
import json
import re

# Add project root to path and chdir so imports and relative paths work from any directory
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from pt_features import PtFeaturesMeta, PtFeatureBase, PtDateFeatureBase, PtNumericFeatureBase
from models import MrModel, DummyModel
from utils import get_dataset, compute_numeric_pct_error, compute_numeric_abs_error
from prompt_optimizers import DummyOptimizer, OPROOptimizer, OursOptimizer

tqdm.pandas()

PROMPT_HISTORY_PATH = "labeled_data/LLM Adjustment Tracking.xlsx"

OPTIMIZERS = {
    "dummy": DummyOptimizer,
    "opro": OPROOptimizer,
    "ours": OursOptimizer,
}


def load_model_id():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
        return config["model"]["id"]


def parse_args():
    parser = argparse.ArgumentParser(description="Label-2-Prompt: automatic prompt optimization")
    parser.add_argument(
        "--inference_type",
        choices=["logit", "cot"],
        default="cot",
        help="Type of inference to use: logit or cot (chain of thought, default)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Downsample the data to the given number of examples per feature",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="List of features to run on. If not specified, runs on all available features",
    )
    parser.add_argument(
        "--data_source",
        choices=["mgb", "mimic"],
        default="mimic",
        help="Data source to use: mgb (default) or mimic",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of optimization iterations per feature (default: 3)",
    )
    parser.add_argument(
        "--optimizer",
        choices=list(OPTIMIZERS.keys()),
        default="dummy",
        help="Prompt optimizer to use (default: dummy)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=50,
        help="Number of parallel workers for inference (default: 50)",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="default",
        help="Note to use as a subdirectory in the output path (e.g. model_dir/note/timestamp)",
    )
    return parser.parse_args()


# Load model ID and initialize model
model_id = load_model_id()
model = MrModel(model_id=model_id)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_baseline_prompts():
    """Load Dorsa's baseline prompts (Prompt #1) from Excel file, keyed by feature name"""
    df = pd.read_excel(PROMPT_HISTORY_PATH)
    dorsa_df = df[df['Prompt Tuner'] == 'DM'].copy()

    prompts = {}
    for _, row in dorsa_df.iterrows():
        feature_name = row['Data Point']
        if pd.isna(feature_name):
            continue
        prompt_text = row['Prompt #1 (baseline)']
        if pd.notna(prompt_text) and str(prompt_text).strip():
            prompts[feature_name] = str(prompt_text).strip()

    return prompts


def format_prompt_with_keyword(prompt_text, keyword):
    """Replace placeholders in prompt text with actual keyword"""
    formatted = prompt_text.replace('{keyword}', keyword)
    formatted = formatted.replace('{abx}', keyword)
    return formatted


def process_single_chunk(args):
    """Process a single chunk and return the result with index"""
    i, chunk, found_kw, target_cls, inference_type, custom_query = args
    preds_for_chunk = {}

    kwargs = {}
    if custom_query:
        kwargs['custom_query'] = custom_query

    pred_dict = target_cls.forward(
        model=model, chunk=chunk, keyword=found_kw, inference_type=inference_type, **kwargs
    )
    preds_for_chunk.update({k: v for k, v in pred_dict.items()})

    return i, preds_for_chunk


def run_inference(chunk_df, prompt_text, feature_name, target_cls, inference_type, n_workers=50, desc=""):
    """Run inference on a dataframe with a given prompt. Returns (chunk_df_with_preds, metrics_dict)."""
    is_numeric_feature = issubclass(target_cls, PtNumericFeatureBase)

    chunk_args = []
    for i, (chunk, found_kw) in enumerate(
        zip(chunk_df["chunk"], chunk_df["found_keywords"])
    ):
        formatted_prompt = format_prompt_with_keyword(prompt_text, found_kw)
        chunk_args.append((i, chunk, found_kw, target_cls, inference_type, formatted_prompt))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(process_single_chunk, args): args[0] for args in chunk_args
        }

        with tqdm(total=len(chunk_args), desc=desc) as pbar:
            for future in as_completed(future_to_index):
                i, preds_for_chunk = future.result()

                for key, value in preds_for_chunk.items():
                    if key not in chunk_df.columns:
                        chunk_df[key] = pd.NA
                    chunk_df.loc[chunk_df.index[i], key] = value

                pbar.update(1)

    ground_truth = chunk_df["val_unified"]
    pred_column = feature_name

    if is_numeric_feature:
        def normalize_numeric(val):
            val_str = str(val).strip()
            if val_str.upper() == "F":
                return "F"
            try:
                return str(int(float(val_str)))
            except (ValueError, TypeError):
                return val_str
        chunk_df[pred_column] = chunk_df[pred_column].apply(normalize_numeric)
        chunk_df["val_unified"] = chunk_df["val_unified"].apply(normalize_numeric)
        ground_truth = chunk_df["val_unified"]

    predictions = chunk_df[pred_column]

    natural_df = chunk_df[chunk_df["is_synthetic"] == False]
    synthetic_df = chunk_df[chunk_df["is_synthetic"] == True]

    correct_combined = (predictions == ground_truth).sum()
    total_combined = len(predictions)
    accuracy_combined = correct_combined / total_combined if total_combined > 0 else 0

    correct_natural = (natural_df[pred_column] == natural_df["val_unified"]).sum() if len(natural_df) > 0 else 0
    total_natural = len(natural_df)
    accuracy_natural = correct_natural / total_natural if total_natural > 0 else 0

    correct_synthetic = (synthetic_df[pred_column] == synthetic_df["val_unified"]).sum() if len(synthetic_df) > 0 else 0
    total_synthetic = len(synthetic_df)
    accuracy_synthetic = correct_synthetic / total_synthetic if total_synthetic > 0 else 0

    metrics = {
        "combined": accuracy_combined,
        "natural": accuracy_natural,
        "synthetic": accuracy_synthetic,
    }

    if is_numeric_feature:
        pct_errors = [compute_numeric_pct_error(p, g) for p, g in zip(predictions, ground_truth)]
        metrics["avg_pct_error"] = sum(pct_errors) / len(pct_errors) if pct_errors else 0.0

        abs_errors = [compute_numeric_abs_error(p, g) for p, g in zip(predictions, ground_truth)]
        abs_errors_valid = [e for e in abs_errors if e is not None]
        metrics["avg_abs_error"] = sum(abs_errors_valid) / len(abs_errors_valid) if abs_errors_valid else None

    # Collect per-example records (all examples, not just errors)
    error_mask = predictions != ground_truth
    metrics["records"] = [
        {
            "chunk": str(chunk_df["chunk"].iloc[i]),
            "keyword": str(chunk_df["found_keywords"].iloc[i]),
            "ground_truth": str(ground_truth.iloc[i]),
            "prediction": str(predictions.iloc[i]),
            "correct": not error_mask.iloc[i],
        }
        for i in range(len(predictions))
    ]
    metrics["errors"] = [r for r in metrics["records"] if not r["correct"]]

    print(f"  Accuracy: {accuracy_combined:.3f} ({correct_combined}/{total_combined})")
    if is_numeric_feature:
        print(f"  Avg percent error: {metrics['avg_pct_error']:.1f}%")
        if metrics.get("avg_abs_error") is not None:
            print(f"  Avg absolute error: {metrics['avg_abs_error']:.1f} days")
    if total_natural > 0:
        print(f"  Natural: {accuracy_natural:.3f} ({correct_natural}/{total_natural})")
    if total_synthetic > 0:
        print(f"  Synthetic: {accuracy_synthetic:.3f} ({correct_synthetic}/{total_synthetic})")

    return chunk_df, metrics


def process_file(file_path, inference_type, downsample=None, data_source=None,
                 baseline_prompts=None, feature_name_override=None, optimizer=None, iterations=3, n_workers=50, note="default"):
    feature_name = feature_name_override or file_path.stem.replace("_chunks", "")
    print(f"\nProcessing {feature_name}")

    preds_dir = Path("preds")
    model_dir = preds_dir / model_id.replace("/", "_")
    timestamp_dir = model_dir / note / timestamp
    feature_dir = timestamp_dir / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    target_cls = PtFeaturesMeta.registry[feature_name]
    feature_metadata = {}
    if getattr(target_cls, "data_source_feature", None) or getattr(target_cls, "gt_column", "val_unified") != "val_unified":
        feature_metadata[feature_name] = {
            "data_source_feature": getattr(target_cls, "data_source_feature", None) or feature_name,
            "gt_column": getattr(target_cls, "gt_column", "val_unified"),
        }

    datasets = get_dataset(
        data_source=data_source or "mgb",
        feature_names=[feature_name],
        train_split=0.5,
        downsample=downsample,
        random_state=42,
        feature_metadata=feature_metadata,
    )

    if feature_name not in datasets:
        raise ValueError(f"Dataset not found for feature: {feature_name}")

    train_df = datasets[feature_name]["train"].copy()
    eval_df = datasets[feature_name]["eval"].copy()

    train_df = train_df[train_df["val_unified"] != "DROP"]
    eval_df = eval_df[eval_df["val_unified"] != "DROP"]

    print(f"Train: {len(train_df)} ({len(train_df[train_df['is_synthetic'] == False])} natural, {len(train_df[train_df['is_synthetic'] == True])} synthetic)")
    print(f"Eval: {len(eval_df)} ({len(eval_df[eval_df['is_synthetic'] == False])} natural, {len(eval_df[eval_df['is_synthetic'] == True])} synthetic)")

    # --- Optimization loop on train set ---
    current_prompt = baseline_prompts[feature_name]
    best_prompt = current_prompt
    best_train_accuracy = 0.0
    train_accuracy_history = []
    prompt_history = []  # track prompt text and accuracy per iteration
    all_records = []  # accumulate per-example records across iterations

    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration} (train) ---")
        print(f"Prompt ({len(current_prompt)} chars): {current_prompt[:120]}...")

        train_chunk_df = train_df.copy()
        _, train_metrics = run_inference(
            train_chunk_df, current_prompt, feature_name, target_cls, inference_type,
            n_workers=n_workers, desc=f"Train iter {iteration}"
        )

        train_accuracy = train_metrics["combined"]
        train_accuracy_history.append(train_accuracy)
        prompt_history.append({"iteration": iteration, "prompt": current_prompt, "train_accuracy": train_accuracy})

        # Tag and accumulate records
        for rec in train_metrics["records"]:
            rec["feature"] = feature_name
            rec["split"] = "train"
            rec["iteration"] = iteration
            rec["prompt"] = current_prompt
        all_records.extend(train_metrics["records"])

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            best_prompt = current_prompt

        if not train_metrics["errors"]:
            print("  No errors on train set -- stopping early")
            break

        # Optimizer proposes a new prompt (gets all records so it can see correct + incorrect)
        optimizer_result = optimizer.step(current_prompt, train_metrics["records"])
        candidate_prompt = optimizer_result["prompt"]
        prompt_history[-1]["candidate_prompt"] = candidate_prompt
        prompt_history[-1]["candidate_raw_response"] = optimizer_result["raw_response"]

        # Evaluate candidate and accept/reject
        candidate_chunk_df = train_df.copy()
        _, candidate_metrics = run_inference(
            candidate_chunk_df, candidate_prompt, feature_name, target_cls, inference_type,
            n_workers=n_workers, desc=f"Candidate iter {iteration}"
        )
        candidate_accuracy = candidate_metrics["combined"]
        accuracy_delta = candidate_accuracy - train_accuracy
        accepted = bool(candidate_accuracy > train_accuracy)
        prompt_history[-1]["candidate_accuracy"] = candidate_accuracy
        prompt_history[-1]["accepted"] = accepted

        rule = optimizer_result.get("rule", candidate_prompt)
        optimizer.feedback(rule, accepted, accuracy_delta)

        if accepted:
            print(f"  ACCEPTED candidate (acc {train_accuracy:.3f} -> {candidate_accuracy:.3f}, {accuracy_delta:+.3f})")
            current_prompt = candidate_prompt
        else:
            print(f"  REJECTED candidate (acc {train_accuracy:.3f} -> {candidate_accuracy:.3f}, {accuracy_delta:+.3f})")

    # Use the best prompt found during optimization
    print(f"\n--- Final eval with best prompt (train acc={best_train_accuracy:.3f}) ---")
    print(f"Best prompt ({len(best_prompt)} chars): {best_prompt[:120]}...")

    eval_chunk_df = eval_df.copy()
    _, eval_metrics = run_inference(
        eval_chunk_df, best_prompt, feature_name, target_cls, inference_type,
        n_workers=n_workers, desc="Eval (best prompt)"
    )

    # Tag eval records
    for rec in eval_metrics["records"]:
        rec["feature"] = feature_name
        rec["split"] = "eval"
        rec["iteration"] = len(train_accuracy_history) - 1  # best iteration
        rec["prompt"] = best_prompt
    all_records.extend(eval_metrics["records"])

    return {
        "feature_name": feature_name,
        "baseline_prompt": baseline_prompts[feature_name],
        "train_chunks": len(train_df),
        "eval_chunks": len(eval_df),
        "train_accuracy_history": train_accuracy_history,
        "best_train_accuracy": best_train_accuracy,
        "eval_metrics": eval_metrics,
        "best_prompt": best_prompt,
        "prompt_history": prompt_history,
        "records": all_records,
    }


def main():
    args = parse_args()
    inference_type = args.inference_type
    downsample = args.downsample
    target_features = args.features

    print(f"Inference type: {inference_type}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Iterations: {args.iterations}")
    if downsample is not None:
        print(f"Downsampling to {downsample} examples per feature")
    if target_features is not None:
        print(f"Features: {target_features}")

    optimizer = OPTIMIZERS[args.optimizer](model=model)

    print("\nLoading baseline prompts...")
    baseline_prompts = load_baseline_prompts()
    print(f"Loaded {len(baseline_prompts)} baseline prompts")

    labeled_data_dir = Path(f"labeled_data/{args.data_source}")

    preds_dir = Path("auto_prompt/preds")
    model_dir = preds_dir / model_id.replace("/", "_")
    timestamp_dir = model_dir / args.note / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    excel_files = []
    for subdir in labeled_data_dir.iterdir():
        if subdir.is_dir():
            chunk_files = list(subdir.glob("*_chunks.xlsx"))
            excel_files.extend(chunk_files)
    excel_files = sorted(excel_files)

    features_to_process = []
    if target_features is not None:
        file_feature_to_path = {}
        for file_path in excel_files:
            file_feature_to_path[file_path.stem.replace("_chunks", "")] = file_path

        for feat_name in target_features:
            target_cls = PtFeaturesMeta.registry[feat_name]
            source_feature = getattr(target_cls, "data_source_feature", None) or feat_name
            if source_feature in file_feature_to_path:
                features_to_process.append((feat_name, file_feature_to_path[source_feature]))
            else:
                print(f"Skipping {feat_name} (no data file for {source_feature})")

        print(f"Filtered to {len(features_to_process)} features: {[f[0] for f in features_to_process]}")
    else:
        for file_path in excel_files:
            feature_name = file_path.stem.replace("_chunks", "")
            features_to_process.append((feature_name, file_path))

    all_results = {}

    records_path = timestamp_dir / "records.jsonl"
    results_path = timestamp_dir / "results.json"

    def save_feature_results(feature_name, results):
        """Save per-feature results immediately after each feature completes."""
        # Append records to .jsonl
        with open(records_path, "a") as f:
            for rec in results["records"]:
                f.write(json.dumps(rec) + "\n")

        # Write/update aggregate results.json with all completed features so far
        serializable = {
            "metadata": {
                "model": model_id,
                "optimizer": args.optimizer,
                "iterations": args.iterations,
            },
            "features": {
                feat: {
                    "baseline_prompt": r["baseline_prompt"],
                    "train_accuracy_history": r["train_accuracy_history"],
                    "best_train_accuracy": r["best_train_accuracy"],
                    "eval_combined": r["eval_metrics"]["combined"],
                    "eval_natural": r["eval_metrics"]["natural"],
                    "eval_synthetic": r["eval_metrics"]["synthetic"],
                    "best_prompt": r["best_prompt"],
                    "prompt_history": r["prompt_history"],
                }
                for feat, r in all_results.items()
            },
        }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)

        # Print per-feature summary line
        train_hist = " -> ".join(f"{a:.3f}" for a in results["train_accuracy_history"])
        eval_acc = results["eval_metrics"]["combined"]
        line = f"  >> {feature_name}: train=[{train_hist}] eval={eval_acc:.3f}"
        if "avg_pct_error" in results["eval_metrics"]:
            line += f" pct_err={results['eval_metrics']['avg_pct_error']:.1f}%"
        print(line)
        print(f"  >> Saved to: {results_path}")

    for feature_name, file_path in features_to_process:
        print(f"\n{'='*80}")
        print(f"Processing {feature_name}...")
        print(f"{'='*80}")
        target_cls = PtFeaturesMeta.registry.get(feature_name)
        if target_cls and issubclass(target_cls, (PtDateFeatureBase, PtNumericFeatureBase)):
            print(f"  Skipping {feature_name} (date/numeric feature, MC only for now)")
            continue
        results = process_file(
            file_path, inference_type, downsample,
            data_source=args.data_source, baseline_prompts=baseline_prompts,
            feature_name_override=feature_name, optimizer=optimizer,
            iterations=args.iterations, n_workers=args.n_workers, note=args.note,
        )
        if results:
            all_results[feature_name] = results
            save_feature_results(feature_name, results)

    # Final summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    summary_path = timestamp_dir / "summary.txt"
    with open(summary_path, "w") as f:
        header = f"Model: {model_id} | Optimizer: {args.optimizer} | Iterations: {args.iterations}\n"
        print(header)
        f.write(header)

        for feature, results in all_results.items():
            train_hist = " -> ".join(f"{a:.3f}" for a in results["train_accuracy_history"])
            eval_acc = results["eval_metrics"]["combined"]
            line = f"{feature}: train=[{train_hist}] eval={eval_acc:.3f}"
            if "avg_pct_error" in results["eval_metrics"]:
                line += f" pct_err={results['eval_metrics']['avg_pct_error']:.1f}%"
            print(line)
            f.write(line + "\n")

            errors = results["eval_metrics"].get("errors", [])
            if errors:
                f.write(f"  First {min(5, len(errors))} eval errors:\n")
                for err in errors[:5]:
                    err_line = f"    pred={err['prediction']}, gt={err['ground_truth']}, chunk={err['chunk'][:150]}"
                    f.write(err_line + "\n")

    print(f"\nAll results: {results_path}")


if __name__ == "__main__":
    main()
