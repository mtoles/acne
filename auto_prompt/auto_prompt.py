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

tqdm.pandas()  # Enable tqdm for pandas operations

PROMPT_HISTORY_PATH = "labeled_data/LLM Adjustment Tracking.xlsx"

# Load MODEL_ID from config.yml
def load_model_id():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
        return config["model"]["id"]


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Compare predictions to ground truth")
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
        default="mgb",
        help="Data source to use: mgb (default) or mimic",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run on the eval set (skip training set)",
    )
    return parser.parse_args()


# Load model ID and initialize model
model_id = load_model_id()
model = MrModel(model_id=model_id)

# Generate timestamp for this run
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


def process_file(file_path, inference_type, downsample=None, data_source=None, baseline_prompts=None, feature_name_override=None, eval_only=False):
    feature_name = feature_name_override or file_path.stem.replace("_chunks", "")
    print(f"\nProcessing {feature_name}")

    # Create feature directory structure: preds/model_id/timestamp/feature_name
    preds_dir = Path("preds")
    model_dir = preds_dir / model_id.replace("/", "_")
    timestamp_dir = model_dir / timestamp
    feature_dir = timestamp_dir / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Build feature metadata from the class attributes
    target_cls = PtFeaturesMeta.registry[feature_name]
    feature_metadata = {}
    if getattr(target_cls, "data_source_feature", None) or getattr(target_cls, "gt_column", "val_unified") != "val_unified":
        feature_metadata[feature_name] = {
            "data_source_feature": getattr(target_cls, "data_source_feature", None) or feature_name,
            "gt_column": getattr(target_cls, "gt_column", "val_unified"),
        }

    # Use get_dataset to load and split the data
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

    # Filter out DROP values
    train_df = train_df[train_df["val_unified"] != "DROP"]
    eval_df = eval_df[eval_df["val_unified"] != "DROP"]

    train_natural_count = len(train_df[train_df["is_synthetic"] == False])
    train_synthetic_count = len(train_df[train_df["is_synthetic"] == True])
    eval_natural_count = len(eval_df[eval_df["is_synthetic"] == False])
    eval_synthetic_count = len(eval_df[eval_df["is_synthetic"] == True])

    print(f"Train set: {len(train_df)} examples ({train_natural_count} natural, {train_synthetic_count} synthetic)")
    print(f"Eval set: {len(eval_df)} examples ({eval_natural_count} natural, {eval_synthetic_count} synthetic)")

    # Get baseline prompt for this feature
    prompt_text = baseline_prompts[feature_name]

    # Store results per split
    results_by_split = {}

    splits = [("eval", eval_df)] if eval_only else [("train", train_df), ("eval", eval_df)]
    for split_name, annot_df in splits:
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} set for {feature_name}")
        print(f"{'='*80}")

        chunk_df = annot_df.copy()

        # Prepare arguments for concurrent processing
        chunk_args = []
        for i, (chunk, found_kw) in enumerate(
            zip(chunk_df["chunk"], chunk_df["found_keywords"])
        ):
            formatted_prompt = format_prompt_with_keyword(prompt_text, found_kw)
            chunk_args.append((i, chunk, found_kw, target_cls, inference_type, formatted_prompt))

        # Process chunks concurrently with progress bar
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {
                executor.submit(process_single_chunk, args): args[0] for args in chunk_args
            }

            with tqdm(total=len(chunk_args), desc=f"Processing chunks ({split_name})") as pbar:
                for future in as_completed(future_to_index):
                    i, preds_for_chunk = future.result()

                    for key, value in preds_for_chunk.items():
                        if key not in chunk_df.columns:
                            chunk_df[key] = pd.NA
                        chunk_df.loc[chunk_df.index[i], key] = value

                    pbar.update(1)

        # Calculate validation metrics
        ground_truth = chunk_df["val_unified"]
        pred_column = feature_name
        is_numeric_feature = issubclass(target_cls, PtNumericFeatureBase)

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

        # Separate natural and synthetic data
        natural_df = chunk_df[chunk_df["is_synthetic"] == False]
        synthetic_df = chunk_df[chunk_df["is_synthetic"] == True]

        predictions = chunk_df[pred_column]
        natural_predictions = natural_df[pred_column] if len(natural_df) > 0 else pd.Series(dtype=object)
        synthetic_predictions = synthetic_df[pred_column] if len(synthetic_df) > 0 else pd.Series(dtype=object)
        natural_ground_truth = natural_df["val_unified"] if len(natural_df) > 0 else pd.Series(dtype=object)
        synthetic_ground_truth = synthetic_df["val_unified"] if len(synthetic_df) > 0 else pd.Series(dtype=object)

        correct_combined = (predictions == ground_truth).sum()
        total_combined = len(predictions)
        accuracy_combined = correct_combined / total_combined if total_combined > 0 else 0

        correct_natural = (natural_predictions == natural_ground_truth).sum() if len(natural_df) > 0 else 0
        total_natural = len(natural_df)
        accuracy_natural = correct_natural / total_natural if total_natural > 0 else 0

        correct_synthetic = (synthetic_predictions == synthetic_ground_truth).sum() if len(synthetic_df) > 0 else 0
        total_synthetic = len(synthetic_df)
        accuracy_synthetic = correct_synthetic / total_synthetic if total_synthetic > 0 else 0

        split_result = {
            "combined": accuracy_combined,
            "natural": accuracy_natural,
            "synthetic": accuracy_synthetic,
        }

        if is_numeric_feature:
            pct_errors = [compute_numeric_pct_error(p, g) for p, g in zip(predictions, ground_truth)]
            avg_pct_error = sum(pct_errors) / len(pct_errors) if pct_errors else 0.0
            split_result["avg_pct_error"] = avg_pct_error

            abs_errors = [compute_numeric_abs_error(p, g) for p, g in zip(predictions, ground_truth)]
            abs_errors_valid = [e for e in abs_errors if e is not None]
            avg_abs_error = sum(abs_errors_valid) / len(abs_errors_valid) if abs_errors_valid else None
            split_result["avg_abs_error"] = avg_abs_error

        # Collect errors for summary
        error_mask = predictions != ground_truth
        split_result["errors"] = [
            {"pred": str(predictions.iloc[i]), "gt": str(ground_truth.iloc[i]), "chunk": str(chunk_df["chunk"].iloc[i])[:200]}
            for i in range(len(predictions)) if error_mask.iloc[i]
        ]

        results_by_split[split_name] = split_result

        print(f"Combined accuracy: {accuracy_combined:.3f} ({correct_combined}/{total_combined})")
        if is_numeric_feature:
            print(f"Avg percent error: {avg_pct_error:.1f}%")
            if avg_abs_error is not None:
                print(f"Avg absolute error: {avg_abs_error:.1f} days")
        if total_natural > 0:
            print(f"Natural accuracy: {accuracy_natural:.3f} ({correct_natural}/{total_natural})")
        if total_synthetic > 0:
            print(f"Synthetic accuracy: {accuracy_synthetic:.3f} ({correct_synthetic}/{total_synthetic})")

    return {
        "feature_name": feature_name,
        "train_processed_chunks": 0 if eval_only else len(train_df),
        "train_natural_chunks": 0 if eval_only else train_natural_count,
        "train_synthetic_chunks": 0 if eval_only else train_synthetic_count,
        "eval_processed_chunks": len(eval_df),
        "eval_natural_chunks": eval_natural_count,
        "eval_synthetic_chunks": eval_synthetic_count,
        "results_by_split": results_by_split,
    }


def main():
    args = parse_args()
    inference_type = args.inference_type
    downsample = args.downsample
    target_features = args.features

    print(f"Using inference type: {inference_type}")
    if downsample is not None:
        print(f"Downsampling to {downsample} examples per feature")
    if target_features is not None:
        print(f"Running on specified features: {target_features}")
    else:
        print("Running on all available features")

    print("\nLoading baseline prompts from Excel file...")
    baseline_prompts = load_baseline_prompts()
    print(f"Loaded baseline prompts for {len(baseline_prompts)} features")
    for feature in baseline_prompts:
        print(f"  {feature}: {baseline_prompts[feature][:80]}...")

    labeled_data_dir = Path(f"labeled_data/{args.data_source}")

    preds_dir = Path("preds")
    model_dir = preds_dir / model_id.replace("/", "_")
    timestamp_dir = model_dir / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Get all xlsx files from subdirectories in val_ds_annotated
    excel_files = []
    for subdir in labeled_data_dir.iterdir():
        if subdir.is_dir():
            chunk_files = list(subdir.glob("*_chunks.xlsx"))
            excel_files.extend(chunk_files)

    excel_files = sorted(excel_files)

    # Build list of (feature_name, file_path) pairs to process
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
                print(f"Skipping {feat_name} (no data file found for source feature {source_feature})")

        print(f"Filtered to {len(features_to_process)} features: {[f[0] for f in features_to_process]}")
    else:
        for file_path in excel_files:
            feature_name = file_path.stem.replace("_chunks", "")
            features_to_process.append((feature_name, file_path))

    all_results = {}
    accuracy_tracking = {}

    for feature_name, file_path in features_to_process:
        print(f"\n{'='*80}")
        print(f"Processing {feature_name}...")
        print(f"{'='*80}")
        if "cancer_date_frequency" in feature_name:
            continue
        results = process_file(file_path, inference_type, downsample, data_source=args.data_source, baseline_prompts=baseline_prompts, feature_name_override=feature_name, eval_only=args.eval_only)
        if results:
            all_results[feature_name] = results
            accuracy_tracking[feature_name] = {
                split_name: results["results_by_split"][split_name]["combined"]
                for split_name in results["results_by_split"]
            }

    # Write accuracy to JSONL file
    jsonl_path = timestamp_dir / "accuracy_over_time.jsonl"
    with open(jsonl_path, "w") as f:
        json.dump(accuracy_tracking, f, indent=2)
    print(f"\nAccuracy saved to: {jsonl_path}")

    # Print and write overall summary
    print("\n" + "="*80)
    print("Overall Summary:")
    print("="*80)
    summary_path = timestamp_dir / "overall_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Overall Summary:\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Inference Type: {inference_type}\n")
        f.write(f"Prompt Source: Dorsa's baseline prompts from {PROMPT_HISTORY_PATH}\n")
        print(f"Model ID: {model_id}")
        print(f"Inference Type: {inference_type}")
        print(f"Prompt Source: Dorsa's baseline prompts from {PROMPT_HISTORY_PATH}")

        for feature, results in all_results.items():
            summary_line = (
                f"\n{feature}:\n"
                f"Train: {results['train_processed_chunks']} chunks "
                f"({results['train_natural_chunks']} natural, {results['train_synthetic_chunks']} synthetic)\n"
                f"Eval: {results['eval_processed_chunks']} chunks "
                f"({results['eval_natural_chunks']} natural, {results['eval_synthetic_chunks']} synthetic)"
            )
            print(summary_line)
            f.write(summary_line)

            for split_name, split_result in results["results_by_split"].items():
                acc_line = f"\n  {split_name.upper()}: Combined={split_result['combined']:.3f}, Natural={split_result['natural']:.3f}, Synthetic={split_result['synthetic']:.3f}"
                if "avg_pct_error" in split_result:
                    acc_line += f", AvgPctError={split_result['avg_pct_error']:.1f}%"
                if split_result.get("avg_abs_error") is not None:
                    acc_line += f", AvgAbsError={split_result['avg_abs_error']:.1f}days"
                print(acc_line)
                f.write(acc_line)

                # Show first 5 errors on eval set
                if split_name == "eval":
                    errors = split_result.get("errors", [])
                    if errors:
                        error_header = f"\n  First {min(5, len(errors))} eval errors:"
                        print(error_header)
                        f.write(error_header)
                        for err in errors[:5]:
                            err_line = f"\n    pred={err['pred']}, gt={err['gt']}, chunk={err['chunk']}"
                            print(err_line)
                            f.write(err_line)

            f.write("\n")
            print()


if __name__ == "__main__":
    main()
