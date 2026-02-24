import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import yaml
from datetime import datetime
import argparse
import json

from pt_features import PtFeaturesMeta, PtFeatureBase, PtDateFeatureBase
from models import MrModel, DummyModel
from utils import get_dataset
import re

tqdm.pandas()  # Enable tqdm for pandas operations

PROMPT_HISTORY_PATH = "labeled_data/LLM Adjustment Tracking.xlsx"

# Load MODEL_ID from config.yml
def load_model_id():
    try:
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            return config["model"]["id"]
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"Error loading config.yml: {e}")
        raise ValueError("Could not load model ID from config.yml")


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Compare predictions to ground truth")
    parser.add_argument(
        "--inference_type",
        choices=["logit", "cot"],
        default="cot",
        help="Type of inference to use: logit (default) or cot (chain of thought)",
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
    return parser.parse_args()


# Load model ID and initialize model
model_id = load_model_id()
model = MrModel(model_id=model_id)

# Generate timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_prompt_history():
    """Load prompt history from Excel file and return prompts organized by feature and tuner"""
    df = pd.read_excel(PROMPT_HISTORY_PATH)
    
    # Filter for James's rows (Prompt Tuner == 'JC') and Dorsa's rows (Prompt Tuner == 'DM')
    james_df = df[df['Prompt Tuner'] == 'JC'].copy()
    dorsa_df = df[df['Prompt Tuner'] == 'DM'].copy()
    
    # Get all prompt columns in order
    prompt_cols = [col for col in df.columns if col.startswith('Prompt #')]
    prompt_cols = sorted(prompt_cols, key=lambda x: int(re.search(r'#(\d+)', x).group(1)) if re.search(r'#(\d+)', x) else 0)
    
    # Organize prompts by feature and tuner
    # Structure: prompts_by_feature[feature_name][tuner] = [list of prompts]
    prompts_by_feature = {}
    
    def process_tuner_df(tuner_df, tuner_name):
        for _, row in tuner_df.iterrows():
            feature_name = row['Data Point']
            if pd.isna(feature_name):
                continue
            
            if feature_name not in prompts_by_feature:
                prompts_by_feature[feature_name] = {}
            
            feature_prompts = []
            for prompt_col in prompt_cols:
                prompt_text = row[prompt_col]
                if pd.notna(prompt_text) and str(prompt_text).strip():
                    # Extract prompt number from column name
                    prompt_num_match = re.search(r'#(\d+)', prompt_col)
                    prompt_num = int(prompt_num_match.group(1)) if prompt_num_match else len(feature_prompts) + 1
                    feature_prompts.append({
                        'number': prompt_num,
                        'text': str(prompt_text).strip(),
                        'tuner': tuner_name
                    })
            
            if feature_prompts:
                prompts_by_feature[feature_name][tuner_name] = feature_prompts
    
    process_tuner_df(james_df, 'james')
    process_tuner_df(dorsa_df, 'dorsa')
    
    return prompts_by_feature


def format_prompt_with_keyword(prompt_text, keyword):
    """Replace placeholders in prompt text with actual keyword"""
    # Replace {keyword} and {abx} with the actual keyword
    formatted = prompt_text.replace('{keyword}', keyword)
    formatted = formatted.replace('{abx}', keyword)
    return formatted


def process_single_chunk(args):
    """Process a single chunk and return the result with index"""
    i, chunk, found_kw, target_cls, inference_type, custom_query = args
    preds_for_chunk = {}

    # Use the forward method to handle the boilerplate logic
    kwargs = {}
    if custom_query:
        kwargs['custom_query'] = custom_query
    
    pred_dict = target_cls.forward(
        model=model, chunk=chunk, keyword=found_kw, inference_type=inference_type, **kwargs
    )
    preds_for_chunk.update({k: v for k, v in pred_dict.items()})

    return i, preds_for_chunk


def generate_error_analysis_md(feature_name, target_cls, chunk_df, ground_truth, pred_columns, feature_dir):
    """Generate markdown file with error analysis for a feature"""
    
    # Get the prompt/query text
    try:
        # Try to get query with a sample keyword
        sample_keyword = target_cls.keywords[0] if hasattr(target_cls, 'keywords') and target_cls.keywords else "sample"
        query_text = target_cls.query(keyword=sample_keyword) if hasattr(target_cls, 'query') else "No query method found"
    except Exception as e:
        query_text = f"Error getting query: {str(e)}"
    
    # Find the main prediction column (usually the feature name itself)
    main_pred_col = None
    for col in pred_columns:
        if col == feature_name or col.lower() == feature_name.lower():
            main_pred_col = col
            break
    
    if main_pred_col is None and pred_columns:
        main_pred_col = pred_columns[0]  # Use first prediction column if no exact match
    
    if main_pred_col is None:
        print(f"Warning: No prediction column found for {feature_name}")
        return
    
    predictions = chunk_df[main_pred_col]
    
    # Find incorrect predictions
    incorrect_mask = predictions != ground_truth
    incorrect_df = chunk_df[incorrect_mask].copy()
    
    if len(incorrect_df) == 0:
        print(f"No errors found for {feature_name}")
        return
    
    # Generate markdown content
    md_content = f"# Error Analysis for {feature_name}\n\n"
    md_content += f"## Prompt/Query\n\n```\n{query_text}\n```\n\n"
    md_content += f"## Summary\n\n"
    md_content += f"- Total examples: {len(chunk_df)}\n"
    md_content += f"- Incorrect predictions: {len(incorrect_df)}\n"
    md_content += f"- Accuracy: {((predictions == ground_truth).sum() / len(predictions) * 100):.1f}%\n\n"
    md_content += f"## Incorrect Predictions\n\n"
    
    for idx, row in incorrect_df.iterrows():
        md_content += f"### Example {idx + 1}\n\n"
        md_content += f"**Real Answer:** {row['val_unified']}\n\n"
        md_content += f"**Prediction:** {row[main_pred_col]}\n\n"
        md_content += f"**Keyword:** {row['found_keywords']}\n\n"
        md_content += f"**Chunk Text:**\n```\n{row['chunk']}\n```\n\n"
        md_content += "---\n\n"
    
    # Save markdown file
    md_file_path = feature_dir / f"{feature_name}_errors.md"
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Generated error analysis: {md_file_path}")


def process_file(file_path, inference_type, downsample=None, data_source=None, prompts_by_feature=None):
    print(f"\nProcessing {file_path}")
    feature_name = file_path.stem.replace("_chunks", "")

    # Create feature directory structure: preds/model_id/timestamp/feature_name
    preds_dir = Path("preds")
    model_dir = preds_dir / model_id.replace(
        "/", "_"
    )  # Replace / with _ for valid directory names
    timestamp_dir = model_dir / timestamp
    feature_dir = timestamp_dir / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Use get_dataset to load and split the data
    datasets = get_dataset(
        data_source=data_source or "mgb",
        feature_names=[feature_name],
        train_split=0.5,  # 50% for eval, 50% for train
        downsample=downsample,
        random_state=42
    )
    
    if feature_name not in datasets:
        raise ValueError(f"Dataset not found for feature: {feature_name}")
    
    # Get train and eval datasets
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

    target_cls = PtFeaturesMeta.registry[feature_name]

    # Get prompts for this feature (organized by tuner)
    feature_prompts_by_tuner = prompts_by_feature[feature_name]

    # Store accuracy over time for this feature, organized by tuner and split
    accuracy_over_time_by_tuner = {}

    # Process both train and eval sets
    for split_name, annot_df in [("train", train_df), ("eval", eval_df)]:
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} set for {feature_name}")
        print(f"{'='*80}")
        
        # Iterate through each tuner (james, dorsa)
        for tuner_name in sorted(feature_prompts_by_tuner.keys()):
            feature_prompts = feature_prompts_by_tuner[tuner_name]
            print(f"\nFound {len(feature_prompts)} prompt versions for {feature_name} by {tuner_name} ({split_name})")
            
            if split_name not in accuracy_over_time_by_tuner:
                accuracy_over_time_by_tuner[split_name] = {}
            
            accuracy_over_time = []
            
            # Iterate through all prompts in order for this tuner
            for prompt_info in feature_prompts:
                prompt_num = prompt_info['number']
                prompt_text = prompt_info['text']
                
                print(f"\nEvaluating Prompt #{prompt_num} for {feature_name} by {tuner_name} ({split_name})...")

                # Use the annot_df directly as chunk_df since it contains both chunks and ground truth
                chunk_df = annot_df.copy()

                # Prepare arguments for concurrent processing with custom query
                chunk_args = []
                for i, (chunk, found_kw) in enumerate(
                    zip(chunk_df["chunk"], chunk_df["found_keywords"])
                ):
                    # Format prompt with actual keyword
                    formatted_prompt = format_prompt_with_keyword(prompt_text, found_kw)
                    chunk_args.append((i, chunk, found_kw, target_cls, inference_type, formatted_prompt))

                # Process chunks concurrently with progress bar
                with ThreadPoolExecutor(max_workers=100) as executor:
                    # Submit all tasks
                    future_to_index = {
                        executor.submit(process_single_chunk, args): args[0] for args in chunk_args
                    }

                    # Process completed tasks with progress bar
                    with tqdm(total=len(chunk_args), desc=f"Processing chunks (Prompt #{prompt_num}, {tuner_name}, {split_name})") as pbar:
                        for future in as_completed(future_to_index):
                            i, preds_for_chunk = future.result()

                            # Add new columns as needed and assign values
                            for key, value in preds_for_chunk.items():
                                if key not in chunk_df.columns:
                                    chunk_df[key] = pd.NA
                                chunk_df.loc[chunk_df.index[i], key] = value

                            pbar.update(1)

                # Calculate validation metrics
                ground_truth = chunk_df["val_unified"]
                
                # The prediction column is the feature class name (same as feature_name)
                pred_column = feature_name

                # Separate natural and synthetic data
                natural_df = chunk_df[chunk_df["is_synthetic"] == False]
                synthetic_df = chunk_df[chunk_df["is_synthetic"] == True]
                
                # Calculate accuracy for combined, natural, and synthetic
                predictions = chunk_df[pred_column]
                natural_predictions = natural_df[pred_column] if len(natural_df) > 0 else pd.Series(dtype=object)
                synthetic_predictions = synthetic_df[pred_column] if len(synthetic_df) > 0 else pd.Series(dtype=object)
                natural_ground_truth = natural_df["val_unified"] if len(natural_df) > 0 else pd.Series(dtype=object)
                synthetic_ground_truth = synthetic_df["val_unified"] if len(synthetic_df) > 0 else pd.Series(dtype=object)
                
                # Combined stats
                correct_combined = (predictions == ground_truth).sum()
                total_combined = len(predictions)
                accuracy_combined = correct_combined / total_combined if total_combined > 0 else 0
                
                # Natural stats
                correct_natural = (natural_predictions == natural_ground_truth).sum() if len(natural_df) > 0 else 0
                total_natural = len(natural_df)
                accuracy_natural = correct_natural / total_natural if total_natural > 0 else 0
                
                # Synthetic stats
                correct_synthetic = (synthetic_predictions == synthetic_ground_truth).sum() if len(synthetic_df) > 0 else 0
                total_synthetic = len(synthetic_df)
                accuracy_synthetic = correct_synthetic / total_synthetic if total_synthetic > 0 else 0

                # Store accuracy for this prompt version
                accuracy_over_time.append({
                    "prompt_number": prompt_num,
                    "combined": accuracy_combined,
                    "natural": accuracy_natural,
                    "synthetic": accuracy_synthetic
                })

                print(f"Prompt #{prompt_num} ({tuner_name}, {split_name}) - Combined accuracy: {accuracy_combined:.3f} ({correct_combined}/{total_combined})")
                if total_natural > 0:
                    print(f"Prompt #{prompt_num} ({tuner_name}, {split_name}) - Natural accuracy: {accuracy_natural:.3f} ({correct_natural}/{total_natural})")
                if total_synthetic > 0:
                    print(f"Prompt #{prompt_num} ({tuner_name}, {split_name}) - Synthetic accuracy: {accuracy_synthetic:.3f} ({correct_synthetic}/{total_synthetic})")
            
            accuracy_over_time_by_tuner[split_name][tuner_name] = accuracy_over_time

    # Return results with accuracy over time organized by split and tuner
    return {
        "feature_name": feature_name,
        "train_processed_chunks": len(train_df),
        "train_natural_chunks": train_natural_count,
        "train_synthetic_chunks": train_synthetic_count,
        "eval_processed_chunks": len(eval_df),
        "eval_natural_chunks": eval_natural_count,
        "eval_synthetic_chunks": eval_synthetic_count,
        "accuracy_over_time_by_split_and_tuner": accuracy_over_time_by_tuner
    }


def main():
    # Parse command line arguments
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

    # Load prompt history
    print("\nLoading prompt history from Excel file...")
    prompts_by_feature = load_prompt_history()
    print(f"Loaded prompts for {len(prompts_by_feature)} features")
    for feature, prompts_by_tuner in prompts_by_feature.items():
        tuner_info = ", ".join([f"{tuner}: {len(prompts)} versions" for tuner, prompts in prompts_by_tuner.items()])
        print(f"  {feature}: {tuner_info}")

    labeled_data_dir = Path(f"labeled_data/{args.data_source}")

    # Create the main preds directory structure: preds/model_id/timestamp
    preds_dir = Path("preds")
    model_dir = preds_dir / model_id.replace(
        "/", "_"
    )  # Replace / with _ for valid directory names
    timestamp_dir = model_dir / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Get all xlsx files from subdirectories in val_ds_annotated
    excel_files = []
    for subdir in labeled_data_dir.iterdir():
        if subdir.is_dir():
            chunk_files = list(subdir.glob("*_chunks.xlsx"))
            excel_files.extend(chunk_files)

    excel_files = sorted(excel_files)
    
    # Filter features if specified
    if target_features is not None:
        # Convert target_features to a set for faster lookup
        target_features_set = set(target_features)
        # Filter excel_files to only include specified features
        filtered_files = []
        for file_path in excel_files:
            feature_name = file_path.stem.replace("_chunks", "")
            if feature_name in target_features_set:
                filtered_files.append(file_path)
            else:
                print(f"Skipping {feature_name} (not in target features list)")
        
        excel_files = filtered_files
        print(f"Filtered to {len(excel_files)} features: {[f.stem.replace('_chunks', '') for f in excel_files]}")

    all_results = {}
    accuracy_tracking = {}  # Track accuracy over time per feature

    for file_path in excel_files:
        feature_name = file_path.stem.replace("_chunks", "")
        print(f"\n{'='*80}")
        print(f"Processing {file_path.name}...")
        print(f"{'='*80}")
        if "cancer_date_frequency" in file_path.name:
            continue
        results = process_file(file_path, inference_type, downsample, data_source=args.data_source, prompts_by_feature=prompts_by_feature)
        if results:
            all_results[feature_name] = results
            # Store accuracy over time for JSONL output
            accuracy_tracking[feature_name] = {}
            for split_name in ["train", "eval"]:
                if split_name in results["accuracy_over_time_by_split_and_tuner"]:
                    accuracy_tracking[feature_name][split_name] = {}
                    for tuner_name, accuracy_list in results["accuracy_over_time_by_split_and_tuner"][split_name].items():
                        accuracy_tracking[feature_name][split_name][tuner_name] = [acc["combined"] for acc in accuracy_list]

    # Write accuracy over time to JSONL file
    jsonl_path = timestamp_dir / "accuracy_over_time.jsonl"
    with open(jsonl_path, "w") as f:
        json.dump(accuracy_tracking, f, indent=2)
    print(f"\nAccuracy over time saved to: {jsonl_path}")
    print(f"Format: {{feature_name: {{'train': {{'james': [...], 'dorsa': [...]}}, 'eval': {{'james': [...], 'dorsa': [...]}}}}}}")

    # Print and write overall summary
    print("\n" + "="*80)
    print("Overall Summary:")
    print("="*80)
    summary_path = timestamp_dir / "overall_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Overall Summary:\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Inference Type: {inference_type}\n")
        f.write(f"Prompt Source: James's and Dorsa's prompts from {PROMPT_HISTORY_PATH}\n")
        print(f"Model ID: {model_id}")
        print(f"Inference Type: {inference_type}")
        print(f"Prompt Source: James's and Dorsa's prompts from {PROMPT_HISTORY_PATH}")
        
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

            # Add accuracy over time for each prompt version by split and tuner
            for split_name in ["train", "eval"]:
                if split_name not in results["accuracy_over_time_by_split_and_tuner"]:
                    continue
                    
                f.write(f"\n\n{split_name.upper()} Accuracy Over Time (by Prompt Version and Tuner):")
                print(f"\n{split_name.upper()} Accuracy Over Time (by Prompt Version and Tuner):")
                for tuner_name in sorted(results["accuracy_over_time_by_split_and_tuner"][split_name].keys()):
                    f.write(f"\n  {tuner_name.upper()}:")
                    print(f"\n  {tuner_name.upper()}:")
                    for acc_info in results["accuracy_over_time_by_split_and_tuner"][split_name][tuner_name]:
                        prompt_num = acc_info["prompt_number"]
                        combined_acc = acc_info["combined"]
                        natural_acc = acc_info["natural"]
                        synthetic_acc = acc_info["synthetic"]
                        
                        acc_line = f"    Prompt #{prompt_num}: Combined={combined_acc:.3f}, Natural={natural_acc:.3f}, Synthetic={synthetic_acc:.3f}"
                        print(acc_line)
                        f.write(f"\n{acc_line}")
            
            f.write("\n")
            print()


if __name__ == "__main__":
    main()
