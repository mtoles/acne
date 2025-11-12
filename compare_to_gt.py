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

from pt_features import PtFeaturesMeta, PtFeatureBase, PtDateFeatureBase
from models import MrModel, DummyModel
import re

tqdm.pandas()  # Enable tqdm for pandas operations


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
        default="logit",
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
    return parser.parse_args()


# Load model ID and initialize model
model_id = load_model_id()
model = MrModel(model_id=model_id)

# Generate timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def process_single_chunk(args):
    """Process a single chunk and return the result with index"""
    i, chunk, found_kw, target_cls, inference_type = args
    preds_for_chunk = {}

    # Use the forward method to handle the boilerplate logic
    pred_dict = target_cls.forward(
        model=model, chunk=chunk, keyword=found_kw, inference_type=inference_type
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


def process_file(file_path, inference_type, downsample=None):
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

    # Read the validation file directly from val_ds_annotated
    annot_df = pd.read_excel(file_path)
    annot_df = annot_df[annot_df["val_unified"].notna()]
    annot_df["val_unified"] = annot_df["val_unified"].astype(str)
    annot_df["Report_Number"] = annot_df["Report_Number"].astype(str)

    # Apply downsampling if specified
    total_rows = len(annot_df)
    if downsample is not None:
        # Downsample to the specified number of examples per feature
        if total_rows > downsample:
            annot_df = annot_df.sample(n=downsample, random_state=42).sort_index()
            print(f"Downsampled to {downsample} out of {total_rows} examples")
        else:
            print(f"Using all {total_rows} examples (less than downsampling target of {downsample})")
    else:
        # Default behavior: limit to first 50% of examples
        limit_rows = total_rows // 2
        annot_df = annot_df.head(limit_rows)
        print(f"Processing first {limit_rows} out of {total_rows} examples")

    target_cls = PtFeaturesMeta.registry[feature_name]

    # Use the annot_df directly as chunk_df since it contains both chunks and ground truth
    chunk_df = annot_df.copy()

    # Ensure a column exists to store the raw dict of predictions per row
    chunk_df["preds"] = None

    # Prepare arguments for concurrent processing
    chunk_args = [
        (i, chunk, found_kw, target_cls, inference_type)
        for i, (chunk, found_kw) in enumerate(
            zip(chunk_df["chunk"], chunk_df["found_keywords"])
        )
    ]

    # Process chunks concurrently with progress bar
    with ThreadPoolExecutor(max_workers=100) as executor:
    # with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_chunk, args): args[0] for args in chunk_args
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(chunk_args), desc="Processing chunks") as pbar:
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

    # Get the main prediction column (usually the first one from the feature class)
    pred_columns = [
        col
        for col in chunk_df.columns
        if col
        not in [
            "Unnamed: 0",
            "level_0",
            "EMPI",
            "EPIC_PMRN",
            "MRN_Type",
            "MRN",
            "Report_Number",
            "Report_Date_Time",
            "Report_Description",
            "Report_Status",
            "Report_Type",
            "Report_Text",
            "found_keywords",
            "chunk",
            "Dorsa",
            "James",
            "val_unified",
            "Comments",
            "preds",
        ]
    ]

    validation_results = {}
    for pred_col in pred_columns:
        predictions = chunk_df[pred_col]
        # Calculate accuracy
        correct = (predictions == ground_truth).sum()
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        validation_results[pred_col] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    # Generate error analysis markdown file
    generate_error_analysis_md(feature_name, target_cls, chunk_df, ground_truth, pred_columns, feature_dir)

    # Clean the DataFrame to remove illegal characters for Excel
    def clean_text_for_excel(text):
        if pd.isna(text) or not isinstance(text, str):
            return text
        # Replace problematic characters
        text = text.replace("\r", " ").replace("\x0c", " ").strip()
        return text

    # Apply cleaning to columns that contain all strings
    for col in chunk_df.columns:
        if chunk_df[col].dtype == "object":
            chunk_df[col] = chunk_df[col].apply(clean_text_for_excel)

    # Save the processed chunk_df
    chunk_df.to_excel(feature_dir / "chunk_df.xlsx")

    print(f"Processed {len(chunk_df)} chunks for {feature_name}")
    return {"processed_chunks": len(chunk_df), "validation_results": validation_results}


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

    labeled_data_dir = Path("labeled_data/mgb")

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

    for file_path in excel_files:
        print(f"\nProcessing {file_path.name}...")
        results = process_file(file_path, inference_type, downsample)
        all_results[file_path.stem.replace("_chunks", "")] = results

    # Print and write overall summary
    print("\nOverall Summary:")
    summary_path = timestamp_dir / "overall_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Overall Summary:\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Inference Type: {inference_type}\n")
        print(f"Model ID: {model_id}")
        print(f"Inference Type: {inference_type}")
        for feature, results in all_results.items():
            summary_line = (
                f"\n{feature}:\nProcessed chunks: {results['processed_chunks']}"
            )
            print(summary_line)
            f.write(summary_line)

            # Add validation results
            if "validation_results" in results:
                f.write("\nValidation Results:")
                print("Validation Results:")
                for pred_col, metrics in results["validation_results"].items():
                    val_line = f"  {pred_col}: {metrics['accuracy']:.3f} accuracy ({metrics['correct']}/{metrics['total']})"
                    print(val_line)
                    f.write(f"\n{val_line}")
            f.write("\n")
            print()


if __name__ == "__main__":
    main()
