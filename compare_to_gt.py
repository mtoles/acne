import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import yaml

from pt_features import PtFeaturesMeta, PtFeatureBase, PtDateFeatureBase
from models import MrModel, DummyModel
import re

tqdm.pandas()  # Enable tqdm for pandas operations

# Load MODEL_ID from config.yml
def load_model_id():
    try:
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)
            return config['model']['id']
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"Error loading config.yml: {e}")
        raise ValueError("Could not load model ID from config.yml")

# Load model ID and initialize model
model_id = load_model_id()
model = MrModel(model_id=model_id)

def process_single_chunk(args):
    """Process a single chunk and return the result with index"""
    i, chunk, found_kw, target_cls = args
    preds_for_chunk = {}
    
    # Use the forward method to handle the boilerplate logic
    pred_dict = target_cls.forward(model=model, chunk=chunk, keyword=found_kw)
    preds_for_chunk.update({k: v for k, v in pred_dict.items()})
    
    return i, preds_for_chunk

def process_file(file_path):
    print(f"\nProcessing {file_path}")
    feature_name = file_path.stem.replace("_chunks", "")

    # Create feature directory structure within preds
    preds_dir = Path("preds")
    feature_dir = preds_dir / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Read the validation file directly from val_ds_annotated
    annot_df = pd.read_excel(file_path)
    annot_df = annot_df[annot_df["val_unified"].notna()]
    annot_df["Report_Number"] = annot_df["Report_Number"].astype(str)
    
    # Limit to first 50% of examples
    total_rows = len(annot_df)
    limit_rows = total_rows // 2
    annot_df = annot_df.head(limit_rows)
    
    print(f"Processing first {limit_rows} out of {total_rows} examples")

    target_cls = PtFeaturesMeta.registry[feature_name]

    # Use the annot_df directly as chunk_df since it contains both chunks and ground truth
    chunk_df = annot_df.copy()
    
    # Ensure a column exists to store the raw dict of predictions per row
    chunk_df['preds'] = None
    
    # Prepare arguments for concurrent processing
    chunk_args = [(i, chunk, found_kw, target_cls) 
                   for i, (chunk, found_kw) in enumerate(zip(chunk_df["chunk"], chunk_df["found_keywords"]))]
    
    # Process chunks concurrently with progress bar
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(process_single_chunk, args): args[0] 
                          for args in chunk_args}
        
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
    ground_truth = chunk_df['val_unified']
    
    # Get the main prediction column (usually the first one from the feature class)
    pred_columns = [col for col in chunk_df.columns if col not in ['Unnamed: 0', 'level_0', 'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number', 'Report_Date_Time', 'Report_Description', 'Report_Status', 'Report_Type', 'Report_Text', 'found_keywords', 'chunk', 'Dorsa', 'James', 'val_unified', 'Comments', 'preds']]
    
    validation_results = {}
    for pred_col in pred_columns:
        predictions = chunk_df[pred_col]
        # Calculate accuracy
        correct = (predictions == ground_truth).sum()
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        validation_results[pred_col] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    # Clean the DataFrame to remove illegal characters for Excel
    def clean_text_for_excel(text):
        if pd.isna(text) or not isinstance(text, str):
            return text
        # Replace problematic characters
        text = text.replace('\r', ' ').replace('\x0c', ' ').strip()
        return text

    # Apply cleaning to columns that contain all strings
    for col in chunk_df.columns:
        if chunk_df[col].dtype == 'object':
            chunk_df[col] = chunk_df[col].apply(clean_text_for_excel)

    # Save the processed chunk_df
    chunk_df.to_excel(feature_dir / "chunk_df.xlsx")
    
    print(f"Processed {len(chunk_df)} chunks for {feature_name}")
    return {
        "processed_chunks": len(chunk_df),
        "validation_results": validation_results
    }


def main():
    labeled_data_dir = Path("labeled_data/val_ds_annotated")
    output_dir = Path("preds")
    output_dir.mkdir(exist_ok=True)  # Create preds directory if it doesn't exist

    # Get all xlsx files from subdirectories in val_ds_annotated
    excel_files = []
    for subdir in labeled_data_dir.iterdir():
        if subdir.is_dir():
            chunk_files = list(subdir.glob("*_chunks.xlsx"))
            excel_files.extend(chunk_files)
    
    excel_files = sorted(excel_files)

    all_results = {}
    
    for file_path in excel_files:
        print(f"\nProcessing {file_path.name}...")
        results = process_file(file_path)
        all_results[file_path.stem.replace("_chunks", "")] = results

    # Print and write overall summary
    print("\nOverall Summary:")
    summary_path = Path("preds") / "overall_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Overall Summary:\n")
        f.write(f"Model ID: {model_id}\n")
        print(f"Model ID: {model_id}")
        for feature, results in all_results.items():
            summary_line = f"\n{feature}:\nProcessed chunks: {results['processed_chunks']}"
            print(summary_line)
            f.write(summary_line)
            
            # Add validation results
            if 'validation_results' in results:
                f.write("\nValidation Results:")
                print("Validation Results:")
                for pred_col, metrics in results['validation_results'].items():
                    val_line = f"  {pred_col}: {metrics['accuracy']:.3f} accuracy ({metrics['correct']}/{metrics['total']})"
                    print(val_line)
                    f.write(f"\n{val_line}")
            f.write("\n")
            print()


if __name__ == "__main__":
    main()