import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm

from make_db import store_dir as file_store_parent_dir
from pt_features import PtFeaturesMeta, PtFeatureBase, PtDateFeatureBase
from utils import chunk_text, has_keyword
from models import MrModel, DummyModel
from sqlalchemy import create_engine, text
from make_db import db_url
import re

tqdm.pandas()  # Enable tqdm for pandas operations

# Initialize model
model = MrModel()

# get all null values from the db that are LlmFeatureBase
classes = [
    cls for cls in PtFeaturesMeta.registry.values() if issubclass(cls, PtFeatureBase)
]
db = create_engine(db_url)


def get_samples(column_name, annot_report_numbers, downsample_size=100):
    print(f"\nGetting samples for column {column_name}:")

    with db.connect() as conn:
        # Quote each report number and join them
        quoted_numbers = [f"'{num}'" for num in annot_report_numbers]
        # Get sample of rows, limited to downsample_size if specified
        base_query = f"SELECT DISTINCT * FROM vis WHERE Report_Number IN ({','.join(quoted_numbers)})"
        if downsample_size is not None:
            base_query += f" LIMIT {downsample_size}"
        sample_query = text(base_query)
        sample_result = conn.execute(sample_query)
        df = pd.DataFrame(sample_result.fetchall(), columns=sample_result.keys())
        print(f"Found {len(df)} unique samples")
        return df


def process_file(file_path):
    print(f"\nProcessing {file_path}")
    feature_name = file_path.stem #.replace("_validation", "")

    # Create feature directory structure within preds
    preds_dir = Path("preds")
    feature_dir = preds_dir / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Read the validation file
    annot_df = pd.read_excel(file_path)
    annot_df["Report_Number"] = annot_df["Report_Number"].astype(str)
    # annot_df = annot_df.dropna(subset=["Report_Number", "val_unified"])
    annot_report_numbers = annot_df["Report_Number"].unique() # downsample here if you want

    target_cls = PtFeaturesMeta.registry[feature_name]

    # Read the chunks.xlsx file from val_ds directory
    val_ds_chunks_path = Path("val_ds") / feature_name / f"{feature_name}_chunks.xlsx"
    chunk_df = pd.read_excel(val_ds_chunks_path, index_col=[0, 1])
    
    # The Excel file has a MultiIndex with found_keywords as the second level
    # Reset the index to make found_keywords a regular column
    chunk_df = chunk_df.reset_index(level=1, drop=False)
    chunk_df = chunk_df.rename(columns={'level_1': 'found_keywords'})
    
    # Filter chunk_df to only include rows with Report_Number in annot_report_numbers
    # chunk_df = chunk_df[chunk_df['Report_Number'].astype(str).isin(annot_report_numbers)]
    
    # Ensure a column exists to store the raw dict of predictions per row
    chunk_df['preds'] = None
        
    for i, (chunk, found_kw) in enumerate(tqdm(zip(chunk_df["chunk"], chunk_df["included_kw"]))):
        preds_for_chunk = {}
        
        # Use the forward method to handle the boilerplate logic
        pred_dict = target_cls.forward(model=model, chunk=chunk, keyword=found_kw)
        preds_for_chunk.update({k: v for k, v in pred_dict.items()})
        
        # Add new columns as needed and assign values
        for key, value in preds_for_chunk.items():
            if key not in chunk_df.columns:
                chunk_df[key] = pd.NA
            chunk_df.loc[chunk_df.index[i], key] = value

        # # Store the raw dict in the 'preds' column
        # chunk_df.loc[chunk_df.index[i], 'preds'] = preds_for_chunk

    # Since we've already filtered to only include rows with keywords, 
    # we can directly assign the predictions
    chunk_df["chunk_pred"] = chunk_df['preds']
    
    # Drop unnecessary columns for the final output
    chunk_df = chunk_df.drop(columns=["Report_Text", "chunk_pred"])

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
    return {"processed_chunks": len(chunk_df)}


def main():
    labeled_data_dir = Path("labeled_data")
    output_dir = Path("preds")
    output_dir.mkdir(exist_ok=True)  # Create preds directory if it doesn't exist

    # Get all xlsx files, sort alphabetically, and filter out _preds files
    excel_files = sorted(
        [f for f in labeled_data_dir.glob("*.xlsx") if not f.stem.endswith("_preds")]
    )

    all_results = {}
    excel_files = [Path("labeled_data/antibiotic_duration.xlsx")]
    
    for file_path in excel_files:
        print(f"\nProcessing {file_path.name}...")
        results = process_file(file_path)
        all_results[file_path.stem] = results

    # Print and write overall summary
    print("\nOverall Summary:")
    summary_path = Path("preds") / "overall_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Overall Summary:\n")
        for feature, results in all_results.items():
            summary_line = f"\n{feature}:\nProcessed chunks: {results['processed_chunks']}"
            print(summary_line)
            f.write(summary_line)


if __name__ == "__main__":
    main()
