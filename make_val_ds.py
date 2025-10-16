# go through each target feature and get N Report_Texts that contain any keyword from the feature. 
# save the relevant data to a txt file named along with the feature name

from pt_features import PtFeaturesMeta, PtFeatureBase, PtDateFeatureBase
from utils import *
from sqlalchemy import create_engine, text
from make_db import db_url
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
from random import randint
import argparse
tqdm.pandas()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate validation dataset for patient features')
parser.add_argument('--data_source', type=str, default='mimic', 
                    help='Data source table name (default: vis)')
parser.add_argument('--synthetic_keywords', action='store_true', default=False,
                    help='Whether to use synthetic keywords')
args = parser.parse_args()

assert args.data_source in ["mgb", "mimic"], "Invalid data source"

db = create_engine(db_url)
BATCH_SIZE = 1000
MAX_DS_SIZE = 20 if args.synthetic_keywords else 200



print(f"Using data source: {args.data_source}")

for cls in PtFeaturesMeta.registry.values():
    # if cls.__name__ not in ["antibiotic_duration"]:
    #     continue
    # if not issubclass(cls, PtDateFeatureBase): # temp, to fill in date features we didn't do yet
        # continue
    if not cls.val_var:
        continue
    if args.synthetic_keywords and not hasattr(cls, "synthetic_keywords"):
        continue

    print(f"Checking {cls.__name__} for keywords...")

    # Create feature directory structure within val_ds
    val_ds_dir = Path("val_ds")
    data_source_dir = f"{args.data_source}_synth" if args.synthetic_keywords else args.data_source
    feature_dir = val_ds_dir / data_source_dir / cls.__name__
    feature_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    offset = 0
    batch_size = 10000  # Initial batch size for querying
    total_processed = 0
    seen_empis = set()  # Track unique EMPIs
    
    if args.data_source == "mgb":
        while len(seen_empis) < MAX_DS_SIZE:
            with db.connect() as conn:
                query = text(f"SELECT * FROM vis ORDER BY SUBSTR(EMPI, -5) LIMIT {batch_size} OFFSET {offset}")
                result = conn.execute(query)
                df_batch = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                # If no more records, break
                if len(df_batch) == 0:
                    break
                    
                # Shuffle the batch for pseudorandomization
                df_batch = df_batch.sample(frac=1)
            
            # Process this batch in smaller chunks
            for start in tqdm(range(0, len(df_batch), BATCH_SIZE), desc=f"Processing batch at offset {offset}"):
                batch = df_batch.iloc[start:start+BATCH_SIZE].copy()
                batch["found_keywords"] = batch["Report_Text"].progress_apply(has_keyword, keywords=cls.keywords if not args.synthetic_keywords else cls.synthetic_keywords)
                filtered_batch = batch[batch["found_keywords"].apply(lambda x: len(x) > 0)]
                
                if len(filtered_batch) > 0:
                    # Add new EMPIs to our tracking set
                    new_empis = set(filtered_batch["EMPI"].unique())
                    seen_empis.update(new_empis)
                    dfs.append(filtered_batch)
                    
                    print(f"Found {len(new_empis)} new unique EMPIs. Total unique EMPIs: {len(seen_empis)}")
                
                if len(seen_empis) >= MAX_DS_SIZE:
                    break
            
            # Update offset for next query
            offset += batch_size
            
            # If we got fewer records than requested, we've reached the end
            if len(df_batch) < batch_size:
                break

            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        print(f"Final count: {len(seen_empis)} unique EMPIs processed")
    elif args.data_source == "mimic":
        # Load the full dataset
        df = pd.read_csv(f"physionet.org/files/mimic-iv-note/2.2/note/discharge.csv")
        #     output_cols = ["EMPI", "EPIC_PMRN", "MRN", "Report_Number", "Report_Date_Time", "chunk", "found_keywords"]
        df = df.rename(columns={"text": "Report_Text", "subject_id": "EMPI", "note_id": "MRN", "note_seq": "Report_Number", "charttime": "Report_Date_Time"})
        df["EPIC_PMRN"] = 0
        
        # Shuffle the entire dataset first
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Process in batches of 10,000
        batch_size = 1000
        dfs = []
        seen_empis = set()  # Track unique EMPIs
        total_processed = 0
        
        # Create progress bar for total dataset processing
        pbar = tqdm(total=len(df), desc="Processing MIMIC data", unit="records")
        
        for start in range(0, len(df), batch_size):
            if len(seen_empis) >= MAX_DS_SIZE:
                break
                
            # Get batch of data
            batch_df = df.iloc[start:start+batch_size].copy()
            
            # Check for keywords in this batch
            batch_df["found_keywords"] = batch_df["Report_Text"].progress_apply(has_keyword, keywords=cls.keywords if not args.synthetic_keywords else cls.synthetic_keywords)
            filtered_batch = batch_df[batch_df["found_keywords"].apply(lambda x: len(x) > 0)]
            
            if len(filtered_batch) > 0:
                # Add new EMPIs to our tracking set
                new_empis = set(filtered_batch["EMPI"].unique())
                seen_empis.update(new_empis)
                dfs.append(filtered_batch)
            
            total_processed += len(batch_df)
            pbar.update(len(batch_df))
            
            if len(seen_empis) >= MAX_DS_SIZE:
                break
        
        pbar.close()
        print(f"Found {len(seen_empis)} unique EMPIs from {total_processed} total records processed")
        
        # Combine all filtered dataframes
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        print(f"Final count: {len(seen_empis)} unique EMPIs processed from {total_processed} total records")
    else:
        raise ValueError(f"Invalid data source: {args.data_source}")

    # Process chunks similar to compare_to_gt.py
    print(f"Processing chunks for {cls.__name__}...")
    df = df.assign(chunk=df["Report_Text"].progress_apply(chunk_text))
    chunk_df = df.explode("chunk")
    chunk_df["found_keywords"] = chunk_df["chunk"].progress_apply(
        has_keyword, keywords=cls.keywords if not args.synthetic_keywords else cls.synthetic_keywords
    )
    # Drop rows where found_keywords is empty
    chunk_df = chunk_df[chunk_df["found_keywords"].apply(lambda x: len(x) > 0)]
    # Create MultiIndex from current index and found_keywords
    chunk_df = chunk_df.explode('found_keywords')
    chunk_df = chunk_df.set_index('found_keywords', append=True)
    # Rename 'found_keywords' to 'kw' throughout the relevant DataFrame
    chunk_df = chunk_df.rename(columns={'found_keywords': 'kw'})
    # Downsample to ensure exactly one chunk per unique ID, up to MAX_DS_SIZE total chunks
    # First, sample one chunk per unique ID
    chunk_df = chunk_df.groupby('EMPI').sample(n=1, random_state=42).reset_index(drop=False)
    
    # If we have more unique IDs than MAX_DS_SIZE, randomly sample MAX_DS_SIZE of them
    if len(chunk_df) > MAX_DS_SIZE:
        chunk_df = chunk_df.sample(n=MAX_DS_SIZE, random_state=42).sort_index()

    # Clean the DataFrame to remove illegal characters for CSV
    def clean_text_for_csv(text):
        if pd.isna(text) or not isinstance(text, str):
            return text
        # Replace problematic characters
        text = text.replace('\r', ' ').replace('\x0c', ' ').strip()
        return text

    # Apply cleaning to columns that contain all strings
    for col in chunk_df.columns:
        if chunk_df[col].dtype == 'object':
            # Check if all non-null values in the column are strings
            if chunk_df[col].notna().all() and all(isinstance(val, str) for val in chunk_df[col].dropna()):
                chunk_df[col] = chunk_df[col].apply(clean_text_for_csv)

    # save the chunk_df to a csv file in the feature directory
    # include cols: EMPI,EPIC_PMRN,MRN,Report_Number,Report_Date_Time,chunk,found_keywords
    output_cols = ["EMPI", "EPIC_PMRN", "MRN", "Report_Number", "Report_Date_Time", "chunk", "found_keywords"]
    available_cols = [col for col in output_cols if col in chunk_df.columns]
    # chunk_df[available_cols].to_csv(feature_dir / "chunks.csv", index=False)
    chunk_df["val_unified"] = ""

    # save the df to a csv file
    # include cols: EMPI,EPIC_PMRN,MRN,Report_Number,Report_Date_Time
    df = df[["EMPI", "EPIC_PMRN", "MRN", "Report_Number", "Report_Date_Time"]]
    chunk_df.to_excel(feature_dir / f"{cls.__name__}_chunks.xlsx")

    # print metadata to a md file
    with open(feature_dir / f"{cls.__name__}.md", "w") as f:
        f.write(f"# {cls.__name__}\n")
        f.write(f"## Query\n")
        f.write(f"Query function: {cls.query.__name__}\n" if hasattr(cls, "query") else "Query function: None\n")
        f.write(f"Example query: {cls.query(chunk='', keyword='{keyword}')}\n" if hasattr(cls, "query") else "Example query: None\n")
        f.write(f"## Keywords\n")
        f.write(f"{cls.keywords if not args.synthetic_keywords else cls.synthetic_keywords}\n")
        # f.write(f"## Number of records with keyword: {len(df[df['included_kw'].notna()])}\n")
        # f.write(f"## Number of records: {len(df)}\n")
        # f.write(f"## Number of chunks: {len(chunk_df)}\n")