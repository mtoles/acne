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
tqdm.pandas()

db = create_engine(db_url)
BATCH_SIZE = 1000
MAX_DS_SIZE = 100



for cls in PtFeaturesMeta.registry.values():
    # if not issubclass(cls, PtDateFeatureBase): # temp, to fill in date features we didn't do yet
        # continue
    if not cls.val_var:
        continue
    if not cls.__name__ == "antibiotic_duration":
        print(f"Skipping {cls.__name__}")
        continue
    # check each record if it has the keywords
    print(f"Checking {cls.__name__} for keywords...")

    # Create feature directory structure within val_ds
    val_ds_dir = Path("val_ds")
    feature_dir = val_ds_dir / cls.__name__
    feature_dir.mkdir(parents=True, exist_ok=True)

    with db.connect() as conn:
        query = text(f"SELECT * FROM vis ORDER BY RANDOM() LIMIT 10000")
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys()).sample(frac=1)

    dfs = []
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[start:start+BATCH_SIZE].copy()
        batch["found_keywords"] = batch["Report_Text"].progress_apply(has_keyword, keywords=cls.keywords)
        batch["has_kw"] = batch["found_keywords"].apply(lambda x: len(x) > 0)
        batch = batch[batch["has_kw"]]
        batch["included_kw"] = batch["found_keywords"]  # Use the found keywords directly
        dfs.append(batch)

        total_len = sum([len(x) for x in dfs])
        if total_len >= MAX_DS_SIZE:
            break

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=df.columns)
    df = df.iloc[:MAX_DS_SIZE]

    # Process chunks similar to compare_to_gt.py
    print(f"Processing chunks for {cls.__name__}...")
    df = df.assign(chunk=df["Report_Text"].progress_apply(chunk_text))
    chunk_df = df.explode("chunk")
    chunk_df["found_keywords"] = chunk_df["chunk"].progress_apply(
        has_keyword, keywords=cls.keywords
    )
    # Drop rows where found_keywords is empty
    chunk_df = chunk_df[chunk_df["found_keywords"].apply(lambda x: len(x) > 0)]
    # Create MultiIndex from current index and found_keywords
    chunk_df = chunk_df.explode('found_keywords')
    chunk_df = chunk_df.set_index('found_keywords', append=True)

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
    chunk_df[available_cols].to_csv(feature_dir / "chunks.csv", index=False)
    chunk_df["val_unified"] = ""

    # save the df to a csv file
    # include cols: EMPI,EPIC_PMRN,MRN,Report_Number,Report_Date_Time
    df = df[["EMPI", "EPIC_PMRN", "MRN", "Report_Number", "Report_Date_Time", "included_kw"]]
    chunk_df.to_excel(feature_dir / f"{cls.__name__}_chunks.xlsx")

    # print metadata to a md file
    with open(feature_dir / f"{cls.__name__}.md", "w") as f:
        f.write(f"# {cls.__name__}\n")
        f.write(f"## Query\n")
        f.write(f"Query function: {cls.query.__name__}\n" if hasattr(cls, "query") else "Query function: None\n")
        f.write(f"Example query: {cls.query(chunk='', keywords=cls.keywords)}\n" if hasattr(cls, "query") else "Example query: None\n")
        f.write(f"## Keywords\n")
        f.write(f"{cls.keywords}\n")
        f.write(f"## Number of records with keyword: {len(df[df['included_kw'].notna()])}\n")
        f.write(f"## Number of records: {len(df)}\n")
        f.write(f"## Number of chunks: {len(chunk_df)}\n")