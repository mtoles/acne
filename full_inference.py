
from pt_features import *
from utils import *
from sqlalchemy import create_engine, text, inspect
from make_db import db_url
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
from random import randint
import argparse
import time
import re
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

### TODO: use the good queries we generated from testing

random.seed(42)

db = create_engine(db_url)
model_id = "Qwen/Qwen2.5-72B-Instruct-AWQ"
model = MrModel(model_id=model_id)


# Counter for total LLM calls
_total_llm_calls = 0

# Cache for prevalence distributions
_prevalence_cache = {}

def get_prevalence_distribution(feature_cls: PtFeatureBase):
    """Get prevalence distribution for a feature class from the Excel file."""
    feature_name = feature_cls.__name__
    
    if feature_name in _prevalence_cache:
        return _prevalence_cache[feature_name]
    
    excel_path = Path(f"labeled_data/mgb/{feature_name}/{feature_name}_chunks.xlsx")
    
    df = pd.read_excel(excel_path)
    
    # Filter out NaN and "DROP" values
    val_unified = df["val_unified"].dropna()
    val_unified = val_unified[val_unified != "DROP"]
    
    # Compute value counts
    value_counts = val_unified.value_counts()
    
    # Filter to only include values that are in feature_cls.options
    valid_value_counts = {k: v for k, v in value_counts.items() if k in feature_cls.options}
    
    # Create distribution dictionary
    distribution = {}
    total_count = sum(valid_value_counts.values())
    
    if total_count > 0:
        for option in feature_cls.options:
            if option in valid_value_counts:
                distribution[option] = valid_value_counts[option] / total_count
            else:
                distribution[option] = 0.0
    else:
        # If no valid values found, fall back to uniform
        distribution = {k: 1.0 / len(feature_cls.options) for k in feature_cls.options}
    
    _prevalence_cache[feature_name] = distribution
    return distribution

def call_llm(feature_cls: PtFeatureBase, chunk: str, keyword: str = None) -> str:
    global _total_llm_calls

    if DUMMY_LLM:
        if not issubclass(feature_cls, PtDateFeatureBase):
            # Sample according to prevalence distribution
            distribution = get_prevalence_distribution(feature_cls)
            options = list(distribution.keys())
            weights = list(distribution.values())
            pred = random.choices(options, weights=weights, k=1)[0]
        else:
            # return a date between 2000-01-01 and 2025-12-31
            year = random.randint(2000, 2025)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            pred = f"{year}{month:02d}{day:02d}"
    else:
        if keyword is not None:
            query = feature_cls.query(chunk=chunk, keyword=keyword)
        else:
            query = feature_cls.query(chunk=chunk)
        options = feature_cls.options
        history = model.format_chunk_qs(q=query, chunk=chunk, options=options)
        pred = model.predict_with_cot(
            history,
            options=options,
            max_answer_tokens=feature_cls.max_tokens,
        )
        # print(f"{feature_cls.__name__}: {pred}")
    
    _total_llm_calls += 1
    print(f"Total LLM calls: {_total_llm_calls}\r", end="", flush=True)
    return pred

def normalize_date(date_value):
    """Convert various date formats to datetime object."""
    # Handle ISO format: "2019-03-21 14:15:00.000000"
    try:
        return pd.to_datetime(date_value, format='%Y-%m-%d %H:%M:%S.%f').to_pydatetime()
    except ValueError:
        # Handle date format: "6/1/2009"
        return pd.to_datetime(date_value, format='%m/%d/%Y').to_pydatetime()


def group_records_by_time_blocks(records, index_date, block_size_months=None, block_size_weeks=None):
    """
    Group records into time blocks from index_date.
    
    Args:
        records: List of record dictionaries with "Report_Date_Time" key
        index_date: Reference date for grouping
        block_size_months: Size of block in months (e.g., 3 for 3-month blocks)
        block_size_weeks: Size of block in weeks (e.g., 2 for 2-week blocks)
    
    Returns:
        Dictionary mapping block_index to list of records
    """
    index_date_dt = normalize_date(index_date)
    records_by_block = defaultdict(list)
    
    for record in records:
        record_date = record["Report_Date_Time"]
        record_date_dt = normalize_date(record_date)
        
        if block_size_months is not None:
            # Calculate which month block this record belongs to
            months_since_index = (record_date_dt.year - index_date_dt.year) * 12 + (record_date_dt.month - index_date_dt.month)
            block_index = months_since_index // block_size_months
        elif block_size_weeks is not None:
            # Calculate which week block this record belongs to
            days_since_index = (record_date_dt - index_date_dt).days
            block_index = days_since_index // (block_size_weeks * 7)
        else:
            raise ValueError("Either block_size_months or block_size_weeks must be provided")
        
        records_by_block[block_index].append(record)
    
    return records_by_block


def select_record_with_most_keywords(records, keywords):
    """
    Select the record with the most keyword matches.
    
    Args:
        records: List of record dictionaries with "Report_Text" key
        keywords: List of keywords to search for
    
    Returns:
        Tuple of (best_record, keyword_count) or (None, 0) if no keywords found
    """
    best_record = None
    max_keyword_count = 0
    
    for record in records:
        record_text = record["Report_Text"].lower()
        keyword_count = sum(1 for kw in keywords if kw.lower() in record_text)
        
        if keyword_count > max_keyword_count:
            max_keyword_count = keyword_count
            best_record = record
    
    return best_record, max_keyword_count


def filter_records_by_date_range(records, start_date=None, end_date=None):
    """
    Filter records by date range. Always includes start_date (>=) and excludes end_date (<).
    
    Args:
        records: List of record dictionaries with "Report_Date_Time" key
        start_date: Start date. If None, no lower bound. Always inclusive (>=).
        end_date: End date. If None, no upper bound. Always exclusive (<).
    
    Returns:
        Filtered list of records
    """
    filtered_records = []
    for record in records:
        record_date = record["Report_Date_Time"]
        record_date_dt = normalize_date(record_date)
        
        if start_date is not None:
            start_date_dt = normalize_date(start_date)
            if record_date_dt < start_date_dt:
                continue
        
        if end_date is not None:
            end_date_dt = normalize_date(end_date)
            if record_date_dt >= end_date_dt:
                continue
        
        filtered_records.append(record)
    
    return filtered_records


def process_feature_grouped(records, index_date, feature_cls, rows, block_size_months=None, block_size_weeks=None, 
                            follow_up_features=None, keywords=None):
    """
    Process a feature by grouping records into time blocks and selecting the record
    with the most keywords in each block.
    
    Args:
        records: List of record dictionaries
        index_date: Reference date for grouping
        feature_cls: Feature class (e.g., smoking_status)
        rows: List to append result rows to
        block_size_months: Size of block in months
        block_size_weeks: Size of block in weeks
        follow_up_features: Dict mapping prediction value to list of (feature_cls, feature_name) tuples
                           e.g., {"C": [(smoking_amount, "smoking_amount")]}
                           or {"A": [(cancer_date_of_diagnosis, "cancer_date_of_diagnosis"),
                                     (cancer_stage_at_diagnosis, "cancer_stage_at_diagnosis")]}
        keywords: Custom keywords list (defaults to feature_cls.keywords)
    """
    # Use custom keywords if provided, otherwise use feature class keywords
    if keywords is None:
        keywords = feature_cls.keywords
    
    # Group records into time blocks
    records_by_block = group_records_by_time_blocks(
        records, index_date, 
        block_size_months=block_size_months, 
        block_size_weeks=block_size_weeks
    )
    
    # Process each block
    for block_index, block_records in records_by_block.items():
        # Find the record with the most keywords
        best_record, keyword_count = select_record_with_most_keywords(block_records, keywords)
        
        # If we found a record with keywords, process it
        if best_record and keyword_count > 0:
            record_date = best_record["Report_Date_Time"]
            record_text = best_record["Report_Text"]
            
            # Process all keywords
            for kw in keywords:
                chunks = get_chunks_by_keyword(record_text, kw)
                # Process all instances (chunks) of the keyword
                for chunk in chunks:
                    pred = call_llm(feature_cls, chunk, kw)
                    rows.append({
                        "feature_name": feature_cls.__name__,
                        "keyword": kw,
                        "date": record_date,
                        "prediction": pred,
                    })
                    
                    # Handle follow-up features
                    if follow_up_features and pred.upper() in follow_up_features:
                        follow_ups = follow_up_features[pred.upper()]
                        # Support both single tuple and list of tuples
                        if isinstance(follow_ups, tuple):
                            follow_ups = [follow_ups]
                        
                        for follow_up_cls, follow_up_name in follow_ups:
                            follow_up_pred = call_llm(follow_up_cls, chunk, kw)
                            rows.append({
                                "feature_name": follow_up_name,
                                "keyword": kw,
                                "date": record_date,
                                "prediction": follow_up_pred,
                            })


def process_feature_all_records(records, feature_cls, rows, follow_up_features=None):
    """
    Process a feature for all records (no grouping).
    
    Args:
        records: List of record dictionaries
        feature_cls: Feature class (e.g., antibiotics)
        rows: List to append result rows to
        follow_up_features: Dict mapping prediction value to (feature_cls, feature_name) tuple
                           e.g., {"A": (antibiotic_duration, "antibiotic_duration")}
    """
    for record in records:
        record_date = record["Report_Date_Time"]
        record_text = record["Report_Text"]
        
        # Process all keywords
        for kw in feature_cls.keywords:
            chunks = get_chunks_by_keyword(record_text, kw)
            # Process all instances (chunks) of the keyword
            for chunk in chunks:
                pred = call_llm(feature_cls, chunk, kw)
                rows.append({
                    "feature_name": feature_cls.__name__,
                    "keyword": kw,
                    "date": record_date,
                    "prediction": pred,
                })
                
                # Handle follow-up features
                if follow_up_features and pred.upper() in follow_up_features:
                    follow_ups = follow_up_features[pred.upper()]
                    # Support both single tuple and list of tuples
                    if isinstance(follow_ups, tuple):
                        follow_ups = [follow_ups]
                    
                    for follow_up_cls, follow_up_name in follow_ups:
                        # Note: antibiotic_duration doesn't use keyword parameter
                        if follow_up_name == "antibiotic_duration":
                            follow_up_pred = call_llm(follow_up_cls, chunk)
                        else:
                            follow_up_pred = call_llm(follow_up_cls, chunk, kw)
                        rows.append({
                            "feature_name": follow_up_name,
                            "keyword": kw,
                            "date": record_date,
                            "prediction": follow_up_pred,
                        })


def get_chunks_by_keyword(record: str, keyword: str, context_words: int = 1000) -> list[str]:
    """Get chunk by keyword, including +/- context_words before and after the keyword."""
    
    chunks = []
    # Split record into words
    words = record.split()
    
    # Create pattern for case-insensitive keyword search
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    
    # Track positions where keyword was found to avoid overlaps
    matched_positions = []
    
    # Find all occurrences of the keyword in the text
    for match in pattern.finditer(record):
        # Calculate approximate word position
        text_before = record[:match.start()]
        word_pos = len(text_before.split())
        
        # Get context window
        start_pos = max(0, word_pos - context_words)
        end_pos = min(len(words), word_pos + context_words + 1)
        
        # Extract chunk
        chunk_words = words[start_pos:end_pos]
        chunk = ' '.join(chunk_words)
        
        # Add chunk if it's not empty and contains the keyword
        if chunk and keyword.lower() in chunk.lower():
            chunks.append(chunk)
            matched_positions.append((start_pos, end_pos))
    
    return chunks


def process_pt(pt_id):
    """
    Process a single patient.
    Return None if the patient can't be processed (e.g. no acne diagnosis)
    dates are in mm/dd/yyyy format
    """
    ### determine index date ###
    # first acne diagnosis 
    with db.connect() as conn:
        query = text(f"SELECT * FROM dia WHERE EMPI = '{pt_id}' AND (Code LIKE '%L70.0%' OR Code LIKE '%L70.8%' OR Code LIKE '%L70.9%' OR Code LIKE '%L70.1%' OR Code LIKE '%706.1%' OR Code LIKE '%acne%') ORDER BY Date ASC LIMIT 1")
        result = conn.execute(query)
        
        fetched = result.mappings().fetchone()
        if fetched is not None:
            index_date = fetched["Date"]
            index_date = normalize_date(index_date)
            outcome_window_start_date = index_date + pd.Timedelta(days=365)
        else:
            print("WARNING: No acne diagnosis found for patient {pt_id}")
            return None
    print(f"Index date: {index_date}")
    print
    

    with db.connect() as conn:
        query = text(f"SELECT * FROM vis WHERE EMPI = '{pt_id}'")
        result = conn.execute(query)
        records = result.mappings().fetchall()


    # target_features = [ft for ft in PtFeaturesMeta.registry.values() if ft.val_var]

    # for target_feature in target_features:

    
    # List to store DataFrame rows
    rows = []

    ### filter which records are necessary for each feature ###
    # smoking, alcohol: 1 per 3 months
    # abx: every record
    # cancer family any: 1 per 6 months
    # cancer cancer: 1 per 2 weeks
    # immunosuppressed disease: 1 per 6 months

    ### smoking status ###
    # 1 per 3 months
    process_feature_grouped(
        records, index_date, smoking_status, rows,
        block_size_months=3,
        follow_up_features={"C": (smoking_amount, "smoking_amount")}
    )
    print(f"Completed smoking status for {pt_id}")
    
    ### alcohol status ###
    # 1 per 3 months
    process_feature_grouped(
        records, index_date, alcohol_status, rows,
        block_size_months=3,
        follow_up_features={"A": (alcohol_amount, "alcohol_amount")}
    )
    print(f"Completed alcohol status for {pt_id}")
    ### transplant ###
    process_feature_all_records(
        records, transplant, rows,
        follow_up_features={"A": (transplant_date, "transplant_date")}
    )
    print(f"Completed transplant for {pt_id}")
    ### immunosuppressed disease ###
    # 1 per 6 months
    process_feature_grouped(
        records, index_date, immunosuppressed_disease, rows,
        block_size_months=6
    )
    print(f"Completed immunosuppressed disease for {pt_id}")
    ### cancer ###
    # 1 per 2 weeks, use SPECIFIC_CANCERS as keywords
    # Only assess cancer AFTER outcome_window_start_date (strictly after, not inclusive)
    cancer_records = filter_records_by_date_range(records, start_date=outcome_window_start_date)
    process_feature_grouped(
        cancer_records, index_date, cancer_cancer, rows,
        block_size_weeks=2,
        keywords=SPECIFIC_CANCERS,
        follow_up_features={"A": [
            (cancer_date_of_diagnosis, "cancer_date_of_diagnosis"),
            (cancer_stage_at_diagnosis, "cancer_stage_at_diagnosis"),
            (cancer_maximum_stage, "cancer_maximum_stage")
        ]}
    )
    print(f"Completed cancer cancer for {pt_id}")
    ### cancer family any ###
    # 1 per 6 months, use SPECIFIC_CANCERS as keywords
    # Only assess cancer AFTER outcome_window_start_date (strictly after, not inclusive)
    process_feature_grouped(
        cancer_records, index_date, cancer_family_any, rows,
        block_size_months=6,
        keywords=SPECIFIC_CANCERS
    )
    print(f"Completed cancer family any for {pt_id}")
    ### antibiotics ###
    # every record
    # Only assess antibiotics BETWEEN index_date and outcome_window_start_date (inclusive on both ends)
    abx_records = filter_records_by_date_range(records, start_date=index_date, end_date=outcome_window_start_date)
    process_feature_all_records(
        abx_records, antibiotics, rows,
        follow_up_features={"A": (antibiotic_duration, "antibiotic_duration")}
    )
    print(f"Completed antibiotics for {pt_id}")
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df
            


parser = argparse.ArgumentParser()
parser.add_argument('--n_workers', type=int, default=1, help='Number of worker threads (default: 1)')
parser.add_argument('--limit', type=int, default=None, help='Max number of patients to process (default: None, process all)')
parser.add_argument('--dummy-llm', action='store_true', help='Use dummy LLM (sample from prevalence / random dates)')
parser.add_argument('--cohort-type', type=str, default='matched',
                    choices=['all', 'matched', 'case', 'control', 'matched_case', 'matched_control'],
                    help='Which cohort to use: all (all_empis), matched (matched pairs only), case (case_empis), control (control_empis), matched_case, matched_control (default: matched)')
args = parser.parse_args()
DUMMY_LLM = args.dummy_llm

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load EMPIs from cohort JSON
cohort_json_path = Path("cohort/empis.json")
print(f"Loading EMPIs from {cohort_json_path}")
with open(cohort_json_path, 'r') as f:
    cohort_data = json.load(f)

# Select which EMPIs to use based on cohort_type
if args.cohort_type == 'all':
    pt_ids = cohort_data['all_empis']
elif args.cohort_type == 'matched':
    # Use both matched cases and controls (all matched pairs)
    pt_ids = cohort_data['matched_case_empis'] + cohort_data['matched_control_empis']
elif args.cohort_type == 'case':
    pt_ids = cohort_data['case_empis']
elif args.cohort_type == 'control':
    pt_ids = cohort_data['control_empis']
elif args.cohort_type == 'matched_case':
    pt_ids = cohort_data['matched_case_empis']
elif args.cohort_type == 'matched_control':
    pt_ids = cohort_data['matched_control_empis']

# Apply limit if specified
if args.limit is not None:
    pt_ids = pt_ids[:args.limit]

print(f"Processing {len(pt_ids)} patients from cohort type: {args.cohort_type}")
    
    pt_dfs = []
    index_dates = []
    
    if args.n_workers > 1:
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(process_pt, pt_id): pt_id for pt_id in pt_ids}
            for future in tqdm(as_completed(futures), total=len(pt_ids)):
                pt_df = future.result()
                if pt_df is not None:
                    pt_dfs.append(pt_df)
    else:
        for pt_id in tqdm(pt_ids):
            pt_df = process_pt(pt_id)
            if pt_df is not None:
                pt_dfs.append(pt_df)
    print

    
    pt_df = pd.concat(pt_dfs)
    
    # Create output directory with timestamp
    timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"full_inference_out/{timestamp_str}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame as JSONL
    if len(pt_df) > 0:
        jsonl_path = output_dir / "results.jsonl"
        with open(jsonl_path, 'w') as f:
            for _, row in pt_df.iterrows():
                json.dump(row.to_dict(), f)
                f.write('\n')
        print(f"\nSaved results to {jsonl_path}")
    
    # Plot histogram of feature frequencies
    if len(pt_df) > 0:
        feature_counts = pt_df["feature_name"].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_counts)), feature_counts.values)
        plt.xticks(range(len(feature_counts)), feature_counts.index, rotation=45, ha='right')
        plt.xlabel("Feature Name")
        plt.ylabel("Frequency")
        plt.title("Frequency of Each Feature")
        plt.tight_layout()
        histogram_path = output_dir / "feature_frequency_histogram.png"
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to {histogram_path}")
        print(f"\nFeature frequencies:")
        for feature, count in feature_counts.items():
            print(f"  {feature}: {count}")
    else:
        print("No data to plot")

end_time = datetime.now()
total_time = end_time - start_time

# Save runtime info to markdown file
runtime_info_path = output_dir / "runtime_info.md"
with open(runtime_info_path, 'w') as f:
    f.write(f"# Runtime Information\n\n")
    f.write(f"- **Start time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- **End time**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- **Total time**: {total_time}\n")
    f.write(f"- **Number of workers**: {args.n_workers}\n")
    f.write(f"- **Cohort type**: {args.cohort_type}\n")
    f.write(f"- **Patient limit**: {args.limit if args.limit is not None else 'None (all)'}\n")
    f.write(f"- **Dummy LLM**: {args.dummy_llm}\n")
    f.write(f"- **Number of patients processed**: {len(pt_ids)}\n")
    if len(pt_df) > 0:
        f.write(f"- **Total rows in results**: {len(pt_df)}\n")

print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {total_time}")
print(f"Saved runtime info to {runtime_info_path}")


