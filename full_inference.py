
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
import threading

### TODO: use the good queries we generated from testing

random.seed(42)

db = create_engine(db_url)
model_id = "Qwen/Qwen2.5-72B-Instruct-AWQ"
model = MrModel(model_id=model_id)


# Counter for total LLM calls
_total_llm_calls = 0

# Counters for structured vs LLM window processing
_structured_windows_count = 0
_llm_windows_count = 0

# Per-patient statistics (all patients, will output first 10 in report)
_per_patient_stats = {}
_per_patient_stats_lock = threading.Lock()
_thread_local = threading.local()

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


def _extract_and_pool_structured(feature_cls, block_records):
    """Extract structured data from all records in a block and pool them.

    Args:
        feature_cls: Feature class with option_from_structured and pooling_fn
        block_records: List of record dictionaries

    Returns:
        str or None: Pooled prediction or None if no/inconclusive structured data
    """
    if not hasattr(feature_cls, 'option_from_structured'):
        return None

    if not hasattr(feature_cls, 'pooling_fn'):
        return None

    # Extract structured value from each record
    structured_hits = []
    for record in block_records:
        pred = feature_cls.option_from_structured([record])
        if pred is not None:
            structured_hits.append(pred)

    # If no predictions, return None (inconclusive - use LLM)
    if not structured_hits:
        return None

    # Filter out inconclusive values before pooling
    # If feature defines inconclusive_values, remove them from predictions
    if hasattr(feature_cls, 'inconclusive_values'):
        structured_hits = [p for p in structured_hits if p not in feature_cls.inconclusive_values]

    # If no conclusive predictions remain after filtering, return None (use LLM, maybe)
    if not structured_hits:
        return None

    # Pool only the conclusive predictions
    pooled = feature_cls.pooling_fn(structured_hits)

    return pooled


def _process_follow_up_features_llm(follow_up_features, pred_value, record_text, keywords, record_date, rows):
    """Process follow-up features using LLM for a given prediction."""
    global _llm_windows_count, _per_patient_stats, _per_patient_stats_lock

    if not follow_up_features or pred_value.upper() not in follow_up_features:
        return

    follow_ups = follow_up_features[pred_value.upper()]
    # Support both single tuple and list of tuples
    if isinstance(follow_ups, tuple):
        follow_ups = [follow_ups]

    for follow_up_cls, follow_up_name in follow_ups:
        # Track LLM usage for follow-up feature
        _llm_windows_count += 1
        if hasattr(_thread_local, 'current_patient_id') and _thread_local.current_patient_id:
            with _per_patient_stats_lock:
                if _thread_local.current_patient_id not in _per_patient_stats:
                    _per_patient_stats[_thread_local.current_patient_id] = {"structured": 0, "llm": 0}
                _per_patient_stats[_thread_local.current_patient_id]["llm"] += 1

        for kw in keywords:
            chunks = get_chunks_by_keyword(record_text, kw)
            for chunk in chunks:
                follow_up_pred = call_llm(follow_up_cls, chunk, kw)
                rows.append({
                    "feature_name": follow_up_name,
                    "keyword": kw,
                    "date": record_date,
                    "prediction": follow_up_pred,
                })


def _process_follow_up_features_structured(follow_up_features, pred_value, block_records, keywords, rows, phy_records=None):
    """Process follow-up features using structured-then-LLM approach."""
    if not follow_up_features or pred_value.upper() not in follow_up_features:
        return

    follow_ups = follow_up_features[pred_value.upper()]
    # Support both single tuple and list of tuples
    if isinstance(follow_ups, tuple):
        follow_ups = [follow_ups]

    for follow_up_cls, follow_up_name in follow_ups:
        # Try structured data first (extract from phy_records and pool)
        structured_records = phy_records if phy_records is not None else []
        structured_value = _extract_and_pool_structured(follow_up_cls, structured_records)

        if structured_value is not None:
            # Structured data was conclusive
            if block_records:
                record_date = max(record["Report_Date_Time"] for record in block_records)
            else:
                record_date = None

            rows.append({
                "feature_name": follow_up_name,
                "keyword": "STRUCTURED_DATA",
                "date": record_date,
                "prediction": structured_value,
            })
        else:
            # Fall back to LLM processing
            best_record, keyword_count = select_record_with_most_keywords(block_records, keywords)
            if best_record and keyword_count > 0:
                record_date = best_record["Report_Date_Time"]
                record_text = best_record["Report_Text"]
                _process_follow_up_features_llm({pred_value.upper(): (follow_up_cls, follow_up_name)},
                                               pred_value, record_text, keywords, record_date, rows)


def process_feature_single_block(block_records, feature_cls, follow_up_features=None, keywords=None, phy_records=None, dia_records=None, fallback_to_llm=True):
    """
    Process a feature for a single time block.

    Extracts structured data from phy_records or dia_records, pools them, and uses the result.
    Optionally falls back to LLM-based extraction using block_records if structured data is unavailable or inconclusive.

    Args:
        block_records: List of vis table record dictionaries for this time block (for LLM)
        feature_cls: Feature class (e.g., smoking_status)
        follow_up_features: Dict mapping prediction value to list of (feature_cls, feature_name) tuples
                           e.g., {"C": [(smoking_amount, "smoking_amount")]}
                           or {"A": [(cancer_date_of_diagnosis, "cancer_date_of_diagnosis"),
                                     (cancer_stage_at_diagnosis, "cancer_stage_at_diagnosis")]}
        keywords: Custom keywords list (defaults to feature_cls.keywords)
        phy_records: List of phy table record dictionaries (for structured data extraction)
        dia_records: List of dia table record dictionaries (for diagnosis-based structured data extraction)
        fallback_to_llm: Whether to fall back to LLM if structured data is unavailable/inconclusive (default: True)

    Returns:
        List of result row dictionaries
    """
    global _structured_windows_count, _llm_windows_count, _per_patient_stats, _per_patient_stats_lock

    rows = []

    # Use custom keywords if provided, otherwise use feature class keywords
    if keywords is None:
        keywords = feature_cls.keywords

    # Try to get value from structured data (extract from phy_records or dia_records and pool)
    # Returns None if no structured data OR if structured data is inconclusive
    # For cancer_cancer, use dia_records; for other features, use phy_records
    if feature_cls.__name__ == "cancer_cancer":
        structured_records = dia_records if dia_records is not None else []
    else:
        structured_records = phy_records if phy_records is not None else []
    structured_value = _extract_and_pool_structured(feature_cls, structured_records)

    # If structured data gave a conclusive answer, use it and process follow-ups
    if structured_value is not None:
        # Track structured window usage
        _structured_windows_count += 1
        if hasattr(_thread_local, 'current_patient_id') and _thread_local.current_patient_id:
            with _per_patient_stats_lock:
                # Initialize stats for this patient if not already tracked
                if _thread_local.current_patient_id not in _per_patient_stats:
                    _per_patient_stats[_thread_local.current_patient_id] = {"structured": 0, "llm": 0}
                _per_patient_stats[_thread_local.current_patient_id]["structured"] += 1

        # Use the most recent record date for the block
        if block_records:
            record_date = max(record["Report_Date_Time"] for record in block_records)
        else:
            record_date = None

        rows.append({
            "feature_name": feature_cls.__name__,
            "keyword": "STRUCTURED_DATA",
            "date": record_date,
            "prediction": structured_value,
        })

        # Process follow-up features using structured-then-LLM approach
        # Note: Follow-up features for cancer should still use LLM (no dia_records passed)
        _process_follow_up_features_structured(follow_up_features, structured_value, block_records, keywords, rows, phy_records=None)

        return rows  # Done processing - structured data was conclusive

    # Structured data was None/inconclusive
    # Only fall back to LLM if fallback_to_llm is True
    if not fallback_to_llm:
        return rows  # Don't fall back to LLM - just skip this feature

    # Fall back to LLM processing
    # Find the record with the most keywords
    best_record, keyword_count = select_record_with_most_keywords(block_records, keywords)

    # If we found a record with keywords, process it
    if best_record and keyword_count > 0:
        # Track LLM window usage (only if we actually process it)
        _llm_windows_count += 1
        if hasattr(_thread_local, 'current_patient_id') and _thread_local.current_patient_id:
            with _per_patient_stats_lock:
                # Initialize stats for this patient if not already tracked
                if _thread_local.current_patient_id not in _per_patient_stats:
                    _per_patient_stats[_thread_local.current_patient_id] = {"structured": 0, "llm": 0}
                _per_patient_stats[_thread_local.current_patient_id]["llm"] += 1
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

                # Handle follow-up features using structured-then-LLM approach
                if follow_up_features and pred.upper() in follow_up_features:
                    follow_ups = follow_up_features[pred.upper()]
                    # Support both single tuple and list of tuples
                    if isinstance(follow_ups, tuple):
                        follow_ups = [follow_ups]

                    for follow_up_cls, follow_up_name in follow_ups:
                        # Try structured data first (from ALL records in block)
                        follow_up_structured = _extract_and_pool_structured(follow_up_cls, block_records)

                        if follow_up_structured is not None:
                            # Structured data was conclusive - use it
                            rows.append({
                                "feature_name": follow_up_name,
                                "keyword": "STRUCTURED_DATA",
                                "date": record_date,
                                "prediction": follow_up_structured,
                            })
                        else:
                            # Fall back to LLM on this chunk
                            follow_up_pred = call_llm(follow_up_cls, chunk, kw)
                            rows.append({
                                "feature_name": follow_up_name,
                                "keyword": kw,
                                "date": record_date,
                                "prediction": follow_up_pred,
                            })

    return rows


def process_feature_all_records(records, feature_cls, follow_up_features=None):
    """
    Process a feature for all records (no grouping).

    Args:
        records: List of record dictionaries
        feature_cls: Feature class (e.g., antibiotics)
        follow_up_features: Dict mapping prediction value to (feature_cls, feature_name) tuple
                           e.g., {"A": (antibiotic_duration, "antibiotic_duration")}

    Returns:
        List of result row dictionaries
    """
    rows = []

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

    return rows


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
    ### Step 1: Determine index date and calculate time windows ###
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

    # Set current patient ID for tracking (only after confirming they have acne diagnosis)
    _thread_local.current_patient_id = pt_id

    print(f"Index date: {index_date}")
    print(f"Outcome window starts: {outcome_window_start_date}")

    ### Step 2: Get all records for patient ###
    with db.connect() as conn:
        # Query vis table for LLM-based extraction (has Report_Text)
        query = text(f"SELECT * FROM vis WHERE EMPI = '{pt_id}'")
        result = conn.execute(query)
        records = result.mappings().fetchall()

        # Query phy table for structured data extraction (has Concept_Name and Result)
        phy_query = text(f"SELECT * FROM phy WHERE EMPI = '{pt_id}'")
        phy_result = conn.execute(phy_query)
        phy_records = phy_result.mappings().fetchall()

        # Query dia table for diagnosis-based structured data extraction (has Code for ICD codes)
        dia_query = text(f"SELECT * FROM dia WHERE EMPI = '{pt_id}'")
        dia_result = conn.execute(dia_query)
        dia_records = dia_result.mappings().fetchall()

    ### Step 3: Define time windows and filter records ###
    # Window 1: All records (for smoking, alcohol, transplant, immunosuppressed_disease)
    all_records = records

    # Window 2: Records after outcome_window_start_date (for cancer features)
    outcome_records = filter_records_by_date_range(records, start_date=outcome_window_start_date)

    # Filter dia_records for outcome window (note: dia table uses "Date" field, not "Report_Date_Time")
    outcome_dia_records = []
    for record in dia_records:
        record_date = record["Date"]
        record_date_dt = normalize_date(record_date)
        if record_date_dt >= normalize_date(outcome_window_start_date):
            outcome_dia_records.append(record)

    # Window 3: Records between index_date and outcome_window_start_date (for antibiotics)
    treatment_window_records = filter_records_by_date_range(records, start_date=index_date, end_date=outcome_window_start_date)

    print(f"Total records: {len(all_records)}, Outcome window: {len(outcome_records)}, Treatment window: {len(treatment_window_records)}, Phy records: {len(phy_records)}, Dia records: {len(dia_records)}")

    ### Step 4: Calculate time blocks for each feature ###
    # Smoking status: 1 per 3 months
    smoking_blocks = group_records_by_time_blocks(all_records, index_date, block_size_months=3)

    # Alcohol status: 1 per 3 months
    alcohol_blocks = group_records_by_time_blocks(all_records, index_date, block_size_months=3)

    # Immunosuppressed disease: 1 per 6 months
    immunosuppressed_blocks = group_records_by_time_blocks(all_records, index_date, block_size_months=6)

    # Cancer cancer: 1 per 3 months
    cancer_cancer_blocks = group_records_by_time_blocks(outcome_records, index_date, block_size_months=3)

    # Cancer family any: 1 per 6 months
    cancer_family_blocks = group_records_by_time_blocks(outcome_records, index_date, block_size_months=6)

    print(f"Time blocks - Smoking: {len(smoking_blocks)}, Alcohol: {len(alcohol_blocks)}, Immunosuppressed: {len(immunosuppressed_blocks)}, Cancer cancer: {len(cancer_cancer_blocks)}, Cancer family: {len(cancer_family_blocks)}")

    ### Step 5: Process features organized by time window ###
    rows = []

    # === Features using all records ===

    # Smoking status: process each 3-month block
    # Note: Extracts structured data from phy_records, pools using pooling_fn
    # Falls back to LLM if structured data is unavailable/inconclusive
    # Follow-up features (smoking_amount) also use structured-then-LLM approach
    # Use LLM if no structured: yes
    for block_index, block_records in smoking_blocks.items():
        block_rows = process_feature_single_block(
            block_records, smoking_status,
            follow_up_features={"C": (smoking_amount, "smoking_amount")},
            phy_records=phy_records
        )
        rows.extend(block_rows)
    print(f"Completed smoking status for {pt_id}")

    # Alcohol status: process each 3-month block
    # Note: Extracts structured data from phy_records, pools using pooling_fn
    # Falls back to LLM if structured data is unavailable/inconclusive
    # Follow-up features (alcohol_amount) also use structured-then-LLM approach
    # Use LLM if no structured: yes
    for block_index, block_records in alcohol_blocks.items():
        block_rows = process_feature_single_block(
            block_records, alcohol_status,
            follow_up_features={"A": (alcohol_amount, "alcohol_amount")},
            phy_records=phy_records
        )
        rows.extend(block_rows)
    print(f"Completed alcohol status for {pt_id}")

    # Transplant: every record
    # Use LLM if no structured: no
    # Note: transplant feature needs option_from_structured method implemented
    # transplant_rows = process_feature_all_records(
    #     all_records, transplant,
    #     follow_up_features={"A": (transplant_date, "transplant_date")}
    # )
    # rows.extend(transplant_rows)
    # print(f"Completed transplant for {pt_id}")

    # Immunosuppressed disease: process each 6-month block
    # Use LLM if no structured: no
    # Note: immunosuppressed_disease feature needs option_from_structured method implemented
    # for block_index, block_records in immunosuppressed_blocks.items():
    #     block_rows = process_feature_single_block(
    #         block_records, immunosuppressed_disease,
    #         fallback_to_llm=False  # Don't fall back to LLM if no structured data
    #     )
    #     rows.extend(block_rows)
    # print(f"Completed immunosuppressed disease for {pt_id}")

    # === Features using outcome window (after outcome_window_start_date) ===

    # Cancer cancer: process each 3-month block using structured data from dia table
    # Follow-up features will use LLM only
    # Use LLM if no structured: no
    for block_index, block_records in cancer_cancer_blocks.items():
        block_rows = process_feature_single_block(
            block_records, cancer_cancer,
            keywords=SPECIFIC_CANCERS,
            follow_up_features={"A": [
                (cancer_date_of_diagnosis, "cancer_date_of_diagnosis"),
                (cancer_stage_at_diagnosis, "cancer_stage_at_diagnosis"),
                (cancer_maximum_stage, "cancer_maximum_stage")
            ]},
            dia_records=outcome_dia_records,  # Pass outcome dia_records for structured cancer detection
            fallback_to_llm=False  # Don't fall back to LLM if no structured data
        )
        rows.extend(block_rows)
    print(f"Completed cancer cancer for {pt_id}")

    # # Cancer family any: process each 6-month block
    # for block_index, block_records in cancer_family_blocks.items():
    #     block_rows = process_feature_single_block(
    #         block_records, cancer_family_any,
    #         keywords=SPECIFIC_CANCERS
    #     )
    #     rows.extend(block_rows)
    # print(f"Completed cancer family any for {pt_id}")

    # # === Features using treatment window (index_date to outcome_window_start_date) ===

    # # Antibiotics: every record
    # antibiotic_rows = process_feature_all_records(
    #     treatment_window_records, antibiotics,
    #     follow_up_features={"A": (antibiotic_duration, "antibiotic_duration")}
    # )
    # rows.extend(antibiotic_rows)
    # print(f"Completed antibiotics for {pt_id}")

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Clear current patient ID
    _thread_local.current_patient_id = None

    return df
            


parser = argparse.ArgumentParser()
parser.add_argument('--n_workers', type=int, default=1, help='Number of worker threads (default: 1)')
parser.add_argument('--limit', type=int, default=None, help='Max number of patients to process (default: None, process all)')
parser.add_argument('--dummy-llm', action='store_true', help='Use dummy LLM (sample from prevalence / random dates)')
args = parser.parse_args()
DUMMY_LLM = args.dummy_llm

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load EMPIs from cohort JSON (matched cases only)
cohort_json_path = Path("cohort/empis.json")
print(f"Loading matched case EMPIs from {cohort_json_path}")
with open(cohort_json_path, 'r') as f:
    cohort_data = json.load(f)

# Use only matched cases
pt_ids = cohort_data['matched_case_empis']

# Apply limit if specified
if args.limit is not None:
    pt_ids = pt_ids[:args.limit]

print(f"Processing {len(pt_ids)} matched case patients")

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
print()

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
    f.write(f"- **Patient limit**: {args.limit if args.limit is not None else 'None (all)'}\n")
    f.write(f"- **Dummy LLM**: {args.dummy_llm}\n")
    f.write(f"- **Number of patients processed**: {len(pt_ids)}\n")
    if len(pt_df) > 0:
        f.write(f"- **Total rows in results**: {len(pt_df)}\n")

    # Add window processing statistics
    f.write(f"\n## Window Processing Breakdown\n\n")
    total_windows = _structured_windows_count + _llm_windows_count
    if total_windows > 0:
        structured_pct = (_structured_windows_count / total_windows) * 100
        llm_pct = (_llm_windows_count / total_windows) * 100
        f.write(f"- **Total windows processed**: {total_windows}\n")
        f.write(f"- **Windows processed via structured data**: {_structured_windows_count} ({structured_pct:.1f}%)\n")
        f.write(f"- **Windows processed via LLM**: {_llm_windows_count} ({llm_pct:.1f}%)\n")
    else:
        f.write(f"- No windows processed\n")

    # Add per-patient breakdown for first 10 patients
    if _per_patient_stats:
        # Take first 10 patients (in order they were added to dict)
        first_10_patients = list(_per_patient_stats.items())[:10]
        f.write(f"\n## Per-Patient Breakdown (First {len(first_10_patients)} Patients)\n\n")
        f.write(f"| Patient ID | Structured | LLM | Total | Structured % |\n")
        f.write(f"|------------|------------|-----|-------|-------------|\n")
        for pt_id, stats in first_10_patients:
            total = stats["structured"] + stats["llm"]
            pct = (stats["structured"] / total * 100) if total > 0 else 0
            f.write(f"| {pt_id} | {stats['structured']} | {stats['llm']} | {total} | {pct:.1f}% |\n")

print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {total_time}")
print(f"Saved runtime info to {runtime_info_path}")


