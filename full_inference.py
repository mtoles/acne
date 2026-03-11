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
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
import numpy as np
import logging
from kw_builder import (
    get_kws_from_icd,
    ICD_TO_TRANSPLANT_DESC,
    ICD9_TRANSPLANT_PROCEDURE_CODES,
    ICD10_TRANSPLANT_PROCEDURE_CODES,
    ICD9_DIAGNOSIS_CODES,
    ICD10_DIAGNOSIS_CODES,
)
from kw_builder import transplant_df as transplant_icd_df, cancer_df

_ICD_TO_CANCER_LABEL = {}
for _code_group, _row in cancer_df.iterrows():
    _label = _row["ICD Label / Group"]
    for _code in str(_code_group).split(", "):
        _ICD_TO_CANCER_LABEL[_code.strip()] = _label


def _icd_to_cancer_label(icd_code):
    """Map an ICD code (e.g. '174.9') to its cancer label by prefix matching."""
    for prefix, label in _ICD_TO_CANCER_LABEL.items():
        if icd_code.startswith(prefix):
            return label
    return icd_code


random.seed(42)

db = create_engine(db_url, pool_size=100, max_overflow=200, pool_timeout=300)
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

# Global records directory (set before processing loop)
_records_dir = None


def _json_default(obj):
    """JSON serializer for datetime and numpy objects."""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
    valid_value_counts = {
        k: v for k, v in value_counts.items() if k in feature_cls.options
    }

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
        if issubclass(feature_cls, PtNumericFeatureBase):
            # Return random days (0-180) or "F"
            if random.random() < 0.15:
                pred = "F"
            else:
                pred = str(random.randint(0, 180))
        elif issubclass(feature_cls, PtDateFeatureBase):
            # return a date between 2000-01-01 and 2025-12-31
            year = random.randint(2000, 2025)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            pred = f"{year}{month:02d}{day:02d}"
        else:
            # Sample according to prevalence distribution
            distribution = get_prevalence_distribution(feature_cls)
            options = list(distribution.keys())
            weights = list(distribution.values())
            pred = random.choices(options, weights=weights, k=1)[0]
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

    # Log the call
    empi = getattr(_thread_local, "current_patient_id", "")
    block_id = getattr(_thread_local, "block_id", "")
    block_dates = getattr(_thread_local, "block_dates", "")
    logging.getLogger("llm_calls").info(
        f"{empi}\t{block_id}\t{block_dates}\t{feature_cls.__name__}\t{keyword}\t{pred}"
    )

    return pred



def group_records_by_time_blocks(
    records, index_date, block_size_months=None, block_size_weeks=None
):
    """
    Group records into time blocks from index_date.

    Args:
        records: DataFrame with "Report_Date_Time" column
        index_date: Reference date for grouping
        block_size_months: Size of block in months (e.g., 3 for 3-month blocks)
        block_size_weeks: Size of block in weeks (e.g., 2 for 2-week blocks)

    Returns:
        Dictionary mapping block_index to DataFrame of records (with block_start_date, block_end_date columns)
    """
    if records.empty:
        return {}

    index_date_dt = normalize_date(index_date)
    records = records.copy()

    record_dates = records["Report_Date_Time"].apply(normalize_date)

    if block_size_months is not None:
        months_since_index = (record_dates.dt.year - index_date_dt.year) * 12 + (
            record_dates.dt.month - index_date_dt.month
        )
        block_indices = months_since_index // block_size_months
        records["_block_index"] = block_indices
        records["block_start_date"] = block_indices.apply(
            lambda bi: index_date_dt + pd.DateOffset(months=int(bi) * block_size_months)
        )
        records["block_end_date"] = block_indices.apply(
            lambda bi: index_date_dt
            + pd.DateOffset(months=(int(bi) + 1) * block_size_months)
        )
    elif block_size_weeks is not None:
        days_since_index = (record_dates - index_date_dt).dt.days
        block_indices = days_since_index // (block_size_weeks * 7)
        records["_block_index"] = block_indices
        records["block_start_date"] = block_indices.apply(
            lambda bi: index_date_dt + pd.Timedelta(weeks=int(bi) * block_size_weeks)
        )
        records["block_end_date"] = block_indices.apply(
            lambda bi: index_date_dt
            + pd.Timedelta(weeks=(int(bi) + 1) * block_size_weeks)
        )
    else:
        raise ValueError(
            "Either block_size_months or block_size_weeks must be provided"
        )

    records_by_block = {}
    for block_index, group in records.groupby("_block_index"):
        records_by_block[block_index] = group.drop(
            columns=["_block_index"]
        ).reset_index(drop=True)

    return records_by_block


def select_record_with_most_keywords(records, keywords):
    """
    Select the record with the most keyword matches.

    Args:
        records: DataFrame with "Report_Text" column
        keywords: List of keywords to search for

    Returns:
        Tuple of (best_record_series, keyword_count) or (None, 0) if no keywords found
    """
    best_record = None
    max_keyword_count = 0

    for _, record in records.iterrows():
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
        records: DataFrame with a date column ("Report_Date_Time", "Date", or "Medication_Date")
        start_date: Start date. If None, no lower bound. Always inclusive (>=).
        end_date: End date. If None, no upper bound. Always exclusive (<).

    Returns:
        Filtered DataFrame
    """
    if records.empty:
        return records.copy()

    if "Report_Date_Time" in records.columns:
        date_col = "Report_Date_Time"
    elif "Date" in records.columns:
        date_col = "Date"
    elif "Medication_Date" in records.columns:
        date_col = "Medication_Date"
    else:
        raise ValueError("Record does not have a date field")

    dates = records[date_col].apply(normalize_date)
    mask = pd.Series(True, index=records.index)

    if start_date is not None:
        start_date_dt = normalize_date(start_date)
        mask = mask & (dates >= start_date_dt)

    if end_date is not None:
        end_date_dt = normalize_date(end_date)
        mask = mask & (dates < end_date_dt)

    return records[mask].reset_index(drop=True)


def _extract_structured(feature_cls, block_records):
    """Extract structured data from all records in a block.

    Args:
        feature_cls: Feature class with option_from_structured
        block_records: DataFrame of records

    Returns:
        list: List of prediction dicts (empty if no conclusive structured data)
    """
    if not hasattr(feature_cls, "option_from_structured"):
        return []

    # Extract structured value from each record
    structured_hits = []
    for _, record in block_records.iterrows():
        pred = feature_cls.option_from_structured([dict(record)])
        if pred is not None:
            out_dict = dict(record)
            out_dict["pred"] = pred
            structured_hits.append(out_dict)

    return structured_hits


def process_single_block_structured(
    block_records, feature_cls, phy_records=None, dia_records=None
):
    """
    Process a feature for a single time block using structured data only.

    Args:
        block_records: List of vis table record dictionaries (used for date)
        feature_cls: Feature class (e.g., smoking_status)
        phy_records: List of phy table record dictionaries (for structured data extraction)
        dia_records: List of dia table record dictionaries (for diagnosis-based structured data extraction)
        med_records: List of med table record dictionaries (for medication-based structured data extraction)
    Returns:
        List of result row dictionaries (empty if no structured data found)
    """
    global _structured_windows_count, _per_patient_stats, _per_patient_stats_lock

    rows = []

    # For cancer_cancer, use dia_records; for other features, use phy_records
    if feature_cls.__name__ == "cancer_cancer":
        structured_records = dia_records if dia_records is not None else pd.DataFrame()
    else:
        structured_records = phy_records if phy_records is not None else pd.DataFrame()
    structured_hits = _extract_structured(feature_cls, structured_records)

    if structured_hits:
        # Pool the hits
        structured_value = feature_cls.pooling_fn([x["pred"] for x in structured_hits])
        # Track structured window usage
        _structured_windows_count += 1
        if (
            hasattr(_thread_local, "current_patient_id")
            and _thread_local.current_patient_id
        ):
            with _per_patient_stats_lock:
                if _thread_local.current_patient_id not in _per_patient_stats:
                    _per_patient_stats[_thread_local.current_patient_id] = {
                        "structured": 0,
                        "llm": 0,
                    }
                _per_patient_stats[_thread_local.current_patient_id]["structured"] += 1

        # Use the most recent record date for the block
        if not block_records.empty:
            record_date = block_records["Report_Date_Time"].max()
        else:
            record_date = None

        rows.append(
            {
                "feature_name": feature_cls.__name__,
                "keyword": "STRUCTURED_DATA",
                "date": record_date,
                "prediction": structured_value,
            }
        )

    return rows, structured_hits


def make_keyword_filter(keywords):
    """Create a chunk filter that returns True if any keyword is present."""

    def filter_fn(chunk):
        return len(has_keyword(chunk, keywords)) > 0

    return filter_fn


def process_single_block_llm(
    block_records, feature_cls, chunk_filter_fn, short_circuit, all_records
):
    """
    Process a feature for a single time block using LLM.

    Args:
        block_records: List of vis table record dictionaries for this time block
        feature_cls: Feature class (e.g., smoking_status)
        chunk_filter_fn: Function(chunk) -> bool. If True, process chunk with LLM.
        short_circuit: If True, stop after finding a conclusive prediction.
        all_records: If True, process every record with keywords. If False, only the record with the most keywords.

    Returns:
        List of result row dictionaries
    """
    global _llm_windows_count, _per_patient_stats, _per_patient_stats_lock

    rows = []
    keywords = feature_cls.keywords

    if all_records:
        records_to_process = []
        for _, record in block_records.iterrows():
            record_text = record.get("Report_Text", "")
            if record_text and len(has_keyword(record_text, keywords)) > 0:
                records_to_process.append(record)
    else:
        best_record, keyword_count = select_record_with_most_keywords(
            block_records, keywords
        )
        records_to_process = [best_record] if best_record is not None and keyword_count > 0 else []

    for record in records_to_process:
        _llm_windows_count += 1
        if (
            hasattr(_thread_local, "current_patient_id")
            and _thread_local.current_patient_id
        ):
            with _per_patient_stats_lock:
                if _thread_local.current_patient_id not in _per_patient_stats:
                    _per_patient_stats[_thread_local.current_patient_id] = {
                        "structured": 0,
                        "llm": 0,
                    }
                _per_patient_stats[_thread_local.current_patient_id]["llm"] += 1
        record_date = record["Report_Date_Time"]
        record_text = record["Report_Text"]

        inconclusive_values = getattr(feature_cls, "inconclusive_values", set())
        found_conclusive = False
        chunks = chunk_text(record_text)
        for chunk in chunks:
            if found_conclusive and short_circuit:
                break
            if chunk_filter_fn(chunk):
                found_kws = has_keyword(chunk, keywords)
                for kw in found_kws:
                    pred = call_llm(feature_cls, chunk, kw)
                    found_supp_kws = []
                    if hasattr(feature_cls, "supplimental_kws"):
                        found_supp_kws = has_keyword(chunk, feature_cls.supplimental_kws)
                    rows.append(
                        {
                            "feature_name": feature_cls.__name__,
                            "keyword": kw,
                            "supplimental_keywords": found_supp_kws,
                            "date": record_date,
                            "prediction": pred,
                        }
                    )
                    if pred not in inconclusive_values:
                        found_conclusive = True
                        break

    return rows


def get_chunks_by_keyword(
    record: str, keyword: str, context_words: int = 200
) -> list[str]:
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
        text_before = record[: match.start()]
        word_pos = len(text_before.split())

        # Get context window
        start_pos = max(0, word_pos - context_words)
        end_pos = min(len(words), word_pos + context_words + 1)

        # Extract chunk
        chunk_words = words[start_pos:end_pos]
        chunk = " ".join(chunk_words)

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
    # Skip if already processed
    jsonl_path = _records_dir / f"{pt_id}.jsonl"
    if jsonl_path.exists():
        print(f"Skipping {pt_id} (already has {jsonl_path})")
        return None

    ### Step 1: Determine index date and calculate time windows ###
    with db.connect() as conn:
        query = text(
            f"SELECT * FROM dia WHERE EMPI = '{pt_id}' AND (Code LIKE '%L70.0%' OR Code LIKE '%L70.8%' OR Code LIKE '%L70.9%' OR Code LIKE '%L70.1%' OR Code LIKE '%706.1%' OR Code LIKE '%acne%') ORDER BY Date ASC LIMIT 1"
        )
        result = conn.execute(query)

        fetched = result.mappings().fetchone()
        if fetched is not None:
            index_date = fetched["Date"]
            index_date = normalize_date(index_date)
            outcome_window_start_date = index_date + pd.Timedelta(days=365 * 2)
        else:
            print("WARNING: No acne diagnosis found for patient {pt_id}")
            return None

    # Set current patient ID for tracking (only after confirming they have acne diagnosis)
    _thread_local.current_patient_id = pt_id

    # print(f"Index date: {index_date}")
    # print(f"Outcome window starts: {outcome_window_start_date}")

    ### Step 2: Get all records for patient ###
    with db.connect() as conn:
        # Query vis table for LLM-based extraction (has Report_Text)
        query = text(f"SELECT * FROM vis WHERE EMPI = '{pt_id}'")
        result = conn.execute(query)
        vis_records = pd.DataFrame(result.mappings().fetchall())

        # Query phy table for structured data extraction (has Concept_Name and Result)
        phy_query = text(f"SELECT * FROM phy WHERE EMPI = '{pt_id}'")
        phy_result = conn.execute(phy_query)
        phy_records = pd.DataFrame(phy_result.mappings().fetchall())

        # Query dia table for diagnosis-based structured data extraction (has Code for ICD codes)
        dia_query = text(f"SELECT * FROM dia WHERE EMPI = '{pt_id}'")
        dia_result = conn.execute(dia_query)
        dia_records = pd.DataFrame(dia_result.mappings().fetchall())

        med_query = text(f"SELECT * FROM med WHERE EMPI = '{pt_id}'")
        med_result = conn.execute(med_query)
        med_records = pd.DataFrame(med_result.mappings().fetchall())

        prc_query = text(f"SELECT * FROM prc WHERE EMPI = '{pt_id}'")
        prc_result = conn.execute(prc_query)
        prc_records = pd.DataFrame(prc_result.mappings().fetchall())

    ### Step 3: Define time windows and filter records ###
    # Window 1: All records (for smoking, alcohol, transplant, immunosuppressed_disease)
    all_records = vis_records

    # Window 2: Records after outcome_window_start_date (for cancer features)
    outcome_records = filter_records_by_date_range(
        vis_records, start_date=outcome_window_start_date
    )

    # Filter dia_records for outcome window (note: dia table uses "Date" field, not "Report_Date_Time")
    outcome_dia_records = filter_records_by_date_range(
        dia_records, start_date=outcome_window_start_date
    )

    # Window 3: Records between index_date and outcome_window_start_date (for antibiotics)
    treatment_window_dia_records = filter_records_by_date_range(
        vis_records, start_date=index_date, end_date=outcome_window_start_date
    )
    med_records = filter_records_by_date_range(
        med_records, start_date=index_date, end_date=outcome_window_start_date
    )

    # print(f"Total records: {len(all_records)}, Outcome window: {len(outcome_records)}, Treatment window: {len(treatment_window_dia_records)}, Phy records: {len(phy_records)}, Dia records: {len(dia_records)}")

    ### Step 4: Calculate time blocks for each feature ###
    # Smoking status: 1 per 3 months
    smoking_blocks = group_records_by_time_blocks(
        filter_records_by_date_range(all_records, end_date=outcome_window_start_date),
        index_date,
        block_size_months=3,
    )

    # Alcohol status: 1 per 3 months
    alcohol_blocks = group_records_by_time_blocks(
        filter_records_by_date_range(all_records, end_date=outcome_window_start_date),
        index_date,
        block_size_months=3,
    )

    # Cancer cancer: 1 per 3 months
    cancer_cancer_blocks = group_records_by_time_blocks(
        outcome_records, index_date, block_size_months=3
    )

    # Cancer family any: earliest records to outcome window start, 1 per 6 months
    cancer_family_blocks = group_records_by_time_blocks(
        filter_records_by_date_range(all_records, end_date=outcome_window_start_date),
        index_date,
        block_size_months=6,
    )

    ### Step 5: Process features organized by time window ###
    rows = []

    # === Features using all records ===

    # Smoking & alcohol: extract structured data from ALL pre-outcome phy records
    # (independent of vis records, so we don't miss phy data in time periods without visits)
    pre_outcome_phy_records = filter_records_by_date_range(
        phy_records, end_date=outcome_window_start_date
    )

    # Smoking status: structured from phy
    smoking_structured_hits = _extract_structured(smoking_status, pre_outcome_phy_records)
    for hit in smoking_structured_hits:
        rows.append({
            "feature_name": "smoking_status",
            "keyword": "STRUCTURED_DATA",
            "date": hit["Date"],
            "prediction": hit["pred"],
        })
    # Smoking amount follow-up: if any structured hit is "C" (current smoker)
    if any(h["pred"] == "C" for h in smoking_structured_hits):
        smoking_amount_hits = _extract_structured(smoking_amount, pre_outcome_phy_records)
        for hit in smoking_amount_hits:
            rows.append({
                "feature_name": "smoking_amount",
                "keyword": "STRUCTURED_DATA",
                "date": hit["Date"],
                "prediction": hit["pred"],
            })

    # Smoking status: LLM fallback on vis blocks without structured data
    for block_index, block_records in smoking_blocks.items():
        _thread_local.block_id = f"smoking_{block_index}"
        _thread_local.block_dates = (
            f"{block_records['Report_Date_Time'].min()[:10]}~{block_records['Report_Date_Time'].max()[:10]}"
            if not block_records.empty
            else ""
        )
        block_rows = process_single_block_llm(
            block_records,
            smoking_status,
            make_keyword_filter(smoking_status.keywords),
            short_circuit=True,
            all_records=False,
        )
        rows.extend(block_rows)
        if block_rows:
            pooled_value = smoking_status.pooling_fn(
                [row["prediction"] for row in block_rows]
            )
            if pooled_value == "C":
                follow_up_rows = process_single_block_llm(
                    block_records,
                    smoking_amount,
                    make_keyword_filter(smoking_amount.keywords),
                    short_circuit=True,
                    all_records=False,
                )
                for row in follow_up_rows:
                    row["feature_name"] = "smoking_amount"
                rows.extend(follow_up_rows)

    # Alcohol status: structured from phy
    alcohol_structured_hits = _extract_structured(alcohol_status, pre_outcome_phy_records)
    for hit in alcohol_structured_hits:
        rows.append({
            "feature_name": "alcohol_status",
            "keyword": "STRUCTURED_DATA",
            "date": hit["Date"],
            "prediction": hit["pred"],
        })
    # Alcohol amount follow-up: if any structured hit is "A" (drinks alcohol)
    if any(h["pred"] == "A" for h in alcohol_structured_hits):
        alcohol_amount_hits = _extract_structured(alcohol_amount, pre_outcome_phy_records)
        for hit in alcohol_amount_hits:
            rows.append({
                "feature_name": "alcohol_amount",
                "keyword": "STRUCTURED_DATA",
                "date": hit["Date"],
                "prediction": hit["pred"],
            })

    # Alcohol status: LLM fallback on vis blocks
    for block_index, block_records in alcohol_blocks.items():
        _thread_local.block_id = f"alcohol_{block_index}"
        _thread_local.block_dates = (
            f"{block_records['Report_Date_Time'].min()[:10]}~{block_records['Report_Date_Time'].max()[:10]}"
            if not block_records.empty
            else ""
        )
        block_rows = process_single_block_llm(
            block_records,
            alcohol_status,
            make_keyword_filter(alcohol_status.keywords),
            short_circuit=True,
            all_records=False,
        )
        rows.extend(block_rows)
        if block_rows:
            pooled_value = alcohol_status.pooling_fn(
                [row["prediction"] for row in block_rows]
            )
            if pooled_value == "A":
                follow_up_rows = process_single_block_llm(
                    block_records,
                    alcohol_amount,
                    make_keyword_filter(alcohol_amount.keywords),
                    short_circuit=True,
                    all_records=False,
                )
                for row in follow_up_rows:
                    row["feature_name"] = "alcohol_amount"
                rows.extend(follow_up_rows)

    # Transplant: earliest records to outcome window start
    # Use LLM if no structured: no
    pre_outcome_dia_records = filter_records_by_date_range(
        dia_records, end_date=outcome_window_start_date
    )
    transplant_df_slices = []
    transplant_df_slices.append(
        pre_outcome_dia_records[
            (pre_outcome_dia_records["Code_Type"] == "ICD9")
            & (pre_outcome_dia_records["Code"].isin(ICD9_DIAGNOSIS_CODES))
        ]
    )
    transplant_df_slices.append(
        pre_outcome_dia_records[
            (pre_outcome_dia_records["Code_Type"] == "ICD10")
            & (pre_outcome_dia_records["Code"].isin(ICD10_DIAGNOSIS_CODES))
        ]
    )
    # todo:
    transplant_structured_df = (
        pd.concat(transplant_df_slices).drop_duplicates().reset_index(drop=True)
    )
    for _, record in transplant_structured_df.iterrows():
        rows.append(
            {
                "feature_name": "transplant",
                "keyword": "STRUCTURED_DATA",
                "code": record["Code"],
                "transplant_description": ICD_TO_TRANSPLANT_DESC[record["Code"]],
                "transplant_code Type": record["Code_Type"],
                "date": record["Date"],
                "prediction": "A",
            }
        )

    # print(f"Completed transplant for {pt_id}")

    ### Diseases ###
    # filter in only records in CONFOUNDING_DISEASE_CODES
    pre_outcome_dia_for_diseases = filter_records_by_date_range(
        dia_records, end_date=outcome_window_start_date
    )
    confounding_disease_records = pre_outcome_dia_for_diseases[
        pre_outcome_dia_for_diseases["Code"].isin(CONFOUNDING_DISEASE_CODES.keys())
    ]
    for _, record in confounding_disease_records.iterrows():
        if record["Code_Type"] == "ICD9":
            icd_dict = CONFOUNDING_DISEASE_ICD9_CODES
        elif record["Code_Type"] == "ICD10":
            icd_dict = CONFOUNDING_DISEASE_ICD10_CODES
        else:
            continue
        code = record["Code"]
        if code not in icd_dict:
            continue
        disease_name = record["Diagnosis_Name"]
        disease_type = icd_dict[code]["disease_type"]
        disease_description = icd_dict[code]["description"]
        rows.append(
            {
                "feature_name": "disease",
                "keyword": "STRUCTURED_DATA",
                "Code": record["Code"],
                "disease": disease_name,
                "disease_type": disease_type,
                "date": record["Date"],
                "prediction": "A",
            }
        )

    # # === Features using outcome window (after outcome_window_start_date) ===

    # Cancer cancer: process each 3-month block using structured data from dia table
    # Structured only, no LLM fallback
    for block_index, block_records in cancer_cancer_blocks.items():
        _thread_local.block_id = f"cancer_{block_index}"
        _thread_local.block_dates = (
            f"{block_records['Report_Date_Time'].min()[:10]}~{block_records['Report_Date_Time'].max()[:10]}"
            if not block_records.empty
            else ""
        )
        block_start_date = (
            block_records.iloc[0]["block_start_date"] if not block_records.empty else ""
        )
        block_end_date = (
            block_records.iloc[0]["block_end_date"] if not block_records.empty else ""
        )
        block_dia_records = filter_records_by_date_range(
            outcome_dia_records, start_date=block_start_date, end_date=block_end_date
        )
        block_rows, structured_hits = process_single_block_structured(
            block_records, cancer_cancer, dia_records=block_dia_records
        )
        cancer_hits = [x for x in structured_hits if x["pred"] == "A"]
        for hit in structured_hits:
            record_date = block_records["Report_Date_Time"].max() if not block_records.empty else None
            rows.append({
                "feature_name": "cancer_cancer",
                "keyword": _icd_to_cancer_label(hit["Code"]),
                "icd_code": hit["Code"],
                "date": record_date,
                "prediction": hit["pred"],
            })
        # Follow-up features if any prediction is "A"
        # if any(row["prediction"].upper() == "A" for row in block_rows):
        for cancer_hit in cancer_hits:
            icd = cancer_hit["Code"]
            kws = get_kws_from_icd(icd)

            # Follow-ups use LLM (no structured data for these)
            def cancer_date_filter(chunk):
                has_kw = len(has_keyword(chunk, kws)) > 0
                has_date = contains_date(chunk)
                return has_kw and has_date

            date_rows = process_single_block_llm(
                block_records,
                cancer_date_of_diagnosis,
                chunk_filter_fn=cancer_date_filter,
                short_circuit=True,
                all_records=False,
            )
            for row in date_rows:
                row["feature_name"] = "cancer_date_of_diagnosis"
            rows.extend(date_rows)

            def cancer_stage_filter(chunk):
                # check if cancer kws are present
                has_cancer_kw = len(has_keyword(chunk, kws)) > 0
                has_stage_kw = len(has_keyword(chunk, CANCER_STAGE_2_PART_KEYWORDS)) > 0
                return has_cancer_kw and has_stage_kw

            stage_rows = process_single_block_llm(
                block_records,
                cancer_stage_at_diagnosis,
                chunk_filter_fn=cancer_stage_filter,
                short_circuit=True,
                all_records=False,
            )
            for row in stage_rows:
                row["feature_name"] = "cancer_stage_at_diagnosis"
            rows.extend(stage_rows)

            max_stage_rows = process_single_block_llm(
                block_records, cancer_maximum_stage, chunk_filter_fn=cancer_stage_filter,
                short_circuit=True, all_records=False,
            )
            for row in max_stage_rows:
                row["feature_name"] = "cancer_maximum_stage"
            rows.extend(max_stage_rows)
    # print(f"Completed cancer cancer for {pt_id}")

    # Cancer family any: process each 6-month block (LLM only)
    def cancer_family_filter(chunk):
        has_cancer = len(has_keyword(chunk, SPECIFIC_CANCERS)) > 0
        has_family = len(has_keyword(chunk, cancer_family_any.supplimental_kws)) > 0
        return has_cancer and has_family

    for block_index, block_records in cancer_family_blocks.items():
        block_rows = process_single_block_llm(
            block_records, cancer_family_any, chunk_filter_fn=cancer_family_filter,
            short_circuit=True, all_records=False,
        )
        rows.extend(block_rows)
    # print(f"Completed cancer family any for {pt_id}")

    # === Features using treatment window (index_date to outcome_window_start_date) ===

    # Antibiotics: process all treatment window records as a single block

    if "Code" in med_records.columns:
        abx_records = med_records[
            med_records.apply(lambda r: (str(r["Code_Type"]), str(r["Code"])) in ABX_CODE_TYPE_PAIRS, axis=1)
        ]
        for _, abx_record in abx_records.iterrows():
            rows.append(
                {
                    "feature_name": "antibiotics",
                    "keyword": "STRUCTURED_DATA",
                    "Medication_Description": abx_record["Medication"],
                    "Medication_Code": abx_record["Code"],
                    "Medication_Quantity": abx_record["Quantity"],
                    "date": abx_record["Medication_Date"],
                    "prediction": "A",
                }
            )
        def contains_contraceptive_kw(s):
            for contraceptive_kw in CONTRACEPTIVE_NAMES:
                pattern = r'(?:^|[\W])' + re.escape(contraceptive_kw.lower()) + r'(?:$|[\W])'
                if re.search(pattern, s.lower()):
                    return True
            return False

        contraceptive_records = med_records[
            med_records.apply(lambda r: contains_contraceptive_kw(r["Medication"]), axis=1)
        ]
        for _, contraceptive_record in contraceptive_records.iterrows():
            rows.append(
                {
                    "feature_name": "contraceptives",
                    "keyword": "STRUCTURED_DATA",
                    "Medication_Description": contraceptive_record["Medication"],
                    "Medication_Code": contraceptive_record.get("Code", ""),
                    "Medication_Quantity": contraceptive_record.get("Quantity", ""),
                    "date": contraceptive_record["Medication_Date"],
                    "prediction": "A",
                }
            )

    duration_numeric_rows = process_single_block_llm(
        treatment_window_dia_records,
        antibiotic_duration_numeric,
        make_keyword_filter(antibiotic_duration_numeric.keywords),
        short_circuit=False,
        all_records=True,
    )
    rows.extend(duration_numeric_rows)

    # Save per-patient JSONL
    if rows:
        jsonl_path = _records_dir / f"{pt_id}.jsonl"
        with open(jsonl_path, "w") as f:
            for row in rows:
                out_row = dict(row)
                out_row["prediction"] = prediction_to_description(
                    out_row["feature_name"], out_row["prediction"]
                )
                json.dump(out_row, f, default=_json_default)
                f.write("\n")

    # Build lightweight stats
    feature_counts = Counter()
    structured_feature_counts = Counter()
    llm_feature_counts = Counter()
    for row in rows:
        feature_counts[row["feature_name"]] += 1
        if row["keyword"] == "STRUCTURED_DATA":
            structured_feature_counts[row["feature_name"]] += 1
        else:
            llm_feature_counts[row["feature_name"]] += 1

    # Clear current patient ID
    _thread_local.current_patient_id = None

    return {
        "feature_counts": feature_counts,
        "structured_feature_counts": structured_feature_counts,
        "llm_feature_counts": llm_feature_counts,
        "total_rows": len(rows),
    }



parser = argparse.ArgumentParser()
parser.add_argument(
    "--n-workers", type=int, default=1, help="Number of worker threads (default: 1)"
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Max number of patients to process (default: None, process all)",
)
parser.add_argument(
    "--dummy-llm",
    action="store_true",
    help="Use dummy LLM (sample from prevalence / random dates)",
)
parser.add_argument(
    "--target-empis",
    type=str,
    default=[],
    nargs="*",
    help="List of specific EMPIs to process (default: None, process all in cohort)",
)
args = parser.parse_args()
DUMMY_LLM = args.dummy_llm

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load EMPIs from cohort JSON (matched cases only)
cohort_json_path = Path("cohort/empis.json")
print(f"Loading matched case EMPIs from {cohort_json_path}")
with open(cohort_json_path, "r") as f:
    cohort_data = json.load(f)

# Load matched case and control EMPIs, cases first
matched_case_empis = cohort_data["matched_case_empis"]
matched_control_empis = cohort_data["matched_control_empis"]
pt_ids = matched_case_empis + matched_control_empis

# Apply limit if specified
if args.limit is not None:
    pt_ids = pt_ids[: args.limit]

pt_ids = args.target_empis + pt_ids

print(f"Processing {len(matched_case_empis)} matched cases + {len(matched_control_empis)} matched controls = {len(pt_ids)} patients")

# Create output directory with timestamp
timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"full_inference_out/{timestamp_str}")
output_dir.mkdir(parents=True, exist_ok=True)

# Create records directory for per-EMPI JSONL files
_records_dir = Path("full_inference_out/records")
_records_dir.mkdir(parents=True, exist_ok=True)

# Set up LLM call logging
llm_logger = logging.getLogger("llm_calls")
llm_logger.setLevel(logging.INFO)
llm_handler = logging.FileHandler(output_dir / "llm_calls.log")
llm_handler.setFormatter(logging.Formatter("%(message)s"))
llm_logger.addHandler(llm_handler)
llm_logger.info("empi\tblock_id\tblock_dates\tfeature\tkeyword\tanswer")

agg_feature_counts = Counter()
agg_structured_feature_counts = Counter()
agg_llm_feature_counts = Counter()
agg_total_rows = 0
patients_processed = 0

if args.n_workers > 1:
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(process_pt, pt_id): pt_id for pt_id in pt_ids}
        for future in tqdm(as_completed(futures), total=len(pt_ids)):
            stats = future.result()
            if stats is not None:
                agg_feature_counts += stats["feature_counts"]
                agg_structured_feature_counts += stats["structured_feature_counts"]
                agg_llm_feature_counts += stats["llm_feature_counts"]
                agg_total_rows += stats["total_rows"]
                patients_processed += 1
else:
    for pt_id in tqdm(pt_ids):
        stats = process_pt(pt_id)
        if stats is not None:
            agg_feature_counts += stats["feature_counts"]
            agg_structured_feature_counts += stats["structured_feature_counts"]
            agg_llm_feature_counts += stats["llm_feature_counts"]
            agg_total_rows += stats["total_rows"]
            patients_processed += 1
print()

# Plot histogram of feature frequencies
if agg_total_rows > 0:
    feature_counts = pd.Series(agg_feature_counts).sort_index()

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_counts)), feature_counts.values)
    plt.xticks(
        range(len(feature_counts)), feature_counts.index, rotation=45, ha="right"
    )
    plt.xlabel("Feature Name")
    plt.ylabel("Frequency")
    plt.title("Frequency of Each Feature")
    plt.tight_layout()
    histogram_path = output_dir / "feature_frequency_histogram.png"
    plt.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to {histogram_path}")
    print(f"\nFeature frequencies:")
    for feature, count in feature_counts.items():
        print(f"  {feature}: {count}")
else:
    print("No data to plot")

# Plot pie charts for LLM vs Structured data breakdown
total_windows = _structured_windows_count + _llm_windows_count
if total_windows > 0:
    # Overall windows breakdown pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart 1: Windows breakdown (Structured vs LLM)
    window_labels = ["Structured Data", "LLM"]
    window_sizes = [_structured_windows_count, _llm_windows_count]
    window_colors = ["#2ecc71", "#3498db"]
    window_explode = (0.05, 0.05)

    ax1.pie(
        window_sizes,
        explode=window_explode,
        labels=window_labels,
        colors=window_colors,
        autopct="%1.1f%%",
        shadow=False,
        startangle=90,
    )
    ax1.set_title(f"Windows Processing Breakdown\n(Total: {total_windows} windows)")

    # Pie chart 2: Extraction method by feature
    extraction_counts = {"Structured Only": 0, "LLM Only": 0, "Mixed (Both)": 0}

    all_features = set(agg_structured_feature_counts.keys()) | set(
        agg_llm_feature_counts.keys()
    )
    for feature in all_features:
        has_structured = agg_structured_feature_counts[feature] > 0
        has_llm = agg_llm_feature_counts[feature] > 0

        if has_structured and has_llm:
            extraction_counts["Mixed (Both)"] += 1
        elif has_structured:
            extraction_counts["Structured Only"] += 1
        else:
            extraction_counts["LLM Only"] += 1

    method_labels = list(extraction_counts.keys())
    method_sizes = list(extraction_counts.values())
    method_colors = ["#2ecc71", "#3498db", "#f39c12"]
    method_explode = (0.05, 0.05, 0.05)

    # Only plot if we have data
    if sum(method_sizes) > 0:
        ax2.pie(
            method_sizes,
            explode=method_explode,
            labels=method_labels,
            colors=method_colors,
            autopct="%1.1f%%",
            shadow=False,
            startangle=90,
        )
        ax2.set_title(
            f"Features by Extraction Method\n(Total: {sum(method_sizes)} features)"
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "No feature data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    plt.tight_layout()
    pie_chart_path = output_dir / "llm_breakdown_pie_charts.png"
    plt.savefig(pie_chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved LLM breakdown pie charts to {pie_chart_path}")

    # Create a third pie chart showing total extractions (structured vs LLM)
    plt.figure(figsize=(8, 8))
    structured_count = sum(agg_structured_feature_counts.values())
    llm_count = sum(agg_llm_feature_counts.values())

    extraction_labels = ["Structured Data Extractions", "LLM Extractions"]
    extraction_sizes = [structured_count, llm_count]
    extraction_colors = ["#2ecc71", "#3498db"]
    extraction_explode = (0.05, 0.05)

    plt.pie(
        extraction_sizes,
        explode=extraction_explode,
        labels=extraction_labels,
        colors=extraction_colors,
        autopct="%1.1f%%",
        shadow=False,
        startangle=90,
    )
    plt.title(
        f"Total Extractions Breakdown\n(Total: {structured_count + llm_count} extractions)"
    )
    plt.tight_layout()

    extraction_pie_path = output_dir / "total_extractions_pie_chart.png"
    plt.savefig(extraction_pie_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved total extractions pie chart to {extraction_pie_path}")

    # Create pie chart showing LLM calls by feature name
    if agg_llm_feature_counts:
        llm_feature_series = pd.Series(agg_llm_feature_counts).sort_values(
            ascending=False
        )

        plt.figure(figsize=(10, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(llm_feature_series)))
        labels = [
            f"{name}\n({count:,})"
            for name, count in zip(llm_feature_series.index, llm_feature_series.values)
        ]

        plt.pie(
            llm_feature_series.values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            shadow=False,
            startangle=90,
        )
        plt.title(
            f"LLM Calls by Feature\n(Total: {sum(llm_feature_series.values):,} calls)"
        )
        plt.tight_layout()

        feature_pie_path = output_dir / "llm_calls_by_feature_pie.png"
        plt.savefig(feature_pie_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved LLM calls by feature pie chart to {feature_pie_path}")
else:
    print("No window processing data to plot")

end_time = datetime.now()
total_time = end_time - start_time

# Save runtime info to markdown file
runtime_info_path = output_dir / "runtime_info.md"
with open(runtime_info_path, "w") as f:
    f.write(f"# Runtime Information\n\n")
    f.write(f"- **Start time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- **End time**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- **Total time**: {total_time}\n")
    f.write(f"- **Number of workers**: {args.n_workers}\n")
    f.write(
        f"- **Patient limit**: {args.limit if args.limit is not None else 'None (all)'}\n"
    )
    f.write(f"- **Dummy LLM**: {args.dummy_llm}\n")
    f.write(f"- **Number of patients processed**: {patients_processed}\n")
    if agg_total_rows > 0:
        f.write(f"- **Total rows in results**: {agg_total_rows}\n")

    # Add window processing statistics
    f.write(f"\n## Window Processing Breakdown\n\n")
    total_windows = _structured_windows_count + _llm_windows_count
    if total_windows > 0:
        structured_pct = (_structured_windows_count / total_windows) * 100
        llm_pct = (_llm_windows_count / total_windows) * 100
        f.write(f"- **Total windows processed**: {total_windows}\n")
        f.write(
            f"- **Windows processed via structured data**: {_structured_windows_count} ({structured_pct:.1f}%)\n"
        )
        f.write(
            f"- **Windows processed via LLM**: {_llm_windows_count} ({llm_pct:.1f}%)\n"
        )
    else:
        f.write(f"- No windows processed\n")

    # Add extraction-level statistics
    if agg_total_rows > 0:
        structured_count = sum(agg_structured_feature_counts.values())
        llm_count = sum(agg_llm_feature_counts.values())
        total_extractions = structured_count + llm_count

        f.write(f"\n## Extraction-Level Breakdown\n\n")
        if total_extractions > 0:
            structured_extraction_pct = (structured_count / total_extractions) * 100
            llm_extraction_pct = (llm_count / total_extractions) * 100
            f.write(f"- **Total extractions**: {total_extractions}\n")
            f.write(
                f"- **Structured data extractions**: {structured_count} ({structured_extraction_pct:.1f}%)\n"
            )
            f.write(f"- **LLM extractions**: {llm_count} ({llm_extraction_pct:.1f}%)\n")

        # Feature breakdown by extraction method
        f.write(f"\n## Features by Extraction Method\n\n")
        all_features = set(agg_structured_feature_counts.keys()) | set(
            agg_llm_feature_counts.keys()
        )
        structured_only = []
        llm_only = []
        mixed = []

        for feature in all_features:
            has_structured = agg_structured_feature_counts[feature] > 0
            has_llm = agg_llm_feature_counts[feature] > 0

            if has_structured and has_llm:
                mixed.append(feature)
            elif has_structured:
                structured_only.append(feature)
            else:
                llm_only.append(feature)

        if structured_only:
            f.write(f"**Structured Data Only** ({len(structured_only)} features):\n")
            for feat in sorted(structured_only):
                f.write(f"  - {feat}\n")
            f.write(f"\n")

        if llm_only:
            f.write(f"**LLM Only** ({len(llm_only)} features):\n")
            for feat in sorted(llm_only):
                f.write(f"  - {feat}\n")
            f.write(f"\n")

        if mixed:
            f.write(f"**Mixed (Both Methods)** ({len(mixed)} features):\n")
            for feat in sorted(mixed):
                f.write(f"  - {feat}\n")
            f.write(f"\n")

    # Add per-patient breakdown for first 10 patients
    if _per_patient_stats:
        # Take first 10 patients (in order they were added to dict)
        first_10_patients = list(_per_patient_stats.items())[:10]
        f.write(
            f"\n## Per-Patient Breakdown (First {len(first_10_patients)} Patients)\n\n"
        )
        f.write(f"| Patient ID | Structured | LLM | Total | Structured % |\n")
        f.write(f"|------------|------------|-----|-------|-------------|\n")
        for pt_id, stats in first_10_patients:
            total = stats["structured"] + stats["llm"]
            pct = (stats["structured"] / total * 100) if total > 0 else 0
            f.write(
                f"| {pt_id} | {stats['structured']} | {stats['llm']} | {total} | {pct:.1f}% |\n"
            )

print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {total_time}")
print(f"Saved runtime info to {runtime_info_path}")
