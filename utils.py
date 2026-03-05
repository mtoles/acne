import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import enum
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
# from textsplit.tools import split_single


CHUNKSIZE = 200  # Maximum words per chunk


def normalize_date(date_value):
    """Convert various date formats to datetime object."""
    # Handle ISO format: "2019-03-21 14:15:00.000000"
    try:
        return pd.to_datetime(date_value, format="%Y-%m-%d %H:%M:%S.%f").to_pydatetime()
    except ValueError:
        # Handle date format: "6/1/2009"
        return pd.to_datetime(date_value, format="%m/%d/%Y").to_pydatetime()


def chunk_text(text) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNKSIZE,
        chunk_overlap=20,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def has_keyword(text, keywords):
    """
    Check if any keywords are found in the text and return the list of found keywords.
    Uses exact word matching (not substring matching).
    
    Args:
        text (str): The text to search in
        keywords (list): List of keywords to search for
        
    Returns:
        list: List of keywords found in the text (empty list if none found)
    """
    
    found_keywords = []
    for keyword in keywords:
        # Create a regex pattern with word boundaries for exact matching
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_keywords.append(keyword)
    
    return list(set(found_keywords))  # Remove duplicates and return as list

class OptionType(enum.Enum):
    DATE = "date"
    CHOICES = "choices"
    NUMERIC = "numeric"


def compute_numeric_pct_error(pred, gt):
    """Compute percent error for a single (pred, gt) pair for numeric features.
    Returns 100.0 for non-numeric values."""
    def to_numeric(val):
        val_str = str(val).strip()
        if val_str.upper() == "F":
            return None
        try:
            return int(float(val_str))
        except (ValueError, TypeError):
            return None

    pred_num = to_numeric(pred)
    gt_num = to_numeric(gt)

    # If either is non-numeric, 100% error
    if pred_num is None or gt_num is None:
        return 100.0
    # If gt is 0: exact match = 0%, otherwise 100%
    if gt_num == 0:
        return 0.0 if pred_num == 0 else 100.0
    # Normal case
    return min(abs(pred_num - gt_num) / gt_num * 100, 100.0)


def compute_numeric_abs_error(pred, gt):
    """Compute absolute error (in days) for a single (pred, gt) pair for numeric features.
    Returns None if either value is non-numeric."""
    def to_numeric(val):
        val_str = str(val).strip()
        if val_str.upper() == "F":
            return None
        return int(float(val_str))

    pred_num = to_numeric(pred)
    gt_num = to_numeric(gt)

    if pred_num is None or gt_num is None:
        return None
    return abs(pred_num - gt_num)


def get_dataset(
    data_source: str,
    feature_names: Optional[List[str]] = None,
    train_split: float = 0.5,
    downsample: Optional[int] = None,
    random_state: int = 42,
    feature_metadata: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load and split datasets for all excel files in the labeled_data directory.

    For each feature, loads natural and synthetic data (if available), combines them,
    and splits into train and eval sets. The split is done separately for natural
    and synthetic data to maintain proper distribution.

    Args:
        data_source: Data source to use (e.g., "mgb", "mimic")
        feature_names: Optional list of specific feature names to load. If None, loads all features.
        train_split: Fraction of data to use for training (rest goes to eval). Default 0.5.
                    Note: This is applied separately to natural and synthetic data.
        downsample: Optional number of examples to downsample to per feature. If None, uses all data.
        random_state: Random state for downsampling and shuffling.
        feature_metadata: Optional dict mapping feature_name -> {"data_source_feature": str, "gt_column": str}.
                         data_source_feature: load data from this feature's file instead of own name.
                         gt_column: use this column as ground truth instead of "val_unified".

    Returns:
        Dictionary mapping feature names to dictionaries containing:
        - "train": Combined train DataFrame (natural + synthetic)
        - "eval": Combined eval DataFrame (natural + synthetic)
        - "train_natural": Natural train DataFrame
        - "train_synthetic": Synthetic train DataFrame (empty if no synthetic data)
        - "eval_natural": Natural eval DataFrame
        - "eval_synthetic": Synthetic eval DataFrame (empty if no synthetic data)
        - "all": Combined DataFrame with all data (before split)

    Example:
        datasets = get_dataset(data_source="mgb", feature_names=["smoking_status"])
        train_df = datasets["smoking_status"]["train"]
        eval_df = datasets["smoking_status"]["eval"]
    """
    if feature_metadata is None:
        feature_metadata = {}

    labeled_data_dir = Path(f"labeled_data/{data_source}")

    if not labeled_data_dir.exists():
        raise ValueError(f"Data source directory not found: {labeled_data_dir}")

    # Get all xlsx files from subdirectories
    excel_files = []
    for subdir in labeled_data_dir.iterdir():
        if subdir.is_dir():
            chunk_files = list(subdir.glob("*_chunks.xlsx"))
            excel_files.extend(chunk_files)

    excel_files = sorted(excel_files)

    # Build mapping from file feature name -> [(requested_feature_name, gt_column)]
    # This handles features that load from another feature's file
    file_to_features = {}
    if feature_names is not None:
        for fname in feature_names:
            meta = feature_metadata.get(fname, {})
            file_feature = meta.get("data_source_feature", fname)
            gt_col = meta.get("gt_column", "val_unified")
            if file_feature not in file_to_features:
                file_to_features[file_feature] = []
            file_to_features[file_feature].append((fname, gt_col))

    # Filter features if specified
    if feature_names is not None:
        target_file_features = set(file_to_features.keys())
        filtered_files = []
        for file_path in excel_files:
            file_feature_name = file_path.stem.replace("_chunks", "")
            if file_feature_name in target_file_features:
                filtered_files.append(file_path)
        excel_files = filtered_files

    if not excel_files:
        raise ValueError(f"No excel files found for data_source={data_source}, feature_names={feature_names}")
    
    all_datasets = {}

    for file_path in excel_files:
        file_feature_name = file_path.stem.replace("_chunks", "")

        # Determine which requested features map to this file
        if feature_names is not None:
            features_from_file = file_to_features.get(file_feature_name, [(file_feature_name, "val_unified")])
        else:
            features_from_file = [(file_feature_name, "val_unified")]

        # Load the raw Excel once per file
        raw_df = pd.read_excel(file_path)

        for requested_feature_name, gt_col in features_from_file:
            print(f"Loading dataset for {requested_feature_name}...")

            # Load natural data using the appropriate gt column
            natural_df = raw_df.copy()
            # Rename gt column to val_unified so downstream code is uniform
            if gt_col != "val_unified":
                natural_df["val_unified"] = natural_df[gt_col]
            natural_df = natural_df[natural_df["val_unified"].notna()]
            natural_df["val_unified"] = natural_df["val_unified"].astype(str)
            natural_df["Report_Number"] = natural_df["Report_Number"].astype(str)
            natural_df["is_synthetic"] = False

            # Load synthetic data if available
            synth_data_source = f"{data_source}_synth"
            feature_dir_name = file_path.parent.name
            synth_file_path = Path(f"labeled_data/synthetic/{synth_data_source}/{feature_dir_name}/{file_feature_name}_chunks.xlsx")

            synthetic_df = pd.DataFrame()
            if synth_file_path.exists():
                print(f"  Found synthetic data: {synth_file_path}")
                synthetic_df = pd.read_excel(synth_file_path)
                if gt_col != "val_unified" and gt_col in synthetic_df.columns:
                    synthetic_df["val_unified"] = synthetic_df[gt_col]
                synthetic_df = synthetic_df[synthetic_df["val_unified"].notna()]
                synthetic_df["val_unified"] = synthetic_df["val_unified"].astype(str)
                synthetic_df["Report_Number"] = synthetic_df["Report_Number"].astype(str)
                synthetic_df["is_synthetic"] = True
            else:
                print(f"  No synthetic data found at {synth_file_path}")

            # Combine natural and synthetic data
            if len(synthetic_df) > 0:
                combined_df = pd.concat([natural_df, synthetic_df], ignore_index=True)
            else:
                combined_df = natural_df.copy()

            # Apply downsampling if specified
            if downsample is not None and len(combined_df) > downsample:
                combined_df = combined_df.sample(n=downsample, random_state=random_state).reset_index(drop=True)
                # Re-separate after downsampling
                natural_df = combined_df[combined_df["is_synthetic"] == False].copy()
                synthetic_df = combined_df[combined_df["is_synthetic"] == True].copy()

            # Split natural data into train/eval
            natural_total = len(natural_df)
            natural_eval_size = int(natural_total * train_split)

            if natural_total > 0:
                natural_eval = natural_df.head(natural_eval_size).copy()
                natural_train = natural_df.iloc[natural_eval_size:].copy()
            else:
                natural_eval = pd.DataFrame()
                natural_train = pd.DataFrame()

            # Split synthetic data into train/eval
            synthetic_total = len(synthetic_df)
            synthetic_eval_size = int(synthetic_total * train_split)

            if synthetic_total > 0:
                synthetic_eval = synthetic_df.head(synthetic_eval_size).copy()
                synthetic_train = synthetic_df.iloc[synthetic_eval_size:].copy()
            else:
                synthetic_eval = pd.DataFrame()
                synthetic_train = pd.DataFrame()

            # Combine train and eval sets
            train_df = pd.concat([natural_train, synthetic_train], ignore_index=True) if len(natural_train) > 0 or len(synthetic_train) > 0 else pd.DataFrame()
            eval_df = pd.concat([natural_eval, synthetic_eval], ignore_index=True) if len(natural_eval) > 0 or len(synthetic_eval) > 0 else pd.DataFrame()

            all_datasets[requested_feature_name] = {
                "train": train_df,
                "eval": eval_df,
                "train_natural": natural_train,
                "train_synthetic": synthetic_train,
                "eval_natural": natural_eval,
                "eval_synthetic": synthetic_eval,
                "all": combined_df
            }

            print(f"  Loaded {requested_feature_name}: {len(combined_df)} total ({len(natural_df)} natural, {len(synthetic_df)} synthetic)")
            print(f"    Train: {len(train_df)} ({len(natural_train)} natural, {len(synthetic_train)} synthetic)")
            print(f"    Eval: {len(eval_df)} ({len(natural_eval)} natural, {len(synthetic_eval)} synthetic)")

    return all_datasets


def contains_date(text):
    """
    Checks if a string contains a date.
    - Matches standard formats: 12/12/2023, 12-12-23, 12.12.23
    - Matches textual formats: Jan 12, 12th January, May 2023
    - EXCLUDES pure number lists separated only by spaces (e.g., "12 3 99")
    """
    if not text:
        return False

    # 1. Define Helpers
    
    # Word boundary (\b) prevents matching inside words (e.g. "maybe", "marching")
    months = (
        r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
    )
    
    suffix = r'(?:st|nd|rd|th)?'

    # STRICT SEPARATOR: For numeric dates (e.g. 10/10/20)
    # Must contain -, /, or . (optionally surrounded by whitespace)
    # This excludes "12 12 12" because there is no symbol.
    strict_sep = r'\s*[-/.]\s*'

    # LOOSE SEPARATOR: For textual dates (e.g. Jan 10)
    # Allows whitespace, commas, dashes, slashes, or dots.
    loose_sep = r'(?:\s+|[-/.,]\s*)'

    # 2. Define Patterns

    # Pattern A: Numeric Only (e.g., 10/12/20, 2020-10-12, 1.1.2000)
    # Uses strict_sep, so "12 3 99" returns False, but "12/3/99" returns True.
    pattern_numeric = r'\b\d{1,4}' + strict_sep + r'\d{1,2}' + strict_sep + r'\d{2,4}\b'

    # Pattern B: Month + Day + Optional Year (e.g., "Jan 12", "January 12th 2023")
    pattern_md = months + loose_sep + r'\d{1,2}' + suffix + r'(?:' + loose_sep + r'\d{2,4})?\b'

    # Pattern C: Day + Month + Optional Year (e.g., "12 Jan", "12th January 2023")
    pattern_dm = r'\b\d{1,2}' + suffix + loose_sep + months + r'(?:' + loose_sep + r'\d{2,4})?\b'

    # Pattern D: Month + Year (e.g., "May 2023", "Jan '23")
    pattern_my = months + loose_sep + r"[']?\d{2,4}\b"

    # 3. Combine and Search
    full_pattern = f"({pattern_numeric}|{pattern_md}|{pattern_dm}|{pattern_my})"
    
    # Ignore case so "jan" matches "Jan" or "JAN"
    match = re.search(full_pattern, text, re.IGNORECASE)

    return bool(match)