import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import enum
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
# from textsplit.tools import split_single


CHUNKSIZE = 200  # Maximum words per chunk


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


def get_dataset(
    data_source: str = "mgb",
    feature_names: Optional[List[str]] = None,
    train_split: float = 0.5,
    downsample: Optional[int] = None,
    random_state: int = 42
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
    
    # Filter features if specified
    if feature_names is not None:
        target_features_set = set(feature_names)
        filtered_files = []
        for file_path in excel_files:
            feature_name = file_path.stem.replace("_chunks", "")
            if feature_name in target_features_set:
                filtered_files.append(file_path)
        excel_files = filtered_files
    
    if not excel_files:
        raise ValueError(f"No excel files found for data_source={data_source}, feature_names={feature_names}")
    
    all_datasets = {}
    
    for file_path in excel_files:
        feature_name = file_path.stem.replace("_chunks", "")
        print(f"Loading dataset for {feature_name}...")
        
        # Load natural data
        natural_df = pd.read_excel(file_path)
        natural_df = natural_df[natural_df["val_unified"].notna()]
        natural_df["val_unified"] = natural_df["val_unified"].astype(str)
        natural_df["Report_Number"] = natural_df["Report_Number"].astype(str)
        natural_df["is_synthetic"] = False
        
        # Load synthetic data if available
        synth_data_source = f"{data_source}_synth"
        feature_dir_name = file_path.parent.name
        synth_file_path = Path(f"labeled_data/synthetic/{synth_data_source}/{feature_dir_name}/{feature_name}_chunks.xlsx")
        
        synthetic_df = pd.DataFrame()
        if synth_file_path.exists():
            print(f"  Found synthetic data: {synth_file_path}")
            synthetic_df = pd.read_excel(synth_file_path)
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
        
        all_datasets[feature_name] = {
            "train": train_df,
            "eval": eval_df,
            "train_natural": natural_train,
            "train_synthetic": synthetic_train,
            "eval_natural": natural_eval,
            "eval_synthetic": synthetic_eval,
            "all": combined_df
        }
        
        print(f"  Loaded {feature_name}: {len(combined_df)} total ({len(natural_df)} natural, {len(synthetic_df)} synthetic)")
        print(f"    Train: {len(train_df)} ({len(natural_train)} natural, {len(synthetic_train)} synthetic)")
        print(f"    Eval: {len(eval_df)} ({len(natural_eval)} natural, {len(synthetic_eval)} synthetic)")
    
    return all_datasets