import os
import sys
import pandas as pd
from pathlib import Path
from utils import get_dataset
import argparse

# We need to import pt_features but it has a dependency chain through models -> vllm
# To avoid this, we'll mock the models module before importing
class MockModels:
    """Mock models module to avoid vllm dependency"""
    class MrModel:
        pass
    
    @staticmethod
    def retry_with_validation(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Mock the models module
sys.modules['models'] = MockModels()

# Now we can safely import pt_features
from pt_features import PtFeaturesMeta


def make_map_fn(feature_name, target_cls, split):
    """
    Create a mapping function to transform data into verl format.
    
    Args:
        feature_name: Name of the feature being extracted
        target_cls: The feature class from PtFeaturesMeta.registry
        split: 'train' or 'eval'
    
    Returns:
        Function that processes a row into verl format
    """
    def process_fn(row, idx):
        # Extract the text chunk and ground truth
        chunk = row['chunk']
        ground_truth = row['val_unified']
        found_keywords = row['found_keywords']
        
        # Handle found_keywords - could be string or list
        if isinstance(found_keywords, list):
            keyword = found_keywords[0] if found_keywords else ""
        else:
            keyword = str(found_keywords) if pd.notna(found_keywords) else ""
        
        # Generate instruction using feature's query method (same as train_sft_7b.py)
        instruction = target_cls.query(chunk=chunk, keyword=keyword)
        
        # Create the prompt in chat template format with the actual instruction
        prompt_text = f"{instruction}\n\n{chunk}"
        
        data = {
            "data_source": feature_name,
            "prompt": [{
                "role": "user",
                "content": prompt_text
            }],
            "ability": "clinical_extraction",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'report_number': row['Report_Number'],
                'is_synthetic': row['is_synthetic'],
                'keyword': keyword
            }
        }
        return data
    
    return process_fn


def prepare_dataset_for_grpo(
    data_source="mgb",
    feature_names=None,
    train_split=0.5,
    downsample=None,
    random_state=42,
    output_dir="labeled_data/grpo"
):
    """
    Prepare dataset in parquet format for GRPO training.
    
    Args:
        data_source: Data source to use (e.g., "mgb", "mimic")
        feature_names: Optional list of specific feature names to load
        train_split: Fraction of data to use for training
        downsample: Optional number of examples to downsample to per feature
        random_state: Random state for downsampling
        output_dir: Directory to save parquet files
    """
    print(f"Loading dataset from {data_source}...")
    datasets = get_dataset(
        data_source=data_source,
        feature_names=feature_names,
        train_split=train_split,
        downsample=downsample,
        random_state=random_state
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each feature
    all_train_data = []
    all_eval_data = []
    
    for feature_name, data_dict in datasets.items():
        print(f"\nProcessing {feature_name}...")
        
        # Get the feature class to access query method
        if feature_name not in PtFeaturesMeta.registry:
            print(f"  Warning: Feature {feature_name} not found in registry, skipping...")
            continue
        
        target_cls = PtFeaturesMeta.registry[feature_name]
        
        # Process train data
        train_df = data_dict['train']
        if len(train_df) > 0:
            print(f"  Processing {len(train_df)} train examples...")
            map_fn = make_map_fn(feature_name, target_cls, 'train')
            for idx, row in train_df.iterrows():
                processed_row = map_fn(row, idx)
                all_train_data.append(processed_row)
        
        # Process eval data
        eval_df = data_dict['eval']
        if len(eval_df) > 0:
            print(f"  Processing {len(eval_df)} eval examples...")
            map_fn = make_map_fn(feature_name, target_cls, 'eval')
            for idx, row in eval_df.iterrows():
                processed_row = map_fn(row, idx)
                all_eval_data.append(processed_row)
    
    # Convert to DataFrames and save as parquet
    print(f"\nSaving data to {output_dir}...")
    
    if all_train_data:
        train_parquet_df = pd.DataFrame(all_train_data)
        train_output_path = output_path / "train.parquet"
        train_parquet_df.to_parquet(train_output_path, index=False)
        print(f"  Saved {len(train_parquet_df)} train examples to {train_output_path}")
    else:
        print("  No train data to save")
    
    if all_eval_data:
        eval_parquet_df = pd.DataFrame(all_eval_data)
        # format with instructions
        # append "Let's think step by step and output the final answer after \"####\"" to the prompt

        def update_prompt(row):
            prompt = row["prompt"].copy()
            prompt[0]["content"] = prompt[0]["content"] + "\n\nLet's think step by step and output the final answer after ####"
            return prompt
        eval_parquet_df["prompt"] = eval_parquet_df.apply(update_prompt, axis=1)
        eval_output_path = output_path / "test.parquet"
        eval_parquet_df.to_parquet(eval_output_path, index=False)
        print(f"  Saved {len(eval_parquet_df)} eval examples to {eval_output_path}")
    else:
        print("  No eval data to save")
    
    print("\nDataset preparation complete!")
    print(f"Data saved in: {output_dir}")
    
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for GRPO training in parquet format')
    parser.add_argument('--data_source', default='mgb', help='Data source to use (e.g., mgb, mimic)')
    parser.add_argument('--feature_names', nargs='+', default=None, 
                        help='Specific feature names to load (if None, loads all)')
    parser.add_argument('--train_split', type=float, default=0.5,
                        help='Fraction of data to use for training')
    parser.add_argument('--downsample', type=int, default=None,
                        help='Number of examples to downsample to per feature')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for downsampling')
    parser.add_argument('--output_dir', default='labeled_data/grpo',
                        help='Directory to save parquet files')
    
    args = parser.parse_args()
    
    prepare_dataset_for_grpo(
        data_source=args.data_source,
        feature_names=args.feature_names,
        train_split=args.train_split,
        downsample=args.downsample,
        random_state=args.random_state,
        output_dir=args.output_dir
    )

