#!/usr/bin/env python3
"""
Evaluate a trained Qwen 2.5 7B Instruct model with LoRA adapters

CUDA_VISIBLE_DEVICES=3 python train_eval.py \
  --model_path training_runs/20260113_191705 \
  --downsample_eval 128
"""

import os
import sys
import argparse
from pathlib import Path

# Fix tokenizers parallelism warning when using multiprocessing data loaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import PeftModel
from datasets import Dataset

# Import modular functions from train_sft_7b
from train_sft_7b import (
    ALPACA_PROMPT,
    ensure_single_gpu,
    print_gpu_memory,
    load_config,
    get_config_value,
    resolve_model_config,
    resolve_dataset_config,
    load_tokenizer,
    build_bnb_config,
    load_model,
    load_feature_datasets,
    normalize_keyword,
    build_eval_record,
    GenerationEvalCallback,
)


# ============================================================================
# Model Loading for Evaluation
# ============================================================================

def load_trained_model(model_path, model_name, load_in_4bit):
    """Load a trained model with LoRA adapters for evaluation."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    lora_model_path = model_path / "lora_model"
    if not lora_model_path.exists():
        raise FileNotFoundError(f"LoRA model path does not exist: {lora_model_path}")

    # Load base model with same config as training
    print("Loading base model with optimizations...")
    bnb_config = build_bnb_config(load_in_4bit)
    
    base_model = load_model(model_name, bnb_config)
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from {lora_model_path}...")
    model = PeftModel.from_pretrained(base_model, str(lora_model_path))
    model.eval()
    print("\nMemory usage after model loading:")
    print_gpu_memory()
    
    return model


# ============================================================================
# Evaluation Pipeline
# ============================================================================

def build_eval_dataset(datasets_dict):
    """Build evaluation dataset from datasets dictionary (only eval data)."""
    all_eval_data = []
    
    for feature_name, feature_data in datasets_dict.items():
        eval_df = feature_data["eval"]
        if len(eval_df) > 0:
            # Use the same logic as build_examples_from_datasets but only for eval
            from pt_features import PtFeaturesMeta, PtDateFeatureBase
            
            if feature_name not in PtFeaturesMeta.registry:
                continue
            
            target_cls = PtFeaturesMeta.registry[feature_name]
            if issubclass(target_cls, PtDateFeatureBase):
                continue  # skip date features
            
            for _, row in eval_df.iterrows():
                keyword = normalize_keyword(row["found_keywords"])
                all_eval_data.append(
                    build_eval_record(
                        target_cls=target_cls,
                        chunk=row["chunk"],
                        keyword=keyword,
                        val_unified=row["val_unified"],
                        feature_name=feature_name,
                    )
                )
    
    return all_eval_data


def summarize_eval_dataset(all_eval_data, datasets_dict, downsample_eval, data_source=None):
    """Print evaluation dataset summary."""
    print("\n" + "=" * 80)
    print("Dataset Summary (before downsampling):")
    print(f"  Total eval examples: {len(all_eval_data)} (from {len(datasets_dict)} features)")
    if data_source:
        print(f"  Data source: {data_source}")
    if downsample_eval is not None:
        print("\nDownsampling settings:")
        print(f"  Eval: {downsample_eval} examples")
    print("=" * 80)


def run_evaluation(model, eval_callback):
    """Run evaluation using the callback."""
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    if eval_callback and eval_callback.eval_dataset_raw:
        print(f"  Eval dataset size: {len(eval_callback.eval_dataset_raw)}")
    print("=" * 80)

    # Create a dummy state object for the callback
    class DummyState:
        def __init__(self):
            self.global_step = 0

    dummy_state = DummyState()

    # Run the evaluation
    eval_callback.on_evaluate(
        args=None,
        state=dummy_state,
        control=None,
        model=model,
    )


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def main():
    """Main evaluation pipeline."""
    # Ensure single GPU
    ensure_single_gpu()

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Evaluate trained SFT model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--downsample_eval", type=int, default=None, help="Downsample eval set"
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default=None,
        choices=["mgb", "mimic"],
        help="Filter evaluation data by data source (mgb or mimic). Overrides config file setting.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Load config file
    config = load_config(Path("config_sft.yaml"))

    # Create a dummy args object for config resolution
    class DummyArgs:
        def __init__(self, data_source_override=None):
            # Model args
            self.model_name = None
            self.max_seq_length = None
            self.dtype = None
            self.load_in_4bit = None
            # Dataset args - use CLI override if provided
            self.data_source = data_source_override
            self.train_split = None
            self.random_state_dataset = None
            self.downsample_train = None
            self.downsample_eval = None

    dummy_args = DummyArgs(data_source_override=args.data_source)

    # Resolve configurations
    model_config = resolve_model_config(dummy_args, config)
    dataset_config = resolve_dataset_config(dummy_args, config)
    
    # Print data source being used
    if args.data_source:
        print(f"Using data source override: {args.data_source}")

    print("=" * 80)
    print("Evaluating Qwen 2.5 7B Instruct with LoRA")
    print("=" * 80)
    print(f"Model: {model_config['model_name']}")
    print(f"Model path: {model_path}")
    print("=" * 80)

    # Load tokenizer
    tokenizer = load_tokenizer(model_config["model_name"])

    # Load trained model
    model = load_trained_model(
        model_path, model_config["model_name"], model_config["load_in_4bit"]
    )

    # Load datasets
    datasets_dict = load_feature_datasets(dataset_config)

    # Build eval dataset (only eval data, not training data)
    all_eval_data = build_eval_dataset(datasets_dict)

    # Summarize dataset
    summarize_eval_dataset(
        all_eval_data, datasets_dict, args.downsample_eval, dataset_config["data_source"]
    )

    # Convert to HuggingFace dataset and apply downsampling
    eval_dataset_raw = Dataset.from_list(all_eval_data) if all_eval_data else None

    if eval_dataset_raw is None or len(eval_dataset_raw) == 0:
        raise ValueError("No eval data available")

    if args.downsample_eval is not None and len(eval_dataset_raw) > args.downsample_eval:
        print(
            f"\nDownsampling eval dataset from {len(eval_dataset_raw)} to {args.downsample_eval} examples..."
        )
        eval_dataset_raw = eval_dataset_raw.select(range(args.downsample_eval))

    # Create output directory for evaluation reports
    eval_output_dir = model_path / "eval_reports"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluation callback
    eval_callback = GenerationEvalCallback(
        eval_dataset_raw=eval_dataset_raw,
        alpaca_prompt=ALPACA_PROMPT,
        tokenizer=tokenizer,
        output_dir=eval_output_dir,
        data_source=dataset_config["data_source"],
    )

    # Run evaluation
    run_evaluation(model, eval_callback)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
