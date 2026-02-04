#!/usr/bin/env python3
"""
Train Qwen 2.5 7B Instruct using PEFT and LoRA on single GPU (production version)

CUDA_VISIBLE_DEVICES=3 python train_sft_7b.py \
  --downsample_train 512 \
  --downsample_eval 128 \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Fix tokenizers parallelism warning when using multiprocessing data loaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datasets import Dataset
from utils import get_dataset
from pt_features import PtFeaturesMeta
import pandas as pd
import wandb
import yaml
from pt_features import PtDateFeatureBase, PtFeatureBase

# ============================================================================
# Constants
# ============================================================================

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


# ============================================================================
# GPU Utilities
# ============================================================================

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
            f"{total:.2f}GB total"
        )


def ensure_single_gpu():
    """Ensure single GPU mode and check CUDA availability."""
    if "LOCAL_RANK" in os.environ:
        raise RuntimeError(
            "This script is for single GPU training only. Do not use torchrun."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please ensure CUDA-capable GPU is accessible."
        )

    print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    print_gpu_memory()


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path):
    """Load YAML configuration file."""
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Config file not found: {config_path}")


def get_config_value(section, key, cli_value=None, default=None, config=None):
    """Get config value, prioritizing CLI args, then config file, then default"""
    if cli_value is not None:
        return cli_value
    if config and section in config and key in config[section]:
        value = config[section][key]
        return None if value == "null" or value is None else value
    return default


def validate_required(value, name):
    """Validate that a required value is not None."""
    if value is None:
        raise ValueError(f"{name} must be specified in config file or via CLI")


def build_arg_parser():
    """Build argument parser for CLI overrides."""
    parser = argparse.ArgumentParser(
        description="Train SFT model with config file and optional CLI overrides"
    )
    # Model args
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument(
        "--max_seq_length", type=int, default=None, help="Maximum sequence length"
    )
    parser.add_argument("--dtype", type=str, default=None, help="Data type (null for auto)")
    parser.add_argument(
        "--load_in_4bit",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Load in 4-bit",
    )
    # LoRA args
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=None, help="LoRA dropout")
    parser.add_argument(
        "--lora_lr",
        type=float,
        default=None,
        help="LoRA learning rate (defaults to 10x learning_rate)",
    )
    parser.add_argument("--lora_bias", type=str, default=None, help="LoRA bias")
    parser.add_argument("--pretrained_lora", type=str, default=None, help="Path to pretrained LoRA adapter to load")
    parser.add_argument("--random_state", type=int, default=None, help="Random state")
    # Training args
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=None,
        help="Eval batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Warmup steps (deprecated, use warmup_ratio)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="Warmup ratio (fraction of total steps)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--optim", type=str, default=None, help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default=None, help="LR scheduler type"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=None, help="Logging steps")
    parser.add_argument(
        "--evals_per_epoch", type=int, default=None, help="Number of evaluations per epoch"
    )
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluation steps")
    # Dataset args
    parser.add_argument("--data_source", type=str, default=None, help="Data source")
    parser.add_argument("--train_split", type=float, default=None, help="Train split ratio")
    parser.add_argument("--downsample_train", type=int, default=None, help="Downsample training set")
    parser.add_argument("--downsample_eval", type=int, default=None, help="Downsample eval set")
    parser.add_argument(
        "--random_state_dataset",
        type=int,
        default=None,
        help="Dataset random state",
    )
    # Wandb args
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument(
        "--wandb_enabled",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Enable wandb",
    )
    # Experiment args
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default",
        help="Experiment name (creates subdirectory in training_runs)",
    )
    parser.add_argument(
        "--pretrain"
        , action="store_true", help="If set, perform pretraining (no answers in prompts)"
    )
    return parser


def resolve_model_config(args_cli, config):
    """Resolve model configuration from CLI args and config file."""
    model_name = get_config_value("model", "model_name", args_cli.model_name, config=config)
    max_seq_length = get_config_value("model", "max_seq_length", args_cli.max_seq_length, config=config)
    dtype = get_config_value("model", "dtype", args_cli.dtype, config=config)
    load_in_4bit = get_config_value("model", "load_in_4bit", args_cli.load_in_4bit, config=config)

    validate_required(model_name, "model.model_name")
    validate_required(max_seq_length, "model.max_seq_length")
    validate_required(load_in_4bit, "model.load_in_4bit")

    return {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
    }


def resolve_lora_config(args_cli, config):
    """Resolve LoRA configuration from CLI args and config file."""
    lora_r = get_config_value("lora", "r", args_cli.lora_r, config=config)
    lora_alpha = get_config_value("lora", "lora_alpha", args_cli.lora_alpha, config=config)
    lora_dropout = get_config_value("lora", "lora_dropout", args_cli.lora_dropout, config=config)
    lora_lr = get_config_value("lora", "lora_lr", args_cli.lora_lr, config=config)
    lora_bias = get_config_value("lora", "bias", args_cli.lora_bias, config=config)
    random_state = get_config_value("lora", "random_state", args_cli.random_state, config=config)

    validate_required(lora_r, "lora.r")
    validate_required(lora_alpha, "lora.lora_alpha")
    validate_required(lora_dropout, "lora.lora_dropout")
    validate_required(lora_bias, "lora.bias")
    validate_required(random_state, "lora.random_state")

    return {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_lr": lora_lr,
        "bias": lora_bias,
        "random_state": random_state,
    }


def resolve_dataset_config(args_cli, config):
    """Resolve dataset configuration from CLI args and config file."""
    data_source = get_config_value("dataset", "data_source", args_cli.data_source, config=config)
    feature_names = get_config_value("dataset", "feature_names", None, config=config)
    train_split = get_config_value("dataset", "train_split", args_cli.train_split, config=config)
    downsample = get_config_value("dataset", "downsample", None, config=config)
    random_state_dataset = get_config_value(
        "dataset", "random_state", args_cli.random_state_dataset, config=config
    )
    downsample_train = get_config_value(
        "dataset", "downsample_train", args_cli.downsample_train, None, config=config
    )
    downsample_eval = get_config_value(
        "dataset", "downsample_eval", args_cli.downsample_eval, None, config=config
    )

    validate_required(data_source, "dataset.data_source")
    validate_required(train_split, "dataset.train_split")
    validate_required(random_state_dataset, "dataset.random_state")

    return {
        "data_source": data_source,
        "feature_names": feature_names,
        "train_split": train_split,
        "downsample": downsample,
        "random_state": random_state_dataset,
        "downsample_train": downsample_train,
        "downsample_eval": downsample_eval,
    }


def resolve_training_config(args_cli, config):
    """Resolve training configuration from CLI args and config file."""
    per_device_train_batch_size = get_config_value(
        "training", "per_device_train_batch_size", args_cli.per_device_train_batch_size, config=config
    )
    per_device_eval_batch_size = get_config_value(
        "training", "per_device_eval_batch_size", args_cli.per_device_eval_batch_size, config=config
    )
    gradient_accumulation_steps = get_config_value(
        "training", "gradient_accumulation_steps", args_cli.gradient_accumulation_steps, config=config
    )
    warmup_steps = get_config_value("training", "warmup_steps", args_cli.warmup_steps, config=config)
    warmup_ratio = get_config_value("training", "warmup_ratio", args_cli.warmup_ratio, config=config)
    epochs = get_config_value("training", "epochs", args_cli.epochs, config=config)
    max_steps = get_config_value("training", "max_steps", args_cli.max_steps, config=config)
    learning_rate = get_config_value("training", "learning_rate", args_cli.learning_rate, config=config)
    optim = get_config_value("training", "optim", args_cli.optim, config=config)
    weight_decay = get_config_value("training", "weight_decay", args_cli.weight_decay, config=config)
    lr_scheduler_type = get_config_value(
        "training", "lr_scheduler_type", args_cli.lr_scheduler_type, config=config
    )
    seed = get_config_value("training", "seed", args_cli.seed, config=config)
    logging_steps = get_config_value("training", "logging_steps", args_cli.logging_steps, config=config)
    evals_per_epoch = get_config_value(
        "training", "evals_per_epoch", args_cli.evals_per_epoch, config=config
    )
    eval_steps = get_config_value("training", "eval_steps", args_cli.eval_steps, config=config)

    validate_required(per_device_train_batch_size, "training.per_device_train_batch_size")
    validate_required(per_device_eval_batch_size, "training.per_device_eval_batch_size")
    validate_required(gradient_accumulation_steps, "training.gradient_accumulation_steps")
    if warmup_steps is None and warmup_ratio is None:
        raise ValueError(
            "Either training.warmup_steps or training.warmup_ratio must be specified"
        )
    if warmup_steps is not None and warmup_ratio is not None:
        print("\nWarning: Both warmup_steps and warmup_ratio specified - using warmup_ratio")
        warmup_steps = None
    validate_required(epochs, "training.epochs")
    validate_required(learning_rate, "training.learning_rate")
    validate_required(optim, "training.optim")
    validate_required(weight_decay, "training.weight_decay")
    validate_required(lr_scheduler_type, "training.lr_scheduler_type")
    validate_required(seed, "training.seed")
    validate_required(logging_steps, "training.logging_steps")
    validate_required(evals_per_epoch, "training.evals_per_epoch")

    return {
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "epochs": epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "optim": optim,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "seed": seed,
        "logging_steps": logging_steps,
        "evals_per_epoch": evals_per_epoch,
        "eval_steps": eval_steps,
    }


def resolve_wandb_config(args_cli, config):
    """Resolve wandb configuration from CLI args and config file."""
    wandb_project = get_config_value("wandb", "project", args_cli.wandb_project, config=config)
    wandb_enabled = get_config_value("wandb", "enabled", args_cli.wandb_enabled, config=config)

    validate_required(wandb_project, "wandb.project")
    validate_required(wandb_enabled, "wandb.enabled")

    return {"project": wandb_project, "enabled": wandb_enabled}


# ============================================================================
# Model Loading and Configuration
# ============================================================================

def load_tokenizer(model_name):
    """Load and configure tokenizer."""
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_bnb_config(load_in_4bit):
    """Build BitsAndBytes configuration for quantization."""
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
    )


def load_model(model_name, bnb_config):
    """Load model with optimizations."""
    print("Loading model with optimizations...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Single GPU
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
    )
    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # More memory efficient
    )


def build_lora_config(lora_config):
    """Build LoRA configuration."""
    return LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type="CAUSAL_LM",
    )


def add_lora_adapters(model, lora_config):
    """Add LoRA adapters to model."""
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("\nMemory usage after model loading:")
    print_gpu_memory()
    return model


# ============================================================================
# Dataset Processing
# ============================================================================

def load_feature_datasets(dataset_config):
    """Load feature datasets from data source."""
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)
    datasets_dict = get_dataset(
        data_source=dataset_config["data_source"],
        feature_names=dataset_config["feature_names"],  # Load all features if None
        train_split=dataset_config["train_split"],
        downsample=dataset_config["downsample"],
        random_state=dataset_config["random_state"],
    )
    print(f"\nLoaded {len(datasets_dict)} features: {list(datasets_dict.keys())}")
    return datasets_dict


def normalize_keyword(found_keywords):
    """Normalize keyword field from dataset."""
    if isinstance(found_keywords, list):
        return found_keywords[0] if found_keywords else ""
    if pd.notna(found_keywords):
        return str(found_keywords)
    return ""


def build_training_record(target_cls, chunk, keyword, val_unified, feature_name):
    """Build a training record from feature data."""
    if issubclass(target_cls, PtDateFeatureBase):
        format_instruction = (
            "Respond ONLY with a single letter or with the date in the format of YYYYMMDD, "
            "nothing else."
        )
    else:
        format_instruction = "Respond ONLY with a single letter, nothing else."
    instruction = target_cls.query(chunk=chunk, keyword=keyword) + " " + format_instruction
    input_text = str(chunk) if pd.notna(chunk) else ""
    output_text = str(val_unified) if pd.notna(val_unified) else ""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "feature": feature_name,
    }


def build_eval_record(target_cls, chunk, keyword, val_unified, feature_name):
    """Build an evaluation record from feature data."""
    instruction = target_cls.query(chunk=chunk, keyword=keyword)
    input_text = str(chunk) if pd.notna(chunk) else ""
    output_text = str(val_unified) if pd.notna(val_unified) else ""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "feature": feature_name,
    }


def build_examples_from_datasets(datasets_dict):
    """Build training and evaluation examples from datasets."""
    all_train_data = []
    all_eval_data = []

    for feature_name, feature_data in datasets_dict.items():
        print(f"\nProcessing feature: {feature_name}")

        if feature_name not in PtFeaturesMeta.registry:
            print(f"  Warning: Feature {feature_name} not found in registry, skipping...")
            continue

        target_cls = PtFeaturesMeta.registry[feature_name]
        if issubclass(target_cls, PtDateFeatureBase):
            continue  # skip date features because we don't have MIMIC data for them

        train_df = feature_data["train"]
        if len(train_df) > 0:
            print(f"  Processing {len(train_df)} train examples...")
            for _, row in train_df.iterrows():
                keyword = normalize_keyword(row["found_keywords"])
                all_train_data.append(
                    build_training_record(
                        target_cls=target_cls,
                        chunk=row["chunk"],
                        keyword=keyword,
                        val_unified=row["val_unified"],
                        feature_name=feature_name,
                    )
                )

        eval_df = feature_data["eval"]
        if len(eval_df) > 0:
            print(f"  Processing {len(eval_df)} eval examples...")
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

    return all_train_data, all_eval_data


def summarize_dataset(all_train_data, all_eval_data, datasets_dict, downsample_train, downsample_eval, data_source=None, epochs=None, output_dir=None):
    """Save dataset summary to markdown file in the training runs directory."""
    summary_lines = []
    summary_lines.append("# Dataset Summary")
    summary_lines.append("")
    if data_source:
        summary_lines.append(f"- **Dataset**: {data_source}")
    if epochs is not None:
        summary_lines.append(f"- **Epochs**: {epochs}")
    summary_lines.append("")
    summary_lines.append("## Dataset Statistics (before downsampling)")
    summary_lines.append("")
    summary_lines.append(f"- **Total train examples**: {len(all_train_data)} (from {len(datasets_dict)} features)")
    summary_lines.append(f"- **Total eval examples**: {len(all_eval_data)} (from {len(datasets_dict)} features)")
    summary_lines.append("- **Training strategy**: Training ONE model on ALL features combined")
    
    if downsample_train is not None or downsample_eval is not None:
        summary_lines.append("")
        summary_lines.append("## Downsampling Settings")
        summary_lines.append("")
        if downsample_train is not None:
            summary_lines.append(f"- **Train**: {downsample_train} examples")
        if downsample_eval is not None:
            summary_lines.append(f"- **Eval**: {downsample_eval} examples")
    
    summary_text = "\n".join(summary_lines)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / "dataset_summary.md"
        with open(summary_file, "w") as f:
            f.write(summary_text)
        print(f"\nDataset summary saved to: {summary_file}")


def downsample_dataset(dataset, limit, label, seed):
    """Downsample dataset to specified limit with random sampling."""
    if limit is not None and len(dataset) > limit:
        print(f"\nDownsampling {label} dataset from {len(dataset)} to {limit} examples (seed={seed})...")
        return dataset.shuffle(seed=seed).select(range(limit))
    return dataset


def format_datasets(train_dataset, eval_dataset, tokenizer, alpaca_prompt=ALPACA_PROMPT):
    """Format datasets with Alpaca prompt template."""
    eos_token = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input_text, output_text) + eos_token
            texts.append(text)
        return {"text": texts}

    print("\nFormatting datasets...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    if eval_dataset:
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    if len(train_dataset) > 0:
        print("\nSample formatted training text:")
        sample_text = train_dataset[0]["text"]
        print(f"  Length: {len(sample_text)} characters")
        print(f"  Preview: {sample_text[:200]}...")

    return train_dataset, eval_dataset


def build_tokenize_function(tokenizer, max_seq_length):
    """Build tokenization function for datasets."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # Dynamic padding via data collator
        )
    return tokenize_function


def tokenize_datasets(train_dataset, eval_dataset, tokenizer, max_seq_length):
    """Tokenize datasets."""
    print("\nTokenizing datasets...")
    tokenize_function = build_tokenize_function(tokenizer, max_seq_length)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    if eval_dataset:
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    return train_dataset, eval_dataset


# ============================================================================
# Evaluation Callback
# ============================================================================

class GenerationEvalCallback(TrainerCallback):
    """Callback for generation-based evaluation during training."""
    
    def __init__(self, eval_dataset_raw, alpaca_prompt, tokenizer, output_dir, data_source=None):
        # Store the raw eval dataset (before formatting) to access instruction, input, output
        self.eval_dataset_raw = eval_dataset_raw
        self.alpaca_prompt = alpaca_prompt
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.data_source = data_source
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or self.eval_dataset_raw is None or len(self.eval_dataset_raw) == 0:
            return

        print(f"\n{'='*80}")
        print("Running Generation-based Evaluation...")
        print(f"{'='*80}")

        # Set model to eval mode
        was_training = model.training
        model.eval()

        correct = 0
        total = 0
        wrong_answers = []

        with torch.no_grad():
            for idx, example in enumerate(self.eval_dataset_raw):
                # Format prompt without the answer
                prompt = self.alpaca_prompt.format(
                    example["instruction"],
                    example["input"],
                    "",
                )

                # Get expected output
                expected = str(example["output"]).strip()

                # Tokenize the expected output to determine how many tokens to generate
                expected_tokens = self.tokenizer.encode(expected, add_special_tokens=False)
                num_tokens_to_generate = len(expected_tokens)

                # Generate with some buffer (at least 1, max 20 for safety)
                # max_new_tokens = min(20, max(1, num_tokens_to_generate + 2))
                max_new_tokens = 1

                # Tokenize the prompt
                inputs = self.tokenizer([prompt], return_tensors="pt").to(model.device)

                # Generate with temperature=0 for deterministic outputs
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,  # Disable temperature for greedy
                    pad_token_id=self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

                # Decode only the generated tokens (skip the prompt)
                generated_text = self.tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0].strip()

                # Exact match comparison
                if generated_text == expected:
                    correct += 1
                else:
                    # Store wrong answer details
                    wrong_answers.append({
                        "index": idx,
                        "feature": example["feature"],
                        "instruction": example["instruction"],
                        "input": example["input"],
                        "ground_truth": expected,
                        "generated": generated_text,
                    })
                total += 1

                # Print progress and example every 10 examples
                if (idx + 1) % 10 == 0 or idx == 0:
                    current_acc = correct / total if total > 0 else 0
                    print(
                        f"  Evaluated {idx + 1}/{len(self.eval_dataset_raw)} examples "
                        f"(accuracy so far: {current_acc:.3f})"
                    )
                    print(f"    Example #{idx}: Feature={example['feature']}")
                    print(f"      Ground Truth: '{expected}'")
                    print(f"      Generated:    '{generated_text}'")
                    print(f"      Match: {'✓' if generated_text == expected else '✗'}")

                # Clear CUDA cache periodically
                if (idx + 1) % 5 == 0:
                    del inputs, outputs
                    torch.cuda.empty_cache()
                else:
                    del inputs, outputs

        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0.0

        # Log the metric
        print(f"\n{'='*80}")
        print("Generation-based Evaluation:")
        print(f"  Exact Match Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Wrong Answers: {len(wrong_answers)}")
        print(f"{'='*80}\n")

        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "eval/exact_match": accuracy,
                "eval/correct": correct,
                "eval/wrong": len(wrong_answers),
                "eval/total": total,
            })

        # Save wrong answers to markdown file
        if wrong_answers:
            step = state.global_step if hasattr(state, "global_step") else 0
            reports_dir = self.output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            data_source_suffix = f"_{self.data_source}" if self.data_source else ""
            report_path = reports_dir / f"wrong_answers_step_{step:05d}{data_source_suffix}.md"

            with open(report_path, "w") as f:
                f.write("# Evaluation Wrong Answers Report\n\n")
                f.write("## Metadata\n\n")
                f.write(f"- **Training Step**: {step}\n")
                f.write(
                    f"- **Evaluation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"- **Total Examples**: {total}\n")
                f.write(f"- **Correct**: {correct}\n")
                f.write(f"- **Wrong**: {len(wrong_answers)}\n")
                f.write(f"- **Accuracy**: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
                f.write("---\n\n")

                f.write("## Wrong Answers\n\n")
                for i, wrong in enumerate(wrong_answers, 1):
                    f.write(f"### Wrong Answer {i}\n\n")
                    f.write(f"- **Index**: {wrong['index']}\n")
                    f.write(f"- **Feature**: {wrong['feature']}\n")
                    f.write(f"- **Ground Truth**: `{wrong['ground_truth']}`\n")
                    f.write(f"- **Generated**: `{wrong['generated']}`\n")
                    f.write(f"- **Instruction**: {wrong['instruction']}\n")
                    f.write(f"- **Input**:\n")
                    input_text = wrong["input"]
                    if len(input_text) > 500:
                        f.write(f"  ```\n  {input_text[:500]}...\n  ```\n")
                    else:
                        f.write(f"  ```\n  {input_text}\n  ```\n")
                    f.write("\n")

            print(f"Saved wrong answers report to: {report_path}")

        # Restore original training mode
        if was_training:
            model.train()


def build_eval_callback(all_eval_data, downsample_eval, tokenizer, training_runs_dir, alpaca_prompt=ALPACA_PROMPT, data_source=None, seed=None):
    """Build evaluation callback from eval data."""
    eval_dataset_raw = Dataset.from_list(all_eval_data) if all_eval_data else None
    if eval_dataset_raw is not None and downsample_eval is not None and len(eval_dataset_raw) > downsample_eval:
        eval_dataset_raw = eval_dataset_raw.shuffle(seed=seed).select(range(downsample_eval))

    if eval_dataset_raw is None:
        return None

    return GenerationEvalCallback(
        eval_dataset_raw=eval_dataset_raw,
        alpaca_prompt=alpaca_prompt,
        tokenizer=tokenizer,
        output_dir=training_runs_dir,
        data_source=data_source,
    )


# ============================================================================
# Training Configuration and Setup
# ============================================================================

def compute_steps_per_epoch(train_dataset, per_device_train_batch_size, gradient_accumulation_steps):
    """Compute steps per epoch based on dataset size and batch configuration."""
    effective_batch_size = min(per_device_train_batch_size, len(train_dataset))
    if effective_batch_size < per_device_train_batch_size:
        print(
            "\nWarning: Reducing batch size from "
            f"{per_device_train_batch_size} to {effective_batch_size} "
            f"to match dataset size ({len(train_dataset)} examples)"
        )

    total_batch_size = effective_batch_size * gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // total_batch_size
    if len(train_dataset) % total_batch_size != 0:
        steps_per_epoch += 1

    return effective_batch_size, total_batch_size, steps_per_epoch


def finalize_training_steps(training_config, steps_per_epoch, train_dataset):
    """Finalize max_steps in training config based on epochs."""
    max_steps = training_config["max_steps"]
    epochs = training_config["epochs"]
    if max_steps is None:
        max_steps = steps_per_epoch * epochs
        print(f"\n{'='*80}")
        print(f"max_steps not specified - auto-calculated for {epochs} epoch(s):")
        print(f"  Dataset size: {len(train_dataset)}")
        print(f"  Batch size: {training_config['effective_batch_size']}")
        print(f"  Gradient accumulation: {training_config['gradient_accumulation_steps']}")
        print(f"  Effective batch size: {training_config['total_batch_size']}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total steps ({epochs} epochs): {max_steps}")
        print(f"{'='*80}\n")
    training_config["max_steps"] = max_steps


def finalize_eval_steps(training_config, eval_dataset, steps_per_epoch):
    """Finalize eval_steps in training config based on evals_per_epoch."""
    eval_steps = training_config["eval_steps"]
    if eval_steps is None and eval_dataset:
        eval_steps = max(1, steps_per_epoch // training_config["evals_per_epoch"])
        training_config["eval_steps"] = eval_steps
        print(
            "eval_steps not specified - auto-calculated for "
            f"{training_config['evals_per_epoch']} evals per epoch: {eval_steps}"
        )
        print(f"  (steps_per_epoch: {steps_per_epoch}, eval_steps: {eval_steps})")
        print()


def build_training_args(training_runs_dir, training_config, eval_dataset, wandb_enabled):
    """Build TrainingArguments from configuration."""
    training_args_dict = {
        "output_dir": str(training_runs_dir),
        "per_device_train_batch_size": training_config["effective_batch_size"],
        "per_device_eval_batch_size": training_config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "max_steps": training_config["max_steps"],
        "learning_rate": training_config["learning_rate"],
        "fp16": False,
        "bf16": True,
        "logging_steps": training_config["logging_steps"],
        "optim": training_config["optim"],
        "weight_decay": training_config["weight_decay"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "seed": training_config["seed"],
        "report_to": "wandb" if wandb_enabled else "none",
        "eval_strategy": "steps" if eval_dataset else "no",
        "eval_steps": training_config["eval_steps"] if eval_dataset else None,
        "dataloader_drop_last": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_steps": 100,
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "dataloader_pin_memory": True,
        "dataloader_num_workers": 2,
        "max_grad_norm": 1.0,
        "tf32": True
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else False,
    }

    if training_config["warmup_ratio"] is not None:
        training_args_dict["warmup_ratio"] = training_config["warmup_ratio"]
        print(
            f"Using warmup_ratio: {training_config['warmup_ratio']} "
            f"({int(training_config['warmup_ratio'] * training_config['max_steps'])} "
            f"steps out of {training_config['max_steps']} total)"
        )
    else:
        training_args_dict["warmup_steps"] = training_config["warmup_steps"]
        print(
            f"Using warmup_steps: {training_config['warmup_steps']} "
            f"out of {training_config['max_steps']} total steps"
        )

    return TrainingArguments(**training_args_dict)


def build_data_collator(tokenizer):
    """Build data collator for language modeling."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
    )


def build_trainer(model, training_args, train_dataset, eval_dataset, data_collator, eval_callback):
    """Build Trainer instance."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[eval_callback] if eval_callback is not None else None,
    )


# ============================================================================
# Wandb Integration
# ============================================================================

def init_wandb(wandb_config, training_runs_dir, timestamp, config_payload):
    """Initialize wandb logging."""
    if not wandb_config["enabled"]:
        return
    wandb.init(
        project=wandb_config["project"],
        name=f"sft-7b-{timestamp}",
        config=config_payload,
        dir=str(training_runs_dir),
    )


def build_wandb_payload(
    model_config,
    training_config,
    dataset_config,
    datasets_dict,
    train_dataset,
    eval_dataset,
):
    """Build payload for wandb logging."""
    payload = {
        "model_name": model_config["model_name"],
        "max_seq_length": model_config["max_seq_length"],
        "load_in_4bit": model_config["load_in_4bit"],
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "epochs": training_config["epochs"],
        "max_steps": training_config["max_steps"],
        "steps_per_epoch": training_config["steps_per_epoch"],
        "evals_per_epoch": training_config["evals_per_epoch"],
        "eval_steps": training_config["eval_steps"],
        "learning_rate": training_config["learning_rate"],
        "optim": training_config["optim"],
        "weight_decay": training_config["weight_decay"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "seed": training_config["seed"],
        "train_dataset_size": len(train_dataset),
        "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
        "downsample_train": dataset_config["downsample_train"],
        "downsample_eval": dataset_config["downsample_eval"],
        "num_features": len(datasets_dict),
    }
    if training_config["warmup_ratio"] is not None:
        payload["warmup_ratio"] = training_config["warmup_ratio"]
        payload["warmup_steps_calculated"] = int(
            training_config["warmup_ratio"] * training_config["max_steps"]
        )
    else:
        payload["warmup_steps"] = training_config["warmup_steps"]
    return payload


# ============================================================================
# Inference and Model Saving
# ============================================================================

def run_inference_sample(model, tokenizer, all_eval_data, alpaca_prompt=ALPACA_PROMPT):
    """Run a sample inference to test the model."""
    print("\n" + "=" * 80)
    print("Testing inference...")
    print("=" * 80)

    if all_eval_data:
        sample = all_eval_data[0]
        test_instruction = sample["instruction"]
        test_input = sample["input"][:500]
        print(f"\nTest example (feature: {sample['feature']}):")
        print(f"  Instruction: {test_instruction[:150]}...")
        print(f"  Input: {test_input[:150]}...")
        print(f"  Expected output: {sample['output']}")

        test_prompt = alpaca_prompt.format(test_instruction, test_input, "")
        inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")

        print("\nGenerating prediction...")
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True, temperature=0.7)
        prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"\nGenerated prediction:\n{prediction}")
    else:
        print("No eval data available for inference test")


def save_model(trainer, tokenizer, training_runs_dir):
    """Save trained model and tokenizer."""
    print("\nSaving model...")
    trainer.save_model(str(training_runs_dir / "lora_model"))
    tokenizer.save_pretrained(str(training_runs_dir / "lora_model"))
    print("Training complete!")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training pipeline."""
    ensure_single_gpu()

    config = load_config(Path("config_sft.yaml"))

    parser = build_arg_parser()
    args_cli = parser.parse_args()

    model_config = resolve_model_config(args_cli, config)
    lora_config = resolve_lora_config(args_cli, config)
    dataset_config = resolve_dataset_config(args_cli, config)
    training_config = resolve_training_config(args_cli, config)
    wandb_config = resolve_wandb_config(args_cli, config)

    print("=" * 80)
    print("Training Qwen 2.5 7B Instruct with LoRA (PRODUCTION)")
    print("=" * 80)
    print(f"Model: {model_config['model_name']}")
    print(f"LoRA rank: {lora_config['r']}")
    print(f"LoRA alpha: {lora_config['lora_alpha']}")
    print("=" * 80)

    tokenizer = load_tokenizer(model_config["model_name"])
    bnb_config = build_bnb_config(model_config["load_in_4bit"])
    model = load_model(model_config["model_name"], bnb_config)

    if args_cli.pretrained_lora:
        print(f"Loading pretrained LoRA adapter from: {args_cli.pretrained_lora}")
        model = PeftModel.from_pretrained(
            model, 
            args_cli.pretrained_lora + "/lora_model",
            is_trainable=True
        )
        model.train()
        model.print_trainable_parameters()
        print("\nMemory usage after loading pretrained LoRA:")
        print_gpu_memory()
    else:
        lora_cfg = build_lora_config(lora_config)
        model = add_lora_adapters(model, lora_cfg)

    datasets_dict = load_feature_datasets(dataset_config)
    all_train_data, all_eval_data = build_examples_from_datasets(datasets_dict)

    # Create training run directory early so we can save dataset summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args_cli.experiment_name if args_cli.experiment_name else "default"
    training_runs_dir = Path("training_runs") / experiment_name / timestamp
    training_runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining run directory: {training_runs_dir}")
    if experiment_name != "default":
        print(f"Experiment name: {experiment_name}")

    summarize_dataset(
        all_train_data,
        all_eval_data,
        datasets_dict,
        dataset_config["downsample_train"],
        dataset_config["downsample_eval"],
        data_source=dataset_config["data_source"],
        epochs=training_config["epochs"],
        output_dir=training_runs_dir,
    )

    train_dataset = Dataset.from_list(all_train_data)
    eval_dataset = Dataset.from_list(all_eval_data) if all_eval_data else None

    train_dataset = downsample_dataset(
        train_dataset, dataset_config["downsample_train"], "train", seed=dataset_config["random_state"]
    )
    if eval_dataset is not None:
        eval_dataset = downsample_dataset(
            eval_dataset, dataset_config["downsample_eval"], "eval", seed=dataset_config["random_state"]
        )
    # make response empty if pretraining
    if args_cli.pretrain:
        train_dataset = train_dataset.map(lambda x: {'output': ''})
    train_dataset, eval_dataset = format_datasets(train_dataset, eval_dataset, tokenizer)

    eval_callback = build_eval_callback(
        all_eval_data,
        dataset_config["downsample_eval"],
        tokenizer,
        training_runs_dir,
        data_source=dataset_config.get("data_source"),
    )

    if lora_config["lora_lr"] is None:
        lora_config["lora_lr"] = training_config["learning_rate"] * 10
        print(f"Setting LoRA learning rate to 10x base learning rate: {lora_config['lora_lr']}")

    effective_batch_size, total_batch_size, steps_per_epoch = compute_steps_per_epoch(
        train_dataset,
        training_config["per_device_train_batch_size"],
        training_config["gradient_accumulation_steps"],
    )
    training_config["effective_batch_size"] = effective_batch_size
    training_config["total_batch_size"] = total_batch_size
    training_config["steps_per_epoch"] = steps_per_epoch

    finalize_training_steps(training_config, steps_per_epoch, train_dataset)
    finalize_eval_steps(training_config, eval_dataset, steps_per_epoch)

    wandb_payload = build_wandb_payload(
        model_config,
        training_config,
        dataset_config,
        datasets_dict,
        train_dataset,
        eval_dataset,
    )
    init_wandb(wandb_config, training_runs_dir, timestamp, wandb_payload)

    print("\n" + "=" * 80)
    print("Starting training...")
    print(
        f"  Train dataset size: {len(train_dataset)}"
        + (
            f" (downsampled from {len(all_train_data)})"
            if dataset_config["downsample_train"] is not None
            and len(train_dataset) < len(all_train_data)
            else ""
        )
    )
    if eval_dataset:
        print(
            f"  Eval dataset size: {len(eval_dataset)}"
            + (
                f" (downsampled from {len(all_eval_data)})"
                if dataset_config["downsample_eval"] is not None
                and len(eval_dataset) < len(all_eval_data)
                else ""
            )
        )
    print("=" * 80)

    train_dataset, eval_dataset = tokenize_datasets(
        train_dataset,
        eval_dataset,
        tokenizer,
        model_config["max_seq_length"],
    )

    data_collator = build_data_collator(tokenizer)
    training_args = build_training_args(
        training_runs_dir, training_config, eval_dataset, wandb_config["enabled"]
    )

    trainer = build_trainer(
        model,
        training_args,
        train_dataset,
        eval_dataset,
        data_collator,
        eval_callback,
    )

    trainer.train()
    run_inference_sample(model, tokenizer, all_eval_data)
    save_model(trainer, tokenizer, training_runs_dir)


if __name__ == "__main__":
    main()
