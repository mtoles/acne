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

# Fix tokenizers parallelism warning when using multiprocessing data loaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datetime import datetime
from pathlib import Path
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
# Single GPU mode only - check for distributed mode and warn
if "LOCAL_RANK" in os.environ:
    raise RuntimeError("This script is for single GPU training only. Do not use torchrun.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure CUDA-capable GPU is accessible.")

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")

print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
print_gpu_memory()

# Load config file
config_path = Path("config_sft.yaml")
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError(f"Config file not found: {config_path}")

# Parse CLI arguments (all default to None to allow config override)
parser = argparse.ArgumentParser(description='Train SFT model with config file and optional CLI overrides')
# Model args
parser.add_argument('--model_name', type=str, default=None, help='Model name')
parser.add_argument('--max_seq_length', type=int, default=None, help='Maximum sequence length')
parser.add_argument('--dtype', type=str, default=None, help='Data type (null for auto)')
parser.add_argument('--load_in_4bit', type=lambda x: (str(x).lower() == 'true'), default=None, help='Load in 4-bit')
# LoRA args
parser.add_argument('--lora_r', type=int, default=None, help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=None, help='LoRA alpha')
parser.add_argument('--lora_dropout', type=float, default=None, help='LoRA dropout')
parser.add_argument('--lora_lr', type=float, default=None, help='LoRA learning rate (defaults to 10x learning_rate)')
parser.add_argument('--lora_bias', type=str, default=None, help='LoRA bias')
parser.add_argument('--random_state', type=int, default=None, help='Random state')
# Training args
parser.add_argument('--per_device_train_batch_size', type=int, default=None, help='Batch size per device')
parser.add_argument('--per_device_eval_batch_size', type=int, default=None, help='Eval batch size per device')
parser.add_argument('--gradient_accumulation_steps', type=int, default=None, help='Gradient accumulation steps')
parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup steps (deprecated, use warmup_ratio)')
parser.add_argument('--warmup_ratio', type=float, default=None, help='Warmup ratio (fraction of total steps)')
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
parser.add_argument('--max_steps', type=int, default=None, help='Maximum training steps')
parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
parser.add_argument('--optim', type=str, default=None, help='Optimizer')
parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
parser.add_argument('--lr_scheduler_type', type=str, default=None, help='LR scheduler type')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--logging_steps', type=int, default=None, help='Logging steps')
parser.add_argument('--evals_per_epoch', type=int, default=None, help='Number of evaluations per epoch')
parser.add_argument('--eval_steps', type=int, default=None, help='Evaluation steps')
# Dataset args
parser.add_argument('--data_source', type=str, default=None, help='Data source')
parser.add_argument('--train_split', type=float, default=None, help='Train split ratio')
parser.add_argument('--downsample_train', type=int, default=None, help='Downsample training set')
parser.add_argument('--downsample_eval', type=int, default=None, help='Downsample eval set')
parser.add_argument('--random_state_dataset', type=int, default=None, help='Dataset random state')
# Wandb args
parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
parser.add_argument('--wandb_enabled', type=lambda x: (str(x).lower() == 'true'), default=None, help='Enable wandb')
args_cli = parser.parse_args()

# Extract config values with CLI overrides
def get_config_value(section, key, cli_value=None, default=None):
    """Get config value, prioritizing CLI args, then config file, then default"""
    if cli_value is not None:
        return cli_value
    if section in config and key in config[section]:
        value = config[section][key]
        return None if value == "null" or value is None else value
    return default

# Data preparation
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Model configuration (must be specified in config file)
model_name = get_config_value("model", "model_name", args_cli.model_name)
max_seq_length = get_config_value("model", "max_seq_length", args_cli.max_seq_length)
dtype = get_config_value("model", "dtype", args_cli.dtype)
load_in_4bit = get_config_value("model", "load_in_4bit", args_cli.load_in_4bit)

# Validate required config
if model_name is None:
    raise ValueError("model_name must be specified in config file or via --model_name")
if max_seq_length is None:
    raise ValueError("max_seq_length must be specified in config file or via --max_seq_length")
if load_in_4bit is None:
    raise ValueError("load_in_4bit must be specified in config file or via --load_in_4bit")

# LoRA configuration (must be specified in config file)
lora_r = get_config_value("lora", "r", args_cli.lora_r)
lora_alpha = get_config_value("lora", "lora_alpha", args_cli.lora_alpha)
lora_dropout = get_config_value("lora", "lora_dropout", args_cli.lora_dropout)
lora_lr = get_config_value("lora", "lora_lr", args_cli.lora_lr)
lora_bias = get_config_value("lora", "bias", args_cli.lora_bias)
random_state = get_config_value("lora", "random_state", args_cli.random_state)

# Validate required LoRA config
if lora_r is None:
    raise ValueError("lora.r must be specified in config file or via --lora_r")
if lora_alpha is None:
    raise ValueError("lora.lora_alpha must be specified in config file or via --lora_alpha")
if lora_dropout is None:
    raise ValueError("lora.lora_dropout must be specified in config file or via --lora_dropout")
if lora_bias is None:
    raise ValueError("lora.bias must be specified in config file or via --lora_bias")
if random_state is None:
    raise ValueError("lora.random_state must be specified in config file or via --random_state")

print("="*80)
print("Training Qwen 2.5 7B Instruct with LoRA (PRODUCTION)")
print("="*80)
print(f"Model: {model_name}")
print(f"LoRA rank: {lora_r}")
print(f"LoRA alpha: {lora_alpha}")
print("="*80)

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True,
)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

EOS_TOKEN = tokenizer.eos_token

# Load model with 4-bit quantization for memory efficiency
print("Loading model with optimizations...")

# Configure 4-bit quantization with aggressive memory optimization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Single GPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
)

# Prepare model for k-bit training with gradient checkpointing
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}  # More memory efficient
)

# Configure LoRA
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=lora_dropout,
    bias=lora_bias,
    task_type="CAUSAL_LM",
)

# Add LoRA adapters
print("Adding LoRA adapters...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("\nMemory usage after model loading:")
print_gpu_memory()

# Dataset configuration (must be specified in config file)
data_source = get_config_value("dataset", "data_source", args_cli.data_source)
feature_names = get_config_value("dataset", "feature_names", None)
train_split = get_config_value("dataset", "train_split", args_cli.train_split)
downsample = get_config_value("dataset", "downsample", None)
random_state_dataset = get_config_value("dataset", "random_state", args_cli.random_state_dataset)

# Validate required dataset config
if data_source is None:
    raise ValueError("dataset.data_source must be specified in config file or via --data_source")
if train_split is None:
    raise ValueError("dataset.train_split must be specified in config file or via --train_split")
if random_state_dataset is None:
    raise ValueError("dataset.random_state must be specified in config file or via --random_state_dataset")

# Load datasets for all features
print("\n" + "="*80)
print("Loading datasets...")
print("="*80)
datasets_dict = get_dataset(
    data_source=data_source,
    feature_names=feature_names,  # Load all features if None
    train_split=train_split,
    downsample=downsample,
    random_state=random_state_dataset
)

print(f"\nLoaded {len(datasets_dict)} features: {list(datasets_dict.keys())}")

# Convert to training format
all_train_data = []
all_eval_data = []

for feature_name, feature_data in datasets_dict.items():
    print(f"\nProcessing feature: {feature_name}")
    
    # Get the feature class to access query method
    if feature_name not in PtFeaturesMeta.registry:
        print(f"  Warning: Feature {feature_name} not found in registry, skipping...")
        continue
    
    target_cls = PtFeaturesMeta.registry[feature_name]
    
    # Process train data
    train_df = feature_data["train"]
    if len(train_df) > 0:
        print(f"  Processing {len(train_df)} train examples...")
        for idx, row in train_df.iterrows():
            chunk = row["chunk"]
            found_keywords = row["found_keywords"]
            val_unified = row["val_unified"]
            
            # Handle found_keywords - could be string or list
            if isinstance(found_keywords, list):
                keyword = found_keywords[0] if found_keywords else ""
            else:
                keyword = str(found_keywords) if pd.notna(found_keywords) else ""
            
            # Generate instruction using feature's query method
            if issubclass(target_cls, PtDateFeatureBase):
                format_instruction = "Respond ONLY with a single letter or with the date in the format of YYYYMMDD, nothing else."
            else:
                format_instruction = "Respond ONLY with a single letter, nothing else."
            instruction = target_cls.query(chunk=chunk, keyword=keyword) + " " + format_instruction
            
            # Format the input - use chunk as input
            input_text = str(chunk) if pd.notna(chunk) else ""
            output_text = str(val_unified) if pd.notna(val_unified) else ""
            
            all_train_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "feature": feature_name
            })
    
    # Process eval data
    eval_df = feature_data["eval"]
    if len(eval_df) > 0:
        print(f"  Processing {len(eval_df)} eval examples...")
        for idx, row in eval_df.iterrows():
            chunk = row["chunk"]
            found_keywords = row["found_keywords"]
            val_unified = row["val_unified"]
            
            # Handle found_keywords - could be string or list
            if isinstance(found_keywords, list):
                keyword = found_keywords[0] if found_keywords else ""
            else:
                keyword = str(found_keywords) if pd.notna(found_keywords) else ""
            
            # Generate instruction using feature's query method
            instruction = target_cls.query(chunk=chunk, keyword=keyword)
            
            # Format the input - use chunk as input
            input_text = str(chunk) if pd.notna(chunk) else ""
            output_text = str(val_unified) if pd.notna(val_unified) else ""
            
            all_eval_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "feature": feature_name
            })

# Apply downsampling if specified (from config or CLI)
downsample_train = get_config_value("dataset", "downsample_train", args_cli.downsample_train, None)
downsample_eval = get_config_value("dataset", "downsample_eval", args_cli.downsample_eval, None)

print(f"\n" + "="*80)
print(f"Dataset Summary (before downsampling):")
print(f"  Total train examples: {len(all_train_data)} (from {len(datasets_dict)} features)")
print(f"  Total eval examples: {len(all_eval_data)} (from {len(datasets_dict)} features)")
print(f"  Training ONE model on ALL features combined")
if downsample_train is not None or downsample_eval is not None:
    print(f"\nDownsampling settings:")
    if downsample_train is not None:
        print(f"  Train: {downsample_train} examples")
    if downsample_eval is not None:
        print(f"  Eval: {downsample_eval} examples")
print("="*80)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_list(all_train_data)
eval_dataset = Dataset.from_list(all_eval_data) if all_eval_data else None

if downsample_train is not None and len(train_dataset) > downsample_train:
    print(f"\nDownsampling train dataset from {len(train_dataset)} to {downsample_train} examples...")
    train_dataset = train_dataset.select(range(downsample_train))

if eval_dataset is not None and downsample_eval is not None and len(eval_dataset) > downsample_eval:
    print(f"\nDownsampling eval dataset from {len(eval_dataset)} to {downsample_eval} examples...")
    eval_dataset = eval_dataset.select(range(downsample_eval))

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Format the datasets
print("\nFormatting datasets...")
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
if eval_dataset:
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

# Print sample formatted text
if len(train_dataset) > 0:
    print("\nSample formatted training text:")
    sample_text = train_dataset[0]["text"]
    print(f"  Length: {len(sample_text)} characters")
    print(f"  Preview: {sample_text[:200]}...")

# Create evaluation callback for generation-based metrics
class GenerationEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset_raw, alpaca_prompt, tokenizer, output_dir):
        # Store the raw eval dataset (before formatting) to access instruction, input, output
        self.eval_dataset_raw = eval_dataset_raw
        self.alpaca_prompt = alpaca_prompt
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or self.eval_dataset_raw is None or len(self.eval_dataset_raw) == 0:
            return
        
        print(f"\n{'='*80}")
        print(f"Running Generation-based Evaluation...")
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
                    ""
                )
                
                # Get expected output
                expected = str(example["output"]).strip()
                
                # Tokenize the expected output to determine how many tokens to generate
                expected_tokens = self.tokenizer.encode(expected, add_special_tokens=False)
                num_tokens_to_generate = len(expected_tokens)
                
                # Generate with some buffer (at least 1, max 20 for safety)
                max_new_tokens = min(20, max(1, num_tokens_to_generate + 2))
                
                # Tokenize the prompt
                inputs = self.tokenizer([prompt], return_tensors="pt").to(model.device)
                
                # Generate with temperature=0 for deterministic outputs
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,  # Disable temperature for greedy
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                
                # Decode only the generated tokens (skip the prompt)
                generated_text = self.tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
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
                    print(f"  Evaluated {idx + 1}/{len(self.eval_dataset_raw)} examples (accuracy so far: {current_acc:.3f})")
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
        print(f"Generation-based Evaluation:")
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
            step = state.global_step if hasattr(state, 'global_step') else 0
            reports_dir = self.output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"wrong_answers_step_{step:05d}.md"
            
            with open(report_path, 'w') as f:
                f.write("# Evaluation Wrong Answers Report\n\n")
                f.write("## Metadata\n\n")
                f.write(f"- **Training Step**: {step}\n")
                f.write(f"- **Evaluation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
                    input_text = wrong['input']
                    if len(input_text) > 500:
                        f.write(f"  ```\n  {input_text[:500]}...\n  ```\n")
                    else:
                        f.write(f"  ```\n  {input_text}\n  ```\n")
                    f.write("\n")
            
            print(f"Saved wrong answers report to: {report_path}")
        
        # Restore original training mode
        if was_training:
            model.train()

# Create training_runs directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
training_runs_dir = Path("training_runs") / timestamp
training_runs_dir.mkdir(parents=True, exist_ok=True)
print(f"\nTraining run directory: {training_runs_dir}")

# Create the evaluation callback - use the dataset before formatting to access raw fields
eval_dataset_raw = Dataset.from_list(all_eval_data) if all_eval_data else None
if eval_dataset_raw is not None and downsample_eval is not None and len(eval_dataset_raw) > downsample_eval:
    eval_dataset_raw = eval_dataset_raw.select(range(downsample_eval))

eval_callback = GenerationEvalCallback(
    eval_dataset_raw=eval_dataset_raw,
    alpaca_prompt=alpaca_prompt,
    tokenizer=tokenizer,
    output_dir=training_runs_dir
) if eval_dataset_raw is not None else None

# Training configuration (must be specified in config file)
per_device_train_batch_size = get_config_value("training", "per_device_train_batch_size", args_cli.per_device_train_batch_size)
per_device_eval_batch_size = get_config_value("training", "per_device_eval_batch_size", args_cli.per_device_eval_batch_size)
gradient_accumulation_steps = get_config_value("training", "gradient_accumulation_steps", args_cli.gradient_accumulation_steps)
warmup_steps = get_config_value("training", "warmup_steps", args_cli.warmup_steps)
warmup_ratio = get_config_value("training", "warmup_ratio", args_cli.warmup_ratio)
epochs = get_config_value("training", "epochs", args_cli.epochs)  # Optional - defaults to 1 if None
max_steps = get_config_value("training", "max_steps", args_cli.max_steps)  # Optional - will auto-calculate if None
learning_rate = get_config_value("training", "learning_rate", args_cli.learning_rate)
optim = get_config_value("training", "optim", args_cli.optim)
weight_decay = get_config_value("training", "weight_decay", args_cli.weight_decay)
lr_scheduler_type = get_config_value("training", "lr_scheduler_type", args_cli.lr_scheduler_type)
seed = get_config_value("training", "seed", args_cli.seed)
logging_steps = get_config_value("training", "logging_steps", args_cli.logging_steps)
evals_per_epoch = get_config_value("training", "evals_per_epoch", args_cli.evals_per_epoch)
eval_steps = get_config_value("training", "eval_steps", args_cli.eval_steps)

# Validate required training config
if per_device_train_batch_size is None:
    raise ValueError("training.per_device_train_batch_size must be specified in config file or via --per_device_train_batch_size")
if per_device_eval_batch_size is None:
    raise ValueError("training.per_device_eval_batch_size must be specified in config file or via --per_device_eval_batch_size")
if gradient_accumulation_steps is None:
    raise ValueError("training.gradient_accumulation_steps must be specified in config file or via --gradient_accumulation_steps")
# Require either warmup_steps or warmup_ratio (warmup_ratio is preferred)
if warmup_steps is None and warmup_ratio is None:
    raise ValueError("Either training.warmup_steps or training.warmup_ratio must be specified in config file or via CLI args")
if warmup_steps is not None and warmup_ratio is not None:
    print("\nWarning: Both warmup_steps and warmup_ratio specified - using warmup_ratio")
    warmup_steps = None
if epochs is None:
    raise ValueError("training.epochs must be specified in config file or via --epochs")
# max_steps is optional - will be calculated from epochs if not provided
if learning_rate is None:
    raise ValueError("training.learning_rate must be specified in config file or via --learning_rate")
if optim is None:
    raise ValueError("training.optim must be specified in config file or via --optim")
if weight_decay is None:
    raise ValueError("training.weight_decay must be specified in config file or via --weight_decay")
if lr_scheduler_type is None:
    raise ValueError("training.lr_scheduler_type must be specified in config file or via --lr_scheduler_type")
if seed is None:
    raise ValueError("training.seed must be specified in config file or via --seed")
if logging_steps is None:
    raise ValueError("training.logging_steps must be specified in config file or via --logging_steps")
if evals_per_epoch is None:
    raise ValueError("training.evals_per_epoch must be specified in config file or via --evals_per_epoch")
# eval_steps is optional - will be auto-calculated from evals_per_epoch if not provided

# Set LoRA learning rate to 10x base learning rate if not specified
if lora_lr is None:
    lora_lr = learning_rate * 10
    print(f"Setting LoRA learning rate to 10x base learning rate: {lora_lr}")

# Calculate steps per epoch (needed for wandb config and eval_steps calculation)
effective_batch_size = min(per_device_train_batch_size, len(train_dataset))
if effective_batch_size < per_device_train_batch_size:
    print(f"\nWarning: Reducing batch size from {per_device_train_batch_size} to {effective_batch_size} to match dataset size ({len(train_dataset)} examples)")

total_batch_size = effective_batch_size * gradient_accumulation_steps
steps_per_epoch = len(train_dataset) // total_batch_size
if len(train_dataset) % total_batch_size != 0:
    steps_per_epoch += 1  # Add one step for remainder

# Calculate max_steps if not specified (train for specified number of epochs)
if max_steps is None:
    max_steps = steps_per_epoch * epochs
    print(f"\n{'='*80}")
    print(f"max_steps not specified - auto-calculated for {epochs} epoch(s):")
    print(f"  Dataset size: {len(train_dataset)}")
    print(f"  Batch size: {effective_batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {total_batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps ({epochs} epochs): {max_steps}")
    print(f"{'='*80}\n")

# Calculate eval_steps if not specified (evaluate N times per epoch based on evals_per_epoch)
if eval_steps is None and eval_dataset:
    eval_steps = max(1, steps_per_epoch // evals_per_epoch)
    print(f"eval_steps not specified - auto-calculated for {evals_per_epoch} evals per epoch: {eval_steps}")
    print(f"  (steps_per_epoch: {steps_per_epoch}, eval_steps: {eval_steps})")
    print()

# Wandb configuration (must be specified in config file)
wandb_project = get_config_value("wandb", "project", args_cli.wandb_project)
wandb_enabled = get_config_value("wandb", "enabled", args_cli.wandb_enabled)

# Validate required wandb config
if wandb_project is None:
    raise ValueError("wandb.project must be specified in config file or via --wandb_project")
if wandb_enabled is None:
    raise ValueError("wandb.enabled must be specified in config file or via --wandb_enabled")

# Initialize wandb after datasets are created
if wandb_enabled:
    wandb_config = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": epochs,
        "max_steps": max_steps,
        "steps_per_epoch": steps_per_epoch,
        "evals_per_epoch": evals_per_epoch,
        "eval_steps": eval_steps,
        "learning_rate": learning_rate,
        "optim": optim,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "seed": seed,
        "train_dataset_size": len(train_dataset),
        "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
        "downsample_train": downsample_train,
        "downsample_eval": downsample_eval,
        "num_features": len(datasets_dict),
    }
    # Add warmup parameter to config
    if warmup_ratio is not None:
        wandb_config["warmup_ratio"] = warmup_ratio
        wandb_config["warmup_steps_calculated"] = int(warmup_ratio * max_steps)
    else:
        wandb_config["warmup_steps"] = warmup_steps
    
    wandb.init(
        project=wandb_project,
        name=f"sft-7b-{timestamp}",
        config=wandb_config,
        dir=str(training_runs_dir),
    )

# Training
print(f"\n" + "="*80)
print("Starting training...")
print(f"  Train dataset size: {len(train_dataset)}" + 
      (f" (downsampled from {len(all_train_data)})" if downsample_train is not None and len(train_dataset) < len(all_train_data) else ""))
if eval_dataset:
    print(f"  Eval dataset size: {len(eval_dataset)}" + 
          (f" (downsampled from {len(all_eval_data)})" if downsample_eval is not None and len(eval_dataset) < len(all_eval_data) else ""))
print("="*80)

# Tokenize dataset - use dynamic padding for memory efficiency
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,  # Dynamic padding via data collator
    )

print("\nTokenizing datasets...")
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
)
if eval_dataset:
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
    )

# Data collator with dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
)

# Prepare training arguments with memory optimizations
training_args_dict = {
    "output_dir": str(training_runs_dir),
    "per_device_train_batch_size": effective_batch_size,
    "per_device_eval_batch_size": per_device_eval_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "max_steps": max_steps,
    "learning_rate": learning_rate,
    "fp16": False,
    "bf16": True,
    "logging_steps": logging_steps,
    "optim": optim,
    "weight_decay": weight_decay,
    "lr_scheduler_type": lr_scheduler_type,
    "seed": seed,
    "report_to": "wandb" if wandb_enabled else "none",
    "eval_strategy": "steps" if eval_dataset else "no",
    "eval_steps": eval_steps if eval_dataset else None,
    "dataloader_drop_last": False,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},  # More memory efficient
    "save_steps": 100,
    "save_total_limit": 2,
    "load_best_model_at_end": False,
    # Memory optimization flags
    "dataloader_pin_memory": True,  # Pin memory for faster data transfer
    "dataloader_num_workers": 2,  # Use multiple workers for data loading
    "max_grad_norm": 1.0,  # Gradient clipping to prevent instability
    # Training optimizations
    "tf32": True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False,  # Use TF32 on Ampere+ GPUs
}

# Add warmup parameter based on which one is specified (prefer warmup_ratio)
if warmup_ratio is not None:
    training_args_dict["warmup_ratio"] = warmup_ratio
    print(f"Using warmup_ratio: {warmup_ratio} ({int(warmup_ratio * max_steps)} steps out of {max_steps} total)")
else:
    training_args_dict["warmup_steps"] = warmup_steps
    print(f"Using warmup_steps: {warmup_steps} out of {max_steps} total steps")

training_args = TrainingArguments(**training_args_dict)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[eval_callback] if eval_callback is not None else None,
)

# ============================================================================
# PRE-TRAINING EVALUATION (comment out this block to skip)
# ============================================================================
if eval_callback is not None:
    print("\n" + "="*80)
    print("Running PRE-TRAINING evaluation...")
    print("="*80)
    # Create a dummy state object for the callback
    dummy_state = TrainerState()
    dummy_state.global_step = 0
    # Run the evaluation callback manually
    eval_callback.on_evaluate(
        args=training_args,
        state=dummy_state,
        control=None,
        model=model,
    )
    print("="*80)
    print("Pre-training evaluation complete!")
    print("="*80 + "\n")
# ============================================================================
# END PRE-TRAINING EVALUATION
# ============================================================================

# Train
trainer_stats = trainer.train()

# Inference - test with a sample from our dataset
print("\n" + "="*80)
print("Testing inference...")
print("="*80)

# Use a sample from the eval dataset if available
if all_eval_data:
    sample = all_eval_data[0]
    test_instruction = sample["instruction"]
    test_input = sample["input"][:500]  # Limit input length for testing
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

# Save model
print("\nSaving model...")
trainer.save_model(str(training_runs_dir / "lora_model"))
tokenizer.save_pretrained(str(training_runs_dir / "lora_model"))
print("Training complete!")

