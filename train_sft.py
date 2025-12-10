import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer, TrainerCallback
from unsloth import is_bfloat16_supported
from utils import get_dataset
from pt_features import PtFeaturesMeta
import pandas as pd
import wandb
import yaml

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure CUDA-capable GPU is accessible.")

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
parser.add_argument('--lora_bias', type=str, default=None, help='LoRA bias')
parser.add_argument('--use_gradient_checkpointing', type=str, default=None, help='Gradient checkpointing')
parser.add_argument('--random_state', type=int, default=None, help='Random state')
parser.add_argument('--use_rslora', type=lambda x: (str(x).lower() == 'true'), default=None, help='Use RSLoRA')
# Training args
parser.add_argument('--per_device_train_batch_size', type=int, default=None, help='Batch size per device')
parser.add_argument('--gradient_accumulation_steps', type=int, default=None, help='Gradient accumulation steps')
parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup steps')
parser.add_argument('--max_steps', type=int, default=None, help='Maximum training steps')
parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
parser.add_argument('--optim', type=str, default=None, help='Optimizer')
parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
parser.add_argument('--lr_scheduler_type', type=str, default=None, help='LR scheduler type')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--logging_steps', type=int, default=None, help='Logging steps')
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

# Model configuration
model_name = get_config_value("model", "model_name", args_cli.model_name)
max_seq_length = get_config_value("model", "max_seq_length", args_cli.max_seq_length, 20000)
dtype = get_config_value("model", "dtype", args_cli.dtype)
load_in_4bit = get_config_value("model", "load_in_4bit", args_cli.load_in_4bit, True)

# LoRA configuration
lora_r = get_config_value("lora", "r", args_cli.lora_r, 16)
lora_alpha = get_config_value("lora", "lora_alpha", args_cli.lora_alpha, 16)
lora_dropout = get_config_value("lora", "lora_dropout", args_cli.lora_dropout, 0.0)
lora_bias = get_config_value("lora", "bias", args_cli.lora_bias, "none")
use_gradient_checkpointing = get_config_value("lora", "use_gradient_checkpointing", args_cli.use_gradient_checkpointing, "unsloth")
random_state = get_config_value("lora", "random_state", args_cli.random_state, 3407)
use_rslora = get_config_value("lora", "use_rslora", args_cli.use_rslora, False)
loftq_config = get_config_value("lora", "loftq_config", None, None)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_r,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias=lora_bias,
    use_gradient_checkpointing=use_gradient_checkpointing,
    random_state=random_state,
    use_rslora=use_rslora,
    loftq_config=loftq_config,
)

# Data preparation
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

# Dataset configuration
data_source = get_config_value("dataset", "data_source", args_cli.data_source, "mgb")
feature_names = get_config_value("dataset", "feature_names", None, None)
train_split = get_config_value("dataset", "train_split", args_cli.train_split, 0.5)
downsample = get_config_value("dataset", "downsample", None, None)
random_state_dataset = get_config_value("dataset", "random_state", args_cli.random_state_dataset, 42)

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
            chunk = row.get("chunk", "")
            found_keywords = row.get("found_keywords", "")
            val_unified = row.get("val_unified", "")
            
            # Handle found_keywords - could be string or list
            if isinstance(found_keywords, list):
                keyword = found_keywords[0] if found_keywords else ""
            else:
                keyword = str(found_keywords) if pd.notna(found_keywords) else ""
            
            # Generate instruction using feature's query method
            try:
                instruction = target_cls.query(chunk=chunk, keyword=keyword)
            except Exception as e:
                print(f"    Warning: Error generating query for row {idx}: {e}")
                continue
            
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
            chunk = row.get("chunk", "")
            found_keywords = row.get("found_keywords", "")
            val_unified = row.get("val_unified", "")
            
            # Handle found_keywords - could be string or list
            if isinstance(found_keywords, list):
                keyword = found_keywords[0] if found_keywords else ""
            else:
                keyword = str(found_keywords) if pd.notna(found_keywords) else ""
            
            # Generate instruction using feature's query method
            try:
                instruction = target_cls.query(chunk=chunk, keyword=keyword)
            except Exception as e:
                print(f"    Warning: Error generating query for row {idx}: {e}")
                continue
            
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
        
        # Set model to eval mode and enable inference
        model.eval()
        FastLanguageModel.for_inference(model)
        
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
                
                # Get expected output and determine how many tokens it contains
                expected = str(example["output"]).strip()
                
                # Tokenize the expected output to determine how many tokens to generate
                expected_tokens = self.tokenizer.encode(expected, add_special_tokens=False)
                num_tokens_to_generate = len(expected_tokens)
                
                # Generate exactly the number of tokens in the expected output (at least 1)
                max_new_tokens = max(1, num_tokens_to_generate)
                
                # Tokenize the prompt
                inputs = self.tokenizer([prompt], return_tensors="pt").to(model.device)
                
                # Generate the appropriate number of tokens
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Use greedy decoding for deterministic results
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
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
                        "feature": example.get("feature", "unknown"),
                        "instruction": example["instruction"],
                        "input": example["input"],
                        "ground_truth": expected,
                        "generated": generated_text,
                    })
                total += 1
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0.0
        
        # Log the metric
        print(f"\n{'='*80}")
        print(f"Generation-based Evaluation:")
        print(f"  Exact Match Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Wrong Answers: {len(wrong_answers)}")
        print(f"{'='*80}\n")
        
        # Log to wandb (let wandb auto-assign step)
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
            eval_step = state.epoch if hasattr(state, 'epoch') else 0
            reports_dir = self.output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"wrong_answers_step_{step:05d}.md"
            
            with open(report_path, 'w') as f:
                # Write metadata
                f.write("# Evaluation Wrong Answers Report\n\n")
                f.write("## Metadata\n\n")
                f.write(f"- **Training Step**: {step}\n")
                f.write(f"- **Epoch**: {eval_step}\n")
                f.write(f"- **Evaluation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **Total Examples**: {total}\n")
                f.write(f"- **Correct**: {correct}\n")
                f.write(f"- **Wrong**: {len(wrong_answers)}\n")
                f.write(f"- **Accuracy**: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
                f.write("---\n\n")
                
                # Write wrong answers
                f.write("## Wrong Answers\n\n")
                for i, wrong in enumerate(wrong_answers, 1):
                    f.write(f"### Wrong Answer {i}\n\n")
                    f.write(f"- **Index**: {wrong['index']}\n")
                    f.write(f"- **Feature**: {wrong['feature']}\n")
                    f.write(f"- **Ground Truth**: `{wrong['ground_truth']}`\n")
                    f.write(f"- **Generated**: `{wrong['generated']}`\n")
                    f.write(f"- **Instruction**: {wrong['instruction']}\n")
                    f.write(f"- **Input**:\n")
                    # Truncate long inputs for readability
                    input_text = wrong['input']
                    if len(input_text) > 500:
                        f.write(f"  ```\n  {input_text[:500]}...\n  ```\n")
                    else:
                        f.write(f"  ```\n  {input_text}\n  ```\n")
                    f.write("\n")
            
            print(f"Saved wrong answers report to: {report_path}")
        
        # Set model back to train mode
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

# Training configuration
per_device_train_batch_size = get_config_value("training", "per_device_train_batch_size", args_cli.per_device_train_batch_size, 2)
gradient_accumulation_steps = get_config_value("training", "gradient_accumulation_steps", args_cli.gradient_accumulation_steps, 4)
warmup_steps = get_config_value("training", "warmup_steps", args_cli.warmup_steps, 5)
max_steps = get_config_value("training", "max_steps", args_cli.max_steps, 60)
learning_rate = get_config_value("training", "learning_rate", args_cli.learning_rate, 2e-4)
optim = get_config_value("training", "optim", args_cli.optim, "adamw_8bit")
weight_decay = get_config_value("training", "weight_decay", args_cli.weight_decay, 0.01)
lr_scheduler_type = get_config_value("training", "lr_scheduler_type", args_cli.lr_scheduler_type, "cosine")
seed = get_config_value("training", "seed", args_cli.seed, 0)
logging_steps = get_config_value("training", "logging_steps", args_cli.logging_steps, 1)
eval_steps = get_config_value("training", "eval_steps", args_cli.eval_steps, 10)

# Wandb configuration
wandb_project = get_config_value("wandb", "project", args_cli.wandb_project, "acne-sft-training")
wandb_enabled = get_config_value("wandb", "enabled", args_cli.wandb_enabled, True)

# Initialize wandb after datasets are created
if wandb_enabled:
    wandb.init(
        project=wandb_project,
        name=f"sft-{timestamp}",
        config={
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
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
        },
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

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        output_dir=str(training_runs_dir),
        report_to="wandb" if wandb_enabled else "none",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
    ),
    callbacks=[eval_callback] if eval_callback is not None else None,
)

trainer_stats = trainer.train()

# Inference - test with a sample from our dataset
print("\n" + "="*80)
print("Testing inference...")
print("="*80)
FastLanguageModel.for_inference(model)

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
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
