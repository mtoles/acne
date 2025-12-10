import os
import sys
import argparse
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import is_bfloat16_supported
from utils import get_dataset
from pt_features import PtFeaturesMeta
import pandas as pd

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure CUDA-capable GPU is accessible.")
# Parse CLI arguments
parser = argparse.ArgumentParser(description='Train SFT model with optional dataset downsampling')
parser.add_argument('--downsample_train', type=int, default=None, 
                    help='Maximum number of training examples to use (None = use all)')
parser.add_argument('--downsample_eval', type=int, default=None,
                    help='Maximum number of eval examples to use (None = use all)')
args_cli = parser.parse_args()

# Model configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2-7B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
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

# Load datasets for all features
print("\n" + "="*80)
print("Loading datasets...")
print("="*80)
datasets_dict = get_dataset(
    data_source="mgb",
    feature_names=None,  # Load all features
    train_split=0.5,
    downsample=None,
    random_state=42
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

print(f"\n" + "="*80)
print(f"Dataset Summary (before downsampling):")
print(f"  Total train examples: {len(all_train_data)} (from {len(datasets_dict)} features)")
print(f"  Total eval examples: {len(all_eval_data)} (from {len(datasets_dict)} features)")
print(f"  Training ONE model on ALL features combined")
if args_cli.downsample_train is not None or args_cli.downsample_eval is not None:
    print(f"\nDownsampling settings:")
    if args_cli.downsample_train is not None:
        print(f"  Train: {args_cli.downsample_train} examples")
    if args_cli.downsample_eval is not None:
        print(f"  Eval: {args_cli.downsample_eval} examples")
print("="*80)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_list(all_train_data)
eval_dataset = Dataset.from_list(all_eval_data) if all_eval_data else None

# Apply downsampling if specified
if args_cli.downsample_train is not None and len(train_dataset) > args_cli.downsample_train:
    print(f"\nDownsampling train dataset from {len(train_dataset)} to {args_cli.downsample_train} examples...")
    train_dataset = train_dataset.select(range(args_cli.downsample_train))

if eval_dataset is not None and args_cli.downsample_eval is not None and len(eval_dataset) > args_cli.downsample_eval:
    print(f"\nDownsampling eval dataset from {len(eval_dataset)} to {args_cli.downsample_eval} examples...")
    eval_dataset = eval_dataset.select(range(args_cli.downsample_eval))

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

# Training
print(f"\n" + "="*80)
print("Starting training...")
print(f"  Train dataset size: {len(train_dataset)}" + 
      (f" (downsampled from {len(all_train_data)})" if args_cli.downsample_train is not None and len(train_dataset) < len(all_train_data) else ""))
if eval_dataset:
    print(f"  Eval dataset size: {len(eval_dataset)}" + 
          (f" (downsampled from {len(all_eval_data)})" if args_cli.downsample_eval is not None and len(eval_dataset) < len(all_eval_data) else ""))
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        output_dir="outputs",
        report_to="none",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=10 if eval_dataset else None,
    ),
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
