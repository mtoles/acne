#!/usr/bin/env python3
"""
Train Qwen 2.5 72B Instruct using QLoRA on 2 GPUs
Usage: CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_sft.py

Successfully tested on 2×46GB L40S GPUs with peak memory of 41.09 GB.
Key: Skip prepare_model_for_kbit_training to avoid temporary memory spike.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from datasets import load_dataset
import torch.distributed as dist

# Distributed setup
is_distributed = "LOCAL_RANK" in os.environ
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if is_distributed:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    print(f"[Rank {local_rank}/{world_size}] Initialized")

model_name = "Qwen/Qwen2.5-72B-Instruct"
max_seq_length = 2048  # Full context length

# Production LoRA config - all attention + MLP layers
lora_config = LoraConfig(
    r=16,  # LoRA rank (increased for better expressiveness)
    lora_alpha=32,  # LoRA alpha (2x rank is standard)
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./training_output_72b",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch size = 1*4*2 = 8
    num_train_epochs=1,  # Full epoch (change as needed)
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=False,
    report_to="none",  # Change to "wandb" for experiment tracking
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
)

def main():
    if local_rank == 0:
        print("="*80)
        print("Training Qwen 2.5 72B Instruct with QLoRA on 2 GPUs")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"LoRA rank: {lora_config.r}")
        print(f"LoRA alpha: {lora_config.lora_alpha}")
        print(f"Target modules: {lora_config.target_modules}")
        print(f"Sequence length: {max_seq_length}")
        print(f"Optimizer: {training_args.optim}")
        print(f"Batch size per device: {training_args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size}")
        print(f"Number of GPUs: {world_size}")
        print("="*80)
    
    # Load tokenizer
    if local_rank == 0:
        print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with 4-bit quantization
    if local_rank == 0:
        print("Loading model with 4-bit NF4 quantization...")
        print("(This takes ~2-3 minutes)")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    if local_rank == 0:
        mem_gb = torch.cuda.memory_allocated(local_rank) / 1024**3
        print(f"✓ Model loaded: {mem_gb:.2f} GB")
        print(f"   Using Data Parallelism (DDP) - full model on each GPU")
        print(f"   device_map={{'': {local_rank}}}")
        print("\nPreparing model for training...")
        print("(Skipping prepare_model_for_kbit_training to avoid memory spike)")
    
    # Enable gradient checkpointing (before adding LoRA)
    model.gradient_checkpointing_enable()
    
    # Make input embeddings trainable (required for LoRA)
    # Use robust approach from mLLaVA
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Add LoRA adapters
    if local_rank == 0:
        print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    
    # Critical: Cast dtypes properly (learned from mLLaVA)
    if local_rank == 0:
        print("Casting LoRA layers to bfloat16 and norms to float32...")
    from peft.tuners.lora import LoraLayer
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    
    if local_rank == 0:
        model.print_trainable_parameters()
        mem_gb = torch.cuda.memory_allocated(local_rank) / 1024**3
        print(f"✓ After LoRA: {mem_gb:.2f} GB")
    
    # Load dataset
    if local_rank == 0:
        print("\nLoading dataset...")
    
    # Using alpaca as example - replace with your dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # For full training, remove this select()
    # dataset = dataset.select(range(1000))  # Uncomment to test with subset
    
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    if local_rank == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Format dataset using Qwen chat template
    def format_instruction(example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        return {"text": text}
    
    if local_rank == 0:
        print("Formatting dataset...")
    train_dataset = train_dataset.map(
        format_instruction,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        format_instruction,
        remove_columns=eval_dataset.column_names
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
    
    if local_rank == 0:
        print("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    if local_rank == 0:
        print("\n" + "="*80)
        print("Starting training...")
        mem_gb = torch.cuda.memory_allocated(local_rank) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(local_rank) / 1024**3
        print(f"Memory before training:")
        print(f"  Allocated: {mem_gb:.2f} GB")
        print(f"  Reserved: {reserved_gb:.2f} GB")
        print("="*80)
    
    try:
        trainer.train()
        
        if local_rank == 0:
            print("\n" + "="*80)
            print("✓ Training completed successfully!")
            print("="*80)
            peak_gb = torch.cuda.max_memory_allocated(local_rank) / 1024**3
            print(f"Peak memory usage: {peak_gb:.2f} GB")
            
            print("\nSaving model...")
            trainer.save_model("./lora_model_72b")
            tokenizer.save_pretrained("./lora_model_72b")
            print("✓ Model saved to ./lora_model_72b")
            
    except RuntimeError as e:
        if local_rank == 0:
            print(f"\n{'='*80}")
            print("Training failed with error:")
            print(f"{e}")
            print(f"{'='*80}")
            mem_gb = torch.cuda.memory_allocated(local_rank) / 1024**3
            peak_gb = torch.cuda.max_memory_allocated(local_rank) / 1024**3
            print(f"Memory at failure: {mem_gb:.2f} GB (peak: {peak_gb:.2f} GB)")

if __name__ == "__main__":
    main()

