#!/bin/bash

# Check if config.yml exists
if [ ! -f config.yml ]; then
    echo "Error: config.yml file not found"
    exit 1
fi

# Extract MODEL_ID from YAML file
MODEL_ID=$(grep -A 1 "model:" config.yml | grep "id:" | sed 's/.*id: *"\([^"]*\)".*/\1/')

# Check if MODEL_ID was extracted successfully
if [ -z "$MODEL_ID" ]; then
    echo "Error: MODEL_ID not found in config.yml file"
    exit 1
fi

echo "Starting vLLM server with model: $MODEL_ID"

# CUDA_VISIBLE_DEVICES=3 timeout 12h vllm serve RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a16  --tensor-parallel-size 1 --max-model-len 1024

# CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 12h vllm serve RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16 --tensor-parallel-size 4 --max-model-len 1024

# CUDA_VISIBLE_DEVICES=0,1 timeout 12h vllm serve unsloth/Llama-3.3-70B-Instruct-bnb-4bit --pipeline_parallel_size 2 --max-model-len 1024 --gpu-memory-utilization 0.8

# Use MODEL_ID from config.yml file
CUDA_VISIBLE_DEVICES=0,1 timeout 12h vllm serve "$MODEL_ID" --pipeline_parallel_size 2 --max-model-len 1024 --gpu-memory-utilization 0.8
