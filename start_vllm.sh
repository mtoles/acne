#!/bin/bash

# Check if config.yml exists
if [ ! -f config.yml ]; then
    echo "Error: config.yml file not found"
    exit 1
fi

# Extract MODEL_ID from YAML file (ignoring comments)
MODEL_ID=$(grep -A 1 "model:" config.yml | grep -v "^[[:space:]]*#" | grep "id:" | sed 's/.*id: *"\([^"]*\)".*/\1/')

# Check if MODEL_ID was extracted successfully
if [ -z "$MODEL_ID" ]; then
    echo "Error: MODEL_ID not found in config.yml file"
    exit 1
fi

echo "Starting vLLM server with model: $MODEL_ID"

# Use MODEL_ID from config.yml file
CUDA_VISIBLE_DEVICES=2,3 timeout 12h vllm serve "$MODEL_ID" --pipeline-parallel-size 2 --gpu-memory-utilization 0.95 # qwen 72b
# CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 12h vllm serve "$MODEL_ID" --pipeline_parallel_size 4 --max-model-len 1200 --gpu-memory-utilization 0.8 # llama 4 scout 17b

