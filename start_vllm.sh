# CUDA_VISIBLE_DEVICES=3 timeout 12h vllm serve RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a16  --tensor-parallel-size 1 --max-model-len 1024

CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 12h vllm serve RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16 --tensor-parallel-size 4 --max-model-len 1024

CUDA_VISIBLE_DEVICES=2,3 timeout 12h vllm serve unsloth/Llama-3.3-70B-Instruct-bnb-4bit --pipeline_parallel_size 2 --max-model-len 1024 
CUDA_VISIBLE_DEVICES=2,3 timeout 12h vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor_parallel_size 2 --max-model-len 1024 --quantization awq
