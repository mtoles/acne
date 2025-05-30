

# vllm serve Qwen/Qwen2.5-3B-Instruct --dtype auto --api-key token-abc123
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-4-Scout-17B-16E --dtype auto --api-key token-abc123 --tensor-parallel-size 4

# MODEL_ID=$(grep -v '^#' .env | grep MODEL_ID | cut -d '=' -f2)
CUDA_VISIBLE_DEVICES=3 timeout 12h vllm serve RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a16  --tensor-parallel-size 1 --max-model-len 1024
# CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 12h vllm serve RedHatAI/Qwen3-4B-quantized.w4a16 --tensor-parallel-size 1 --max-model-len 768



# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-4-Scout-17B-16E --tensor-parallel-size 4 --max-model-len 512