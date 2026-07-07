#!/bin/bash
# Serve Qwen3-30B-A3B-Instruct-2507 (MoE: 30B total / 3B active, instruction-tuned)
# via the MAIN .venv's vLLM 0.15.1 -- which runs on this box's driver 555 / CUDA 12.5.
#
# Original target was Qwen3.5-27B, but its qwen3_5 architecture needs vLLM >=0.21,
# all of which are built for CUDA 13 and require an NVIDIA driver >=580. This box
# has driver 555 (CUDA 12.5 max), so Qwen3.5 is blocked until the driver is upgraded.
# To run it after a driver upgrade: create a fresh venv, `pip install vllm==0.24.0`
# (keep its default torch 2.11.0+cu130 -- do NOT downgrade to cu128), then
# `vllm serve Qwen/Qwen3.5-27B`. This Qwen3 MoE is the "similar instruction-tuned
# model" that works on the current driver today.
#
# GPUs 1,2 are free (the 72B occupies the rest). Serves on port 9090.
VENV=/local/data/mt/acne/.venv

# --host 0.0.0.0 binds all interfaces so other hosts (coffee/tea/communication on
# the same subnet) can reach it. Without it, some vLLM builds bind 127.0.0.1 and the
# server is only reachable locally -> remote curls get "connection refused".
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=1,2 "$VENV/bin/vllm" serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel 2 --gpu-memory-utilization 0.9 \
    --enforce-eager --disable-custom-all-reduce --host 0.0.0.0 --port 9090
