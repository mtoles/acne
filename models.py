import torch
from openai import OpenAI
from tqdm import tqdm
import re
import argparse
from vllm import LLM, SamplingParams
import os


SYSTEM_PROMPT = "You are a medical assistant."
BATCH_SIZE = 64
# MODEL_ID = "RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a16"
MODEL_ID = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
# assert MODEL_ID is not None, "MODEL_ID is not set"

class MrModel:
    def __init__(
        self,
        model_id=MODEL_ID,
        base_url="http://localhost:8000/v1", 
        api_key="token-abc123",
    ):
        self.model_id = model_id
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        


    def format_chunk_qs(self, q: str, chunk: str, options: list[str]):
        prompt= f"Medical Record Excerpt: {chunk}\n\nQuestion: {q}\n\nDO NOT think out loud. Answer only with one of: {options}"
        return [{"role": "user", "content": prompt}]


    def predict_single(self, history: list[dict], output_choices: set[str]):
        assert isinstance(history, list)
        assert len(history) == 1
        assert "role" in history[0] and "content" in history[0]
        
        # Get model response with logprobs
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=history,
            logprobs=True,
            top_logprobs=20,
            max_tokens=1, # should be 1 for multiple choice, testing
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        print(f"response text: {response.choices[0].message.content}")
        # Get the top generated token ID that is a valid output choice
        # print(response.choices[0].logprobs.content[0].top_logprobs)
        all_top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        top_logprobs = [logprob for logprob in all_top_logprobs if logprob.token in output_choices]
        if len(top_logprobs) != len(output_choices):
            print(f"Warning: {len(top_logprobs)} logprobs found for {len(output_choices)} output choices")
        # Return the token with the highest logprob
        if not top_logprobs:
            print("Warning: No valid output choices found in logprobs")
            return "NA"
        return max(top_logprobs, key=lambda x: x.logprob).token

class DummyModel:
    def __init__(self):
        pass

    def predict(self, examples):
        return [True] * len(examples)
    
    def predict_single(self, history):
        return True