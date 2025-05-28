import torch
from openai import OpenAI
from tqdm import tqdm
import re
import argparse
from joblib import Memory
from vllm import LLM, SamplingParams


# Set up cache directory
memory = Memory(location="hf_cache", verbose=0)

SYSTEM_PROMPT = "You are a medical assistant."
BATCH_SIZE = 64

class MrModel:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    ):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        
        # Initialize YES/NO token IDs by making a test call
        test_response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "give me a random number between 0 and 9, inclusive"}],
            logprobs=True,
            top_logprobs=10,
            top_p=1.0,
            temperature=1.0,
        )
        print(f"test response: {test_response}")
        


    def format_chunk_qs(self, q: str, chunk: list[str]):
        prompts = [f"Medical Record Excerpt: {mr}\n\nQuestion: {q}\n\n Answer only YES or NO:" for mr in chunk]
        histories = [[{"role": "user", "content": prompt}] for prompt in prompts]
        return histories


    def predict_single(self, content: str):
        # Convert history to chat format
        messages = [{"role": "user", "content": content}]
        
        # Get model response with logprobs
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            logprobs=True,
            top_logprobs=1,
            max_tokens=1
        )
        
        # Get the generated token ID
        generated_token = response.choices[0].logprobs.token_ids[0]
        
        # Compare with YES token ID
        return generated_token == self.yes_token_id

class DummyModel:
    def __init__(self):
        pass

    def predict(self, examples):
        return [True] * len(examples)
    
    def predict_single(self, history):
        return True