import torch
from openai import OpenAI
from tqdm import tqdm
import re
import argparse
from vllm import LLM, SamplingParams
import os


SYSTEM_PROMPT = "You are a medical assistant."
# BATCH_SIZE = 64
# MODEL_ID = "RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a16"
# MODEL_ID = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
# MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
# assert MODEL_ID is not None, "MODEL_ID is not set"
def retry_with_validation(model, history, validation_func, max_retries=20):
    """
    Generic retry logic with custom validation function.
    
    Args:
        model: Model instance to use for prediction
        history: The formatted history to pass to the prediction function
        validation_func: Function that takes prediction string and returns (is_valid, parsed_result)
                        where is_valid is a boolean and parsed_result is the parsed value or None
        max_retries: Maximum number of retry attempts (default: 10)
        
    Returns:
        tuple: (parsed_result, prediction_string) or (None, prediction_string) if validation failed
        
    Raises:
        ValueError: If no valid result is found after max_retries
    """
    for attempt in range(max_retries):
        pred = model.predict_single(history, max_tokens=10, sample=True)
        
        is_valid, parsed_result = validation_func(pred)
        if is_valid:
            return parsed_result, pred
            
        if attempt == max_retries - 1:  # Last attempt
            # raise ValueError(f"Failed to get a valid result after {max_retries} attempts")
            print(f"Warning: Failed to get a valid result after {max_retries} attempts")
            return None, "MAX_RETRIES_ERROR"
        # Continue to next attempt


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

    @classmethod
    def format_chunk_qs(cls, q: str, chunk: str, options: list[str]):
        prompt= f"Medical Record Excerpt: {chunk}\n\nQuestion: {q}\n\nDO NOT think out loud. Answer only with one of: {options}. Answer: "
        return [{"role": "user", "content": prompt}]


    def predict_single_with_logit_trick(self, history: list[dict], output_choices: set[str]):
        assert isinstance(history, list)
        assert len(history) == 1
        assert "role" in history[0] and "content" in history[0]
        
        # Get model response with logprobs
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=history,
            logprobs=True,
            top_logprobs=20,
            max_tokens=1, 
            # max_tokens=100, # TESTING
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
    
    def predict_single(self, history: list[dict], max_tokens: int, sample=False):
        # TODO: test on dates
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=history,
            max_tokens=max_tokens,
            temperature=1.0 if sample else 0.0,
            top_p=1.0
        )
        return response.choices[0].message.content

class DummyModel:
    def __init__(self):
        pass

    def predict(self, examples):
        return [True] * len(examples)
    
    def predict_single(self, history):
        return True