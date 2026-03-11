from typing import Union, Optional
import torch
from openai import OpenAI
from tqdm import tqdm
import re
import argparse
from vllm import LLM, SamplingParams
import os
import yaml
from Levenshtein import distance
from utils import OptionType

SYSTEM_PROMPT = "You are a medical assistant."

# Load MODEL_ID from config.yml
def load_model_id():
    try:
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)
            return config['model']['id']
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"Error loading config.yml: {e}")
        # Fallback to default model
        return "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

MODEL_ID = load_model_id()
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
        pred = model.predict_single(history, max_tokens=10)
        
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
        model_id,
        base_url="http://localhost:8001/v1", 
        api_key="token-abc123",
    ):
        self.model_id = model_id
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    @classmethod
    def format_chunk_qs(cls, q: str, chunk: str, options: list[str]):
        # prompt= f"Medical Record Excerpt: {chunk}\n\nQuestion: {q}\n\nDO NOT think out loud. Answer only with one of: {options}. Answer: "
        prompt= f"### Medical Record Excerpt: {chunk}\n\n### Question: {q}\n\n"
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
        # if len(top_logprobs) != len(output_choices):
            # print(f"Warning: {len(top_logprobs)} logprobs found for {len(output_choices)} output choices")
        # Return the token with the highest logprob
        if not top_logprobs:
            print("Warning: No valid output choices found in logprobs")
            return "NA"
        return max(top_logprobs, key=lambda x: x.logprob).token
    
    def _strip_think_tags(self, text: str, keep_content: bool = False) -> str:
        """Strip <think>...</think> reasoning blocks from model output.

        If keep_content=True, removes only the tags but keeps reasoning text.
        If keep_content=False, removes both tags and content between them.
        """
        if keep_content:
            return re.sub(r'</?think>', '', text).strip()
        # Remove complete <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove unclosed <think> tags (model hit token limit mid-thinking)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        return text.strip()

    def predict_single(self, history: list[dict], max_tokens: int, options: Optional[Union[list[str], OptionType]], sample=False, attempts=3):
        # TODO: test on dates
        for attempt in range(attempts):
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=history,
                max_tokens=max_tokens,
                temperature=1.0 if sample else 0.0,
                top_p=1.0
            )
            response_text = response.choices[0].message.content
            # Strip <think>...</think> blocks for answer extraction (e.g. DeepSeek-R1, Qwen3)
            # For free-form (options=None), keep content but strip tags
            response_text = self._strip_think_tags(response_text, keep_content=(options is None))
            if options == OptionType.DATE:
                date = self.find_date(response_text)
                if date:
                    return date
                else:
                    pass
            elif options == OptionType.NUMERIC:
                numeric = self.find_numeric(response_text)
                if numeric is not None:
                    return numeric
                else:
                    pass
            elif isinstance(options, list): # choices
                if response_text in options:
                    return response_text
                else:
                    pass # try again
            elif options is None:
                return response_text # free-form text
            else:
                raise NotImplementedError(f"Option type {options} not implemented") 

        return response_text # if we never found a valid date or choice, return the raw response

    def find_date(self, text: str):
        # find the final 8-digit sequence in the text
        if len(text) == 1:
            return text # most likely X or U
        match = re.search(r'\d{8}$', text)
        if match:
            return match.group(0)
        else:
            return None

    def find_numeric(self, text: str):
        """Extract an integer or 'F' from response text."""
        text = text.strip()
        if text.upper() == "F":
            return "F"
        # Look for an integer in the response
        match = re.search(r'\d+', text)
        if match:
            return match.group(0)
        return None

    def predict_with_cot(self, history: list[dict], options: Union[list[str], OptionType], max_answer_tokens: int, sample=False):
        clean_response = None
        attempts = 0
        if isinstance(options, list):
            target_type = OptionType.CHOICES
        elif options == OptionType.DATE:
            target_type = OptionType.DATE
        elif options == OptionType.NUMERIC:
            target_type = OptionType.NUMERIC
        else:
            raise ValueError(f"Invalid options type: {options}")

        target_type_subprompt_map = {
            OptionType.CHOICES: f"### Instructions:\n\nBased on the thoughts, answer only with one of: {options}. Answer only a single letter of the answer, nothing else.",
            OptionType.DATE: f"### Instructions:\n\nBased on the thoughts, answer only with a date in the format of YYYYMMDD, nothing else.",
            OptionType.NUMERIC: f"### Instructions:\n\nBased on the thoughts, answer only with an integer number of days, or 'F' if the duration cannot be determined. Answer only the number or F, nothing else.",
        }

        max_cot_attempts = 2
        for attempt in range(max_cot_attempts):
            cot_prompt = "### Instructions:\n\nThink step by step, then answer the question."
            original_prompt = history[-1]["content"]
            cot_history = [{"role": "user", "content": original_prompt + cot_prompt}]
            cot_response = self.predict_single(history=cot_history, max_tokens=1024, options=None, sample=sample)
            cot_response = self._strip_think_tags(cot_response, keep_content=True)
            clean_prompt = f"{original_prompt}\n\n### Thoughts: {cot_response}\n\n{target_type_subprompt_map[target_type]}"
            clean_history = [{"role": "user", "content": clean_prompt}]
            # Use higher max_tokens to accommodate models that wrap output in <think> tags
            clean_response = self.predict_single(history=clean_history, max_tokens=max(max_answer_tokens, 200), options=options, sample=sample)
            if clean_response is not None:
                return clean_response

        raise ValueError(
            f"COT failed after {max_cot_attempts} attempts.\n"
            f"  Target type: {target_type}\n"
            f"  Options: {options}\n"
            f"  Original prompt (first 500 chars): {original_prompt[:500]}\n"
            f"  Last COT response (first 500 chars): {cot_response[:500]}\n"
            f"  Last clean response: {clean_response}"
        )

class DummyModel:
    def __init__(self):
        pass

    def predict(self, examples):
        return [True] * len(examples)
    
    def predict_single(self, history):
        return True