import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import re
import argparse
from joblib import Memory

# Set up cache directory
memory = Memory(location="hf_cache", verbose=0)

SYSTEM_PROMPT = "You are a medical assistant."
BATCH_SIZE = 8

class MrModel:
    def __init__(
        self,
        model_name="meta-llama/Llama-3.1-8b-instruct",
        device_map="auto",
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.pipe = self._create_pipeline()
        # Initialize YES/NO token IDs
        self.yes_token_id = self.tokenizer.encode("YES")[-1]
        self.no_token_id = self.tokenizer.encode("NO")[-1]
        # Cache the predict_batch method, ignoring 'self' in the hash
        self._cached_predict_batch = memory.cache(ignore=["self"])(self._predict_batch)

    def _create_pipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        pipe = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self.device_map,
            return_tensors=True,
            return_dict_in_generate=True,
        )
        # Set pad token to enable batching
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        pipe.model.generation_config.pad_token_id = pipe.tokenizer.pad_token_id

        return pipe

    def format_chunk_qs(self, q: str, chunk: list[str]):
        prompts = [f"Medical Record Excerpt: {mr}\n\nQuestion: {q}\n\n Answer only YES or NO:" for mr in chunk]
        histories = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}] for prompt in prompts]
        return histories

    def predict(self, examples):
        # Break examples into batches
        batches = []
        for i in range(0, len(examples), BATCH_SIZE):
            batch = examples[i:i + BATCH_SIZE]
            batches.append(batch)

        # Run predict_batch on each batch and combine results
        all_results = []
        for batch in tqdm(batches):
            batch_results = self.predict_batch(batch)
            all_results.extend(batch_results)

        return all_results

    def predict_batch(self, histories):
        return self._cached_predict_batch(histories)

    def _predict_batch(self, histories, **kwargs):
        # Convert histories to input_ids
        chats = []

        for history in histories:
            # Convert history to chat format
            chats.append(self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True))
            # Tokenize
        encoded = self.tokenizer(chats, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(self.pipe.model.device)
        attention_mask = encoded["attention_mask"].to(self.pipe.model.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.pipe.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                max_new_tokens=1
            )
        
        # Get logits for the next token
        next_token_logits = outputs.logits[:, -1, :]
        
        results = []
        for logits in next_token_logits:
            # Get logits for YES and NO
            yes_logit = logits[self.yes_token_id].item()
            no_logit = logits[self.no_token_id].item()
            
            # Return True if YES has higher probability
            results.append(yes_logit > no_logit)
            
        return results
