"""
Prompt optimizer base class and implementations.

Each optimizer takes a current prompt and training results, and returns a new prompt.
The optimization loop in auto_prompt.py handles evaluation and iteration.
"""

import re
import random
from abc import ABC, abstractmethod


class PromptOptimizer(ABC):
    """Base class for prompt optimizers.

    Subclasses implement `step()` which receives the current prompt and
    all training records, and returns a candidate prompt.
    """

    def __init__(self, model=None):
        """
        Args:
            model: model instance for meta-LLM calls
        """
        self.model = model

    @abstractmethod
    def step(self, current_prompt, records):
        """Produce a candidate prompt given the current prompt and training records.

        Args:
            current_prompt: str, the current prompt text
            records: list of dicts, each with keys:
                - "chunk": str, the input medical record text
                - "keyword": str, the keyword used
                - "ground_truth": str, the correct answer
                - "prediction": str, the model's answer
                - "correct": bool

        Returns:
            dict with keys:
                - "prompt": str, the new candidate prompt (parsed/cleaned)
                - "raw_response": str, the full optimizer response before parsing
        """
        ...


class DummyOptimizer(PromptOptimizer):
    """Dummy optimizer that appends 'lets try again' to the prompt. For testing only."""

    def step(self, current_prompt, records):
        new_prompt = current_prompt + "\nlets try again"
        return {"prompt": new_prompt, "raw_response": new_prompt}


class OPROOptimizer(PromptOptimizer):
    """OPRO-style optimizer: shows errors + random correct examples to a meta-LLM
    and asks it to produce an improved prompt."""

    def __init__(self, model=None, n_correct_examples=5, max_errors=20):
        super().__init__(model=model)
        self.n_correct_examples = n_correct_examples
        self.max_errors = max_errors
        self.trajectory = []  # list of (prompt, accuracy) tuples, sorted by score

    def _format_example(self, rec, label):
        """Format a single record for the meta-prompt."""
        chunk_preview = rec["chunk"]
        return (
            f"[{label}]\n"
            f"Keyword: {rec['keyword']}\n"
            f"Record excerpt: {chunk_preview}\n"
            f"Expected answer: {rec['ground_truth']}\n"
            f"Model answer: {rec['prediction']}\n"
        )

    def step(self, current_prompt, records):
        errors = [r for r in records if not r["correct"]]
        correct = [r for r in records if r["correct"]]
        accuracy = len(correct) / len(records)

        # Record current prompt and score in trajectory
        self.trajectory.append((current_prompt, accuracy))

        # Sample correct examples
        correct_sample = random.sample(correct, min(self.n_correct_examples, len(correct))) if correct else []

        # Cap errors to avoid exceeding context
        error_sample = errors[:self.max_errors]

        # Build meta-prompt
        correct_section = "\n\n".join(self._format_example(r, "CORRECT") for r in correct_sample)
        error_section = "\n\n".join(self._format_example(r, "ERROR") for r in error_sample)

        # Build trajectory section sorted by score (ascending, best last per OPRO paper)
        sorted_trajectory = sorted(self.trajectory, key=lambda x: x[1])
        trajectory_section = "\n\n".join(
            f"[Score: {score:.0%}]\n{prompt}"
            for prompt, score in sorted_trajectory
        )

        meta_prompt = (
            "You are an expert prompt engineer. Your task is to improve a prompt used to extract "
            "information from medical records.\n\n"
            f"## Prompt Optimization Trajectory (sorted by score, best last)\n{trajectory_section}\n\n"
            f"## Current Prompt\n{current_prompt}\n\n"
            f"## Results\n"
            f"Accuracy: {len(correct)}/{len(records)} "
            f"({len(correct)/len(records)*100:.0f}%)\n\n"
        )

        if correct_section:
            meta_prompt += f"## Examples the model got CORRECT\n{correct_section}\n\n"

        meta_prompt += (
            f"## Examples the model got WRONG\n{error_section}\n\n"
            "## Instructions\n"
            "First, reason step-by-step inside <reasoning>...</reasoning> tags. Analyze the errors "
            "above and identify what systematic mistake(s) the model is making. Consider what "
            "instructions would fix these errors while preserving correct answers.\n\n"
            "Then, AFTER the closing </reasoning> tag, output ONLY the improved prompt text.\n\n"
            "Rules:\n"
            "- Keep the same question structure and answer options\n"
            "- Add clarifying rules or instructions that address the specific failure patterns\n"
            "- Do not remove existing instructions that are working\n"
            "- Be concise -- only add what is necessary\n\n"
            "Remember: <reasoning>your analysis here</reasoning>\n"
            "Then the improved prompt text with no other wrapping or explanation."
        )

        history = [{"role": "user", "content": meta_prompt}]
        response = self.model.predict_single(history, max_tokens=4000, options=None)

        # Strip <think> tags if present (Qwen CoT)
        response = self.model._strip_think_tags(response)

        # Extract and log reasoning, then strip it from the output prompt
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, flags=re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            print(f"  OPRO reasoning ({len(reasoning)} chars): {reasoning[:300]}...")
        else:
            print("  OPRO: no <reasoning> tags found in response")

        # Strip <reasoning>...</reasoning> blocks and any unclosed <reasoning> tags
        new_prompt = re.sub(r'<reasoning>.*?</reasoning>', '', response, flags=re.DOTALL)
        new_prompt = re.sub(r'<reasoning>.*', '', new_prompt, flags=re.DOTALL)
        # Strip common preamble the model may add
        new_prompt = re.sub(r'^#+\s*(Modified|Improved|Updated|New)\s*Prompt\s*\n?', '', new_prompt.strip(), flags=re.IGNORECASE)
        new_prompt = new_prompt.strip()

        print(f"  OPRO prompt ({len(new_prompt)} chars): {new_prompt[:150]}...")

        return {"prompt": new_prompt, "raw_response": response}
