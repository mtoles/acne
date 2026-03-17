"""
Prompt optimizer base class and implementations.

Each optimizer takes a current prompt and training results, and returns a new prompt.
The optimization loop in auto_prompt.py handles evaluation and iteration.
"""

import re
import random
from collections import defaultdict
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

    @staticmethod
    def format_example(rec, prompt_text="<INS>", include_prediction=False):
        """Format a single record mirroring MrModel.format_chunk_qs inference format.

        Args:
            rec: dict with chunk, keyword, ground_truth, prediction, correct
            prompt_text: the prompt/instruction to show in the Question slot
            include_prediction: if True, include model prediction and ground truth
        """
        lines = (
            f"### Medical Record Excerpt: {rec['chunk']}\n\n"
            f"### Question: {prompt_text}\n\n"
            f"(keyword = \"{rec['keyword']}\")"
        )
        if include_prediction:
            lines += (
                f"\nModel prediction: {rec['prediction']}"
                f"\nGround truth answer: {rec['ground_truth']}"
            )
        else:
            lines += f"\nGround truth answer:\n{rec['ground_truth']}"
        return lines

    def feedback(self, rule, accepted, accuracy_delta):
        """Called by the outer loop after evaluating a candidate prompt.

        Subclasses can override to track accepted/rejected rules.

        Args:
            rule: str, the rule or prompt that was proposed
            accepted: bool, whether accuracy improved
            accuracy_delta: float, change in accuracy (positive = improvement)
        """
        pass

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
    """OPRO-style optimizer: shows previous instructions with scores and example
    problems to a meta-LLM, and asks it to produce an improved instruction."""

    def __init__(self, model=None, n_examples=20):
        super().__init__(model=model)
        self.n_examples = n_examples
        self.trajectory = []  # list of (prompt, accuracy) tuples

    def step(self, current_prompt, records):
        correct = [r for r in records if r["correct"]]
        accuracy = len(correct) / len(records)

        # Record current prompt and score in trajectory
        self.trajectory.append((current_prompt, accuracy))

        # Sample a random selection of problems (mix of correct and incorrect)
        example_sample = random.sample(records, min(self.n_examples, len(records)))

        # Build trajectory section sorted by score (ascending, best last per OPRO paper)
        sorted_trajectory = sorted(self.trajectory, key=lambda x: x[1])
        trajectory_section = "\n\n".join(
            f"text:\n{prompt}\nscore:\n{round(score * 100)}"
            for prompt, score in sorted_trajectory
        )

        # Build problems section
        problems_section = "\n\n".join(
            f"Problem:\n{self.format_example(r)}" for r in example_sample
        )

        meta_prompt = (
            "Your task is to generate the instruction <INS>. Below are some previous instructions with their scores.\n"
            "The score ranges from 0 to 100.\n\n"
            f"{trajectory_section}\n\n"
            f"Below are some problems.\n\n"
            f"{problems_section}\n\n"
            "Think step by step, then generate an instruction that is different from all the instructions <INS> above, and has a higher score "
            "than all the instructions <INS> above. The instruction should begin with <INS> and end with </INS>. "
            "The instruction should be concise, effective, and generally applicable to all problems above. "
            "Do not edit the options mentioned in the problems. "
            "If you include \{keyword\} in your instruction, it will be replaced with the actual keyword during inference, so it can be used as a placeholder. "

        )

        history = [{"role": "user", "content": meta_prompt}]
        response = self.model.predict_single(history, max_tokens=4000, options=None)

        # Strip <think> tags if present (Qwen CoT)
        response = self.model._strip_think_tags(response)

        # Extract the instruction from <INS>...</INS> tags
        ins_match = re.search(r'<INS>(.*?)</INS>', response, flags=re.DOTALL)
        if ins_match:
            new_prompt = ins_match.group(1).strip()
        else:
            # Fallback: try to use the full response stripped of common wrappers
            print("  OPRO: no <INS>...</INS> tags found in response, using full response")
            new_prompt = response.strip()

        print(f"  OPRO prompt ({len(new_prompt)} chars): {new_prompt[:150]}...")

        return {"prompt": new_prompt, "raw_response": response}

class OursOptimizer(PromptOptimizer):
    """Label-2-Prompt optimizer: uses confusion matrix clustering to infer rules
    from errors and appends them to the prompt.

    Unlike OPRO which replaces the entire prompt each iteration, this optimizer
    grows the prompt additively by inferring one rule per iteration targeting
    a specific confusion pattern (ground_truth -> prediction cell).
    """

    def __init__(self, model=None, n_error_examples=10):
        super().__init__(model=model)
        self.n_error_examples = n_error_examples
        self.rule_history = []  # list of (rule, accepted, accuracy_delta)

    def feedback(self, rule, accepted, accuracy_delta):
        self.rule_history.append((rule, accepted, accuracy_delta))

    def step(self, current_prompt, records):
        errors = [r for r in records if not r["correct"]]
        if not errors:
            return {"prompt": current_prompt, "raw_response": ""}

        # Construct confusion matrix of errors: (ground_truth, prediction) -> [records]
        confusion = defaultdict(list)
        for err in errors:
            confusion[(err["ground_truth"], err["prediction"])].append(err)

        # Choose a cell randomly proportional to its prevalence
        cells = list(confusion.keys())
        weights = [len(confusion[c]) for c in cells]
        chosen_cell = random.choices(cells, weights=weights, k=1)[0]
        cell_errors = confusion[chosen_cell]

        # Sample examples from the chosen confusion cell
        error_sample = random.sample(cell_errors, min(self.n_error_examples, len(cell_errors)))

        errors_section = "\n\n".join(
            f"Example {i+1}:\n{self.format_example(err, prompt_text=current_prompt, include_prediction=True)}"
            for i, err in enumerate(error_sample)
        )

        accuracy = sum(1 for r in records if r["correct"]) / len(records)

        # Build rule history section
        rule_history_section = ""
        if self.rule_history:
            rule_history_section = "\n\nPreviously proposed rules:\n" + "\n".join(
                f"- {'ACCEPTED' if accepted else 'REJECTED'} (accuracy {delta:+.0%}): {rule}"
                for rule, accepted, delta in self.rule_history
            ) + "\n"

        meta_prompt = (
            "You are helping optimize a prompt for a medical record question-answering system.\n\n"
            f"The current prompt (accuracy: {accuracy:.0%}) is:\n"
            f"---\n{current_prompt}\n---\n"
            f"{rule_history_section}\n"
            f"Below are {len(error_sample)} examples where the model answered INCORRECTLY. "
            f"All share the same confusion pattern: the model predicts \"{chosen_cell[1]}\" "
            f"when the correct answer is \"{chosen_cell[0]}\" "
            f"({len(cell_errors)} errors of this type out of {len(errors)} total errors).\n\n"
            f"{errors_section}\n\n"
            "Analyze the errors above. Identify the implicit assumption or pattern the model is missing. "
            "Then write a concise rule that, if appended to the prompt, would fix these errors "
            "without breaking correct answers.\n\n"
            "The rule should begin with <RULE> and end with </RULE>.\n"
            "The rule should be a concrete instruction (e.g. 'If overlapping prescriptions, count total days, not per-antibiotic days.').\n"
            "Do not rewrite the entire prompt -- just output the new rule to append.\n"
            "If you include {keyword} in your rule, it will be replaced with the actual keyword during inference."
        )

        history = [{"role": "user", "content": meta_prompt}]
        response = self.model.predict_single(history, max_tokens=2000, options=None)

        # Strip <think> tags if present (Qwen CoT)
        response = self.model._strip_think_tags(response)

        # Extract the rule from <RULE>...</RULE> tags
        rule_match = re.search(r'<RULE>(.*?)</RULE>', response, flags=re.DOTALL)
        if rule_match:
            rule = rule_match.group(1).strip()
        else:
            print("  Ours: no <RULE>...</RULE> tags found in response, using full response")
            rule = response.strip()

        new_prompt = current_prompt.rstrip() + "\n" + rule
        print(f"  Ours confusion cell: {chosen_cell[0]} -> {chosen_cell[1]} ({len(cell_errors)} errors)")
        print(f"  Ours rule ({len(rule)} chars): {rule[:150]}...")

        return {"prompt": new_prompt, "raw_response": response, "rule": rule}