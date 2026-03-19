"""
Prompt optimizer base class and implementations.

Each optimizer takes a current prompt and training results, and returns a new prompt.
The optimization loop in auto_prompt.py handles evaluation and iteration.
"""

import json
import re
import random
from collections import defaultdict
from abc import ABC, abstractmethod

# Qwen2.5-72B max context = 32768 tokens (vLLM default).
# Reserve 2048 tokens for output, use the rest for input.
MAX_CONTEXT_TOKENS = 32768
MAX_OUTPUT_TOKENS = 2048
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - MAX_OUTPUT_TOKENS
# Rough chars-per-token estimate for English medical text (conservative)
CHARS_PER_TOKEN = 3
MAX_INPUT_CHARS = MAX_INPUT_TOKENS * CHARS_PER_TOKEN


def fill_prompt(template, examples, placeholder="{EXAMPLES}"):
    """Build a prompt by filling a template with as many examples as fit the token budget.

    Args:
        template: prompt string containing `placeholder` where examples go
        examples: list of formatted example strings
        placeholder: marker in template to replace with joined examples

    Returns:
        The filled prompt string with as many examples as fit MAX_INPUT_CHARS
    """
    base_len = len(template) - len(placeholder)
    budget = MAX_INPUT_CHARS - base_len
    rng = random.Random(hash(tuple(examples)))
    examples = rng.sample(examples, len(examples))
    kept = []
    total = 0
    for text in examples:
        if total + len(text) > budget:
            break
        kept.append(text)
        total += len(text)
    return template.replace(placeholder, "\n\n".join(kept))


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

        # Build trajectory section sorted by score (ascending, best last per OPRO paper)
        sorted_trajectory = sorted(self.trajectory, key=lambda x: x[1])
        trajectory_section = "\n\n".join(
            f"text:\n{prompt}\nscore:\n{round(score * 100)}"
            for prompt, score in sorted_trajectory
        )

        # Sample problems, fit to token budget
        shuffled = random.sample(records, len(records))
        problem_texts = [f"Problem:\n{self.format_example(r)}" for r in shuffled]

        meta_template = (
            "Your task is to generate the instruction <INS>. Below are some previous instructions with their scores.\n"
            "The score ranges from 0 to 100.\n\n"
            f"{trajectory_section}\n\n"
            "Below are some problems.\n\n"
            "{EXAMPLES}\n\n"
            "Think step by step, then generate an instruction that is different from all the instructions <INS> above, and has a higher score "
            "than all the instructions <INS> above. The instruction should begin with <INS> and end with </INS>. "
            "The instruction should be concise, effective, and generally applicable to all problems above. "
            "Do not edit the options mentioned in the problems. "
            "If you include \{keyword\} in your instruction, it will be replaced with the actual keyword during inference, so it can be used as a placeholder. "
        )

        meta_prompt = fill_prompt(meta_template, problem_texts)
        print(f"  OPRO: fitting {meta_prompt.count('Problem:')}/{len(records)} examples into meta-prompt")

        history = [{"role": "user", "content": meta_prompt}]
        response = self.model.predict_single(history, max_tokens=MAX_OUTPUT_TOKENS, options=None)

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

        accuracy = sum(1 for r in records if r["correct"]) / len(records)

        # Build rule history section
        rule_history_section = ""
        if self.rule_history:
            rule_history_section = "\n\nPreviously proposed rules:\n" + "\n".join(
                f"- {'ACCEPTED' if accepted else 'REJECTED'} (accuracy {delta:+.0%}): {rule}"
                for rule, accepted, delta in self.rule_history
            ) + "\n"

        # Sample examples from the chosen confusion cell, fit to token budget
        shuffled_errors = random.sample(cell_errors, len(cell_errors))
        error_texts = [
            f"Example {i+1}:\n{self.format_example(err, prompt_text=current_prompt, include_prediction=True)}"
            for i, err in enumerate(shuffled_errors)
        ]

        meta_template = (
            "You are helping optimize a prompt for a medical record question-answering system.\n\n"
            f"The current prompt (accuracy: {accuracy:.0%}) is:\n"
            f"---\n{current_prompt}\n---\n"
            f"{rule_history_section}\n"
            f"Below are examples where the model answered INCORRECTLY. "
            f"All share the same confusion pattern: the model predicts \"{chosen_cell[1]}\" "
            f"when the correct answer is \"{chosen_cell[0]}\" "
            f"({len(cell_errors)} errors of this type out of {len(errors)} total errors).\n\n"
            "{EXAMPLES}\n\n"
            "Analyze the errors above. Identify the implicit assumption or pattern the model is missing. "
            "Then write a concise rule that, if appended to the prompt, would fix these errors "
            "without breaking correct answers.\n\n"
            "The rule should begin with <RULE> and end with </RULE>.\n"
            "The rule should be a concrete instruction (e.g. 'If overlapping prescriptions, count total days, not per-antibiotic days.').\n"
            "Do not rewrite the entire prompt -- just output the new rule to append.\n"
            "If you include {keyword} in your rule, it will be replaced with the actual keyword during inference."
        )

        meta_prompt = fill_prompt(meta_template, error_texts)
        print(f"  Ours: fitting {meta_prompt.count('Example ')}/{len(cell_errors)} error examples into meta-prompt")

        history = [{"role": "user", "content": meta_prompt}]
        response = self.model.predict_single(history, max_tokens=MAX_OUTPUT_TOKENS, options=None)

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


class ETGPOOptimizer(PromptOptimizer):
    """ETGPO: Error Taxonomy-Guided Prompt Optimization.

    Instead of using a fixed confusion matrix, this optimizer asks the meta-LLM
    to create an error taxonomy — grouping errors by semantic failure mode rather
    than by (ground_truth, prediction) pairs. It then generates targeted guidance
    for the most frequent error categories.

    Algorithm:
        1. Error Collection: (done by outer loop)
        2. Error Taxonomy Creation: batch errors, ask LLM to categorize
        3. Error Category Selection: filter ≥2 examples, take top G by count
        4. Guidance Generation: generate guidance for selected categories, append to prompt
    """

    def __init__(self, model=None, batch_size=10, max_categories=5):
        super().__init__(model=model)
        self.batch_size = batch_size
        self.max_categories = max_categories
        self.rule_history = []

    def feedback(self, rule, accepted, accuracy_delta):
        self.rule_history.append((rule, accepted, accuracy_delta))

    def _call_llm(self, prompt):
        history = [{"role": "user", "content": prompt}]
        response = self.model.predict_single(history, max_tokens=MAX_OUTPUT_TOKENS, options=None)
        if response is None:
            print("  WARNING: LLM returned empty response (input may exceed context window)")
            return ""
        return self.model._strip_think_tags(response)

    def _format_failure(self, i, err, current_prompt):
        """Format a single failure in the paper's format."""
        return (
            f"## Failure {i+1}\n"
            f"### Problem\n{err['chunk']}\n"
            f"### Prompt\n{current_prompt}\n"
            f"(keyword = \"{err['keyword']}\")\n"
            f"### Correct Answer\n{err['ground_truth']}\n"
            f"### Model's Answer\n{err['prediction']}\n"
            f"---"
        )

    def _create_taxonomy(self, errors, current_prompt):
        """Step 2: Batch errors and ask LLM to identify error categories.

        Uses separate prompts for first batch (create categories) and subsequent
        batches (reuse or extend existing categories), per the ETGPO paper.
        """
        all_categories = {}  # category_name -> {"indices": [], "summary": str, "description": str}

        rng = random.Random(hash(tuple(e['chunk'] for e in errors)))
        rng.shuffle(errors)
        for batch_start in range(0, len(errors), self.batch_size):
            batch = errors[batch_start:batch_start + self.batch_size]
            is_first_batch = batch_start == 0

            failure_texts = [self._format_failure(i, err, current_prompt) for i, err in enumerate(batch)]

            if is_first_batch:
                meta_template = (
                    "You are an expert at analyzing why language models fail on medical record question-answering.\n\n"
                    "{EXAMPLES}\n\n"
                    "## Your Task\n"
                    "Analyze each failure and identify the root cause of each error. Be as descriptive as possible.\n"
                    "For each failure, find:\n"
                    "1. The EARLIEST point in the reasoning where something went wrong\n"
                    "2. What specifically went wrong (calculation error, wrong approach, misunderstanding, etc.)\n"
                    "3. Why this error led to the wrong final answer\n\n"
                    "Create issue categories that capture each type of error. Categories should be general enough "
                    "to potentially apply to other traces, but specific enough to be meaningful.\n\n"
                    "IMPORTANT: Each category must be SELF-CONTAINED and understandable by someone who has NOT seen the original problems.\n\n"
                    "## Output Format\n"
                    "Return a JSON object with:\n"
                    "```json\n"
                    "{\n"
                    '  "categories": [\n'
                    "    {\n"
                    '      "category_name": "Short descriptive name for this type of error",\n'
                    '      "summary": "One sentence describing the core error pattern.",\n'
                    '      "description": "Detailed description of what goes wrong in this category. Be very specific.",\n'
                    '      "example": "A concrete, self-contained example. Format: \'Problem: [simple problem]. Error: [what the model does wrong]. Correct: [what should happen].\'",\n'
                    '      "error_type": "Type of error (e.g., Calculation Error, Wrong Approach, Conceptual Misunderstanding, Missing Step, Logical Fallacy, Factual Error, Incomplete Reasoning, Misreading the Problem)",\n'
                    '      "why_leads_to_wrong_answer": "Explanation of how this error causes wrong answers"\n'
                    "    }\n"
                    "  ],\n"
                    '  "failure_assignments": [\n'
                    "    {\n"
                    '      "failure_id": 1,\n'
                    '      "category_name": "Name of the category this failure belongs to",\n'
                    '      "trace_details": {\n'
                    '        "trace_specific_location": "Where in the reasoning the error occurred",\n'
                    '        "trace_specific_details": "Specific details about what went wrong"\n'
                    "      }\n"
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "```"
                )
            else:
                # Subsequent batches: provide existing categories for reuse (Figure 6)
                existing_cats = ""
                for name, info in all_categories.items():
                    existing_cats += (
                        f"### Category: {name}\n"
                        f"- Summary: {info.get('summary', '')}\n"
                        f"- Description: {info.get('description', '')}\n"
                        f"- Error Type: {info.get('error_type', '')}\n"
                        f"- Why it leads to wrong answer: {info.get('why_leads_to_wrong_answer', '')}\n"
                        f"- Traces with this issue so far: {len(info['indices'])}\n\n"
                    )
                meta_template = (
                    "You are an expert at analyzing why language models fail on medical record question-answering.\n\n"
                    f"## Existing Issue Categories\n{existing_cats}\n"
                    "{EXAMPLES}\n\n"
                    "## Your Task\n"
                    "For each failure:\n"
                    "1. Determine if the error fits one of the EXISTING categories\n"
                    "2. OR create a NEW category if the error is fundamentally different\n\n"
                    "## Output Format\n"
                    "Return a JSON object with:\n"
                    "```json\n"
                    "{\n"
                    '  "new_categories": [\n'
                    "    {\n"
                    '      "category_name": "Short descriptive name",\n'
                    '      "summary": "One sentence describing the core error pattern.",\n'
                    '      "description": "Detailed description.",\n'
                    '      "example": "A concrete, self-contained example.",\n'
                    '      "error_type": "Type of error",\n'
                    '      "why_leads_to_wrong_answer": "Explanation"\n'
                    "    }\n"
                    "  ],\n"
                    '  "failure_assignments": [\n'
                    "    {\n"
                    '      "failure_id": 1,\n'
                    '      "is_new_category": false,\n'
                    '      "category_name": "Name of existing or new category",\n'
                    '      "trace_details": {\n'
                    '        "trace_specific_location": "Where the error occurred",\n'
                    '        "trace_specific_details": "What went wrong"\n'
                    "      }\n"
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "```\n"
                    'Note: "new_categories" should only contain categories that don\'t exist yet.'
                )

            meta_prompt = fill_prompt(meta_template, failure_texts)

            response = self._call_llm(meta_prompt)
            if not response:
                print("  ETGPO: empty response from taxonomy LLM, skipping batch")
                continue

            # Parse JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, flags=re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, flags=re.DOTALL)
            if not json_match:
                print("  ETGPO: no JSON found in taxonomy response, skipping batch")
                continue

            try:
                parsed = json.loads(json_match.group(1) if '```' in (json_match.group(0) or '') else json_match.group(0))
            except json.JSONDecodeError:
                print("  ETGPO: failed to parse taxonomy JSON, skipping batch")
                continue

            # Register new categories
            new_cats = parsed.get("categories", []) + parsed.get("new_categories", [])
            for cat in new_cats:
                name = cat.get("category_name", "").strip()
                if name and name not in all_categories:
                    all_categories[name] = {
                        "indices": [],
                        "summary": cat.get("summary", ""),
                        "description": cat.get("description", ""),
                        "error_type": cat.get("error_type", ""),
                        "why_leads_to_wrong_answer": cat.get("why_leads_to_wrong_answer", ""),
                    }

            # Assign failures to categories
            for assignment in parsed.get("failure_assignments", []):
                failure_id = assignment.get("failure_id", 0)
                cat_name = assignment.get("category_name", "").strip()
                error_idx = failure_id - 1
                global_idx = batch_start + error_idx
                if global_idx < len(errors) and cat_name:
                    if cat_name not in all_categories:
                        all_categories[cat_name] = {"indices": [], "summary": "", "description": "", "error_type": "", "why_leads_to_wrong_answer": ""}
                    all_categories[cat_name]["indices"].append(global_idx)

        return all_categories

    def _select_categories(self, taxonomy):
        """Step 3: Filter to ≥2 examples, sort by count, take top G."""
        filtered = {k: v for k, v in taxonomy.items() if len(v["indices"]) >= 2}
        sorted_cats = sorted(filtered.items(), key=lambda x: len(x[1]["indices"]), reverse=True)
        return sorted_cats[:self.max_categories]

    def _generate_guidance(self, selected_categories, current_prompt):
        """Step 4: Generate guidance for selected error categories (Figures 8-9)."""
        total_errors = sum(len(info["indices"]) for _, info in selected_categories)
        categories_section = ""
        for i, (cat_name, cat_info) in enumerate(selected_categories, 1):
            n_failures = len(cat_info["indices"])
            coverage_pct = round(100 * n_failures / total_errors) if total_errors else 0
            # Count unique problems
            unique_problems = len(set(cat_info["indices"]))
            categories_section += (
                f"## Category {i}: {cat_name}\n"
                f"**Statistics:** {n_failures} failures ({coverage_pct}%), {unique_problems} problems\n"
                f"**Summary:** {cat_info.get('summary', '')}\n"
                f"**Description:** {cat_info.get('description', '')}\n"
                f"**Error Type:** {cat_info.get('error_type', '')}\n"
                f"**Why it leads to wrong answer:** {cat_info.get('why_leads_to_wrong_answer', '')}\n"
                f"---\n"
            )

        rule_history_section = ""
        if self.rule_history:
            rule_history_section = "\n\nPreviously proposed guidance:\n" + "\n".join(
                f"- {'ACCEPTED' if accepted else 'REJECTED'} ({delta:+.0%}): {rule[:100]}..."
                for rule, accepted, delta in self.rule_history
            ) + "\n"

        meta_prompt = (
            "You are an expert at improving language model performance on medical record question-answering.\n\n"
            "I have identified the following error categories from model failures. "
            "Generate guidance to help avoid these errors.\n\n"
            f"{categories_section}\n"
            f"{rule_history_section}\n"
            "## Your Task\n"
            "Generate guidance text that:\n"
            "1. Addresses each failure category with specific, actionable advice\n"
            "2. Is written as instructions TO the model\n"
            "3. Uses concrete examples where helpful\n"
            "4. Is prioritized by frequency\n\n"
            "Generate DETAILED guidance with examples. Each item should include:\n"
            "- Description of the error pattern\n"
            "- Actionable advice on how to avoid it\n"
            "- WRONG example showing the error\n"
            "- CORRECT example showing proper approach\n\n"
            "## Critical Constraints\n"
            '- The goal is ACCURACY, not caution. Never generate guidance that encourages the model to refuse, abstain, or say "not specified" when an answer can be reasonably provided.\n'
            "- CORRECT examples must always show the model providing a substantive answer. Never show abstention/refusal as the correct behavior.\n\n"
            "## Output Format\n"
            "Return a JSON object with:\n"
            "```json\n"
            "{\n"
            '  "guidance_items": [\n'
            "    {\n"
            '      "category_name": "Name of the category",\n'
            '      "guidance_text": "The full guidance text for this category"\n'
            "    }\n"
            "  ],\n"
            '  "preamble": "1-2 sentence introduction",\n'
            '  "full_prompt": "Complete enhanced prompt starting with base instruction"\n'
            "}\n"
            "```\n"
            'The "full_prompt" should start with the original base prompt:\n'
            f'"""\n{current_prompt}\n"""\n'
            "Then add your preamble and guidance items.\n"
            "If you include {keyword} in your output, it will be replaced with the actual keyword during inference."
        )

        response = self._call_llm(meta_prompt)
        if not response:
            print("  ETGPO: empty response from guidance LLM")
            return current_prompt, ""

        # Parse JSON response to extract full_prompt
        json_match = re.search(r'```json\s*(.*?)\s*```', response, flags=re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', response, flags=re.DOTALL)

        new_prompt = None
        if json_match:
            try:
                raw_json = json_match.group(1) if '```' in json_match.group(0) else json_match.group(0)
                parsed = json.loads(raw_json)
                # Prefer full_prompt if present and non-empty
                full_prompt = parsed.get("full_prompt", "").strip()
                if full_prompt and len(full_prompt) > len(current_prompt):
                    new_prompt = full_prompt
                else:
                    # Construct from preamble + guidance_items (more robust if full_prompt truncated)
                    preamble = parsed.get("preamble", "").strip()
                    items = parsed.get("guidance_items", [])
                    guidance_text = "\n".join(item.get("guidance_text", "") for item in items if item.get("guidance_text"))
                    if guidance_text:
                        parts = [current_prompt.rstrip()]
                        if preamble:
                            parts.append(preamble)
                        parts.append(guidance_text)
                        new_prompt = "\n\n".join(parts)
            except json.JSONDecodeError:
                print("  ETGPO: failed to parse guidance JSON")

        if not new_prompt:
            print("  ETGPO: no structured guidance extracted, using raw response appended")
            new_prompt = current_prompt.rstrip() + "\n" + response.strip()

        return new_prompt, response

    def step(self, current_prompt, records):
        errors = [r for r in records if not r["correct"]]
        if not errors:
            return {"prompt": current_prompt, "raw_response": ""}

        accuracy = sum(1 for r in records if r["correct"]) / len(records)
        print(f"  ETGPO: {len(errors)} errors at {accuracy:.0%} accuracy")

        # Step 2: Create error taxonomy
        taxonomy = self._create_taxonomy(errors, current_prompt)
        cat_summary = ", ".join(f"{k} ({len(v['indices'])})" for k, v in taxonomy.items())
        print(f"  ETGPO: {len(taxonomy)} categories identified: {cat_summary}")

        # Step 3: Select top categories
        selected = self._select_categories(taxonomy)
        if not selected:
            print("  ETGPO: no categories with >=2 errors, falling back to all")
            selected = sorted(taxonomy.items(), key=lambda x: len(x[1]["indices"]), reverse=True)[:self.max_categories]

        if not selected:
            return {"prompt": current_prompt, "raw_response": "no taxonomy categories found"}

        sel_summary = ", ".join(f"{k} ({len(v['indices'])})" for k, v in selected)
        print(f"  ETGPO: selected {len(selected)} categories: {sel_summary}")

        # Step 4: Generate guidance (returns full_prompt which includes base prompt + guidance)
        full_prompt, raw_response = self._generate_guidance(selected, current_prompt)

        print(f"  ETGPO prompt ({len(full_prompt)} chars): {full_prompt[:150]}...")

        return {"prompt": full_prompt, "raw_response": raw_response, "rule": full_prompt}