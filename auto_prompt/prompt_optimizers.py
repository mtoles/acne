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


class AMPOOptimizer(PromptOptimizer):
    """AMPO: Automatic Multi-Branched Prompt Optimization (Yang et al., EMNLP 2024).

    Three-agent pipeline per iteration:
        1. LLM-Analyzer: identifies root cause of each sampled failure (Table 4)
        2. LLM-Summarizer: consolidates causes into patterns with importance scores (Table 5)
        3. LLM-Revisor: adds/enhances conditional branches in the prompt, then prunes (Table 6)

    Prompt templates are taken verbatim from the AMPO paper Tables 4-6.
    The prompt grows structurally via if/else branches rather than linear rule
    appending. The target LLM self-routes through branches at inference time.
    """

    # -- Paper prompt templates (Tables 4, 5, 6) --
    # DO NOT edit these templates without permission.

    _ANALYZER_PROMPT = (
        "---ProblemStart---\n"
        "I have some instructions for a specific problem:\n"
        "---InstructionsStart---\n"
        "{initial_prompt}\n"
        "---InstructionsEnd---\n"
        "But it gets the following cases wrong:\n"
        "---BadCasesStart---\n"
        "{bad_examples}\n"
        "---BadCasesEnd---\n"
        "Your task is to identify the underlying causes for my [# Instructions] as an analyzer. "
        "Please follow these steps:\n"
        "(1) Identify what perspectives there are to consider for my problem. Please think as "
        "comprehensively as possible, considering all aspects.\n"
        "(2) Based on these potential perspectives you identified, analyze the pattern of the failed cases.\n"
        "(3) Carefully review each step of my [# Instructions] and identify which step neglects the "
        "key information in the pattern, resulting in failure.\n"
        "(4) Write your reasons and wrap each reason with <START> and <END>."
    )

    _SUMMARIZER_PROMPT = (
        "---ProblemStart---\n"
        "I have some instructions for a specific problem:\n"
        "---InstructionsStart---\n"
        "{initial_prompt}\n"
        "---InstructionsEnd---\n"
        "Here are some reasons why my current instructions cannot solve some problem:\n"
        "---Reasons---\n"
        "{reasons}\n"
        "---Reasons---\n"
        "Your task is to summarize the many reasons provided above into a few major categories "
        "and assign an important score for each category. Be careful to eliminate repetitive and "
        "similar reasons. Each summarized pattern should be wrapped with <START> and <END>."
    )

    _REVISOR_PROMPT = (
        "---ProblemStart---\n"
        "You have some instructions for a specific task:\n"
        "---InstructionsStart---\n"
        "{initial_prompt}\n"
        "---InstructionsEnd---\n"
        "However, due to the complexity of real-world situations, a single flow of instructions "
        "(i.e., sequential instructions) cannot apply to all cases. Therefore, you should transform "
        "the instructions into a conditional approach, which means adopting different instructions "
        "for different patterns.\n"
        "Notably, the key aspect of this process is to create an adaptive prompt structure, thereby "
        "accommodating tasks of varying difficulties. To achieve this, you should find the golden "
        "mean between adding the branches to address the new pattern and providing more details to "
        "enhance the existing branches based on the difficulty of your task and the distribution of "
        "recognized patterns.\n"
        "An expert has pointed some patterns that you don't considered before for your instructions:\n"
        "---ExpertAnalysisStart---\n"
        "{patterns}\n"
        "---ExpertAnalysisEnd---\n"
        "Please optimize your [# Instructions] based on expert analysis step-by-step:\n"
        "(1) Carefully review each step of your instructions.\n"
        "(2) Identify the steps that went wrong due to a lack of key information mentioned in "
        "expert analysis.\n"
        "(3) For each suboptimal step, you have the following options:\n"
        "- 3.1 Consider improving the step to include the key information.\n"
        "- 3.2 Otherwise, you can also consider adding **sub-steps** using an **if** or **if-else** "
        "structure to handle the **new** patterns. Ensure that each substep is specific and avoids "
        "vague instructions.\n"
        "Note that if a step needs to consider multiple situations, break it down into substeps to "
        "make it easier to follow.\n"
        "(4) Include Tips or Cautions: If merely optimizing existing steps with branches like "
        "if-else does not sufficiently to address all aspects, add new tips or cautions to the "
        "current instructions to handle different patterns.\n"
        "(5) Maintain the other main steps unchanged from the initial prompt, in order to not lose "
        "information.\n"
        "(6) At last, review the whole steps and prune the branches to avoid the instructions "
        "overfitting.\n"
        "Please only output the optimized prompt without anything else."
    )

    _PRUNER_PROMPT = (
        "---ProblemStart---\n"
        "You have some optimized instructions for a specific task:\n"
        "---InstructionsStart---\n"
        "{optimized_prompt}\n"
        "---InstructionsEnd---\n"
        "Review the whole instructions and prune any branches that may cause overfitting. "
        "Remove branches that are:\n"
        "- Too specific to a single case and unlikely to generalize\n"
        "- Redundant with other branches\n"
        "- Contradictory to other instructions\n"
        "Please only output the pruned prompt without anything else."
    )

    def __init__(self, model=None, k_failures=5, n_top_patterns=1):
        super().__init__(model=model)
        self.k_failures = k_failures
        self.n_top_patterns = n_top_patterns
        self.revision_history = []  # track (pattern, accepted, delta)

    def feedback(self, rule, accepted, accuracy_delta):
        self.revision_history.append((rule, accepted, accuracy_delta))

    def _call_llm(self, prompt, sample=False):
        history = [{"role": "user", "content": prompt}]
        response = self.model.predict_single(
            history, max_tokens=MAX_OUTPUT_TOKENS, options=None, sample=sample
        )
        if response is None:
            print("  AMPO WARNING: LLM returned empty response")
            return ""
        return self.model._strip_think_tags(response)

    def _format_bad_case(self, i, err, current_prompt):
        """Format a single failure case for the Analyzer prompt."""
        return (
            f"Case {i+1}:\n"
            f"Medical Record: {err['chunk']}\n"
            f"Question: {current_prompt}\n"
            f"(keyword = \"{err['keyword']}\")\n"
            f"Correct Answer: {err['ground_truth']}\n"
            f"Model's Answer: {err['prediction']}"
        )

    def _analyze(self, errors, current_prompt):
        """Agent 1 (LLM-Analyzer, Table 4): identify root cause for each failure.

        Returns list of reason strings extracted from <START>...<END> tags.
        """
        bad_examples = "\n\n".join(
            self._format_bad_case(i, err, current_prompt)
            for i, err in enumerate(errors)
        )

        meta_prompt = self._ANALYZER_PROMPT.format(
            initial_prompt=current_prompt,
            bad_examples=bad_examples,
        )
        print(f"  AMPO Analyzer: analyzing {len(errors)} failures")

        response = self._call_llm(meta_prompt, sample=True)
        if not response:
            return []

        # Extract reasons from <START>...<END> tags
        reasons = re.findall(r'<START>(.*?)<END>', response, flags=re.DOTALL)
        reasons = [r.strip() for r in reasons if r.strip()]

        if not reasons:
            print("  AMPO Analyzer: no <START>...<END> tags found, using full response")
            reasons = [response.strip()]

        return reasons

    def _summarize(self, reasons, current_prompt):
        """Agent 2 (LLM-Summarizer, Table 5): consolidate reasons into patterns.

        Returns list of (pattern_text, importance_score) tuples sorted by score
        descending. Importance scores are parsed from the summarizer output
        (e.g. "Important score: 8/10" or "Score: 8").
        """
        reasons_text = "\n".join(f"Reason {i+1}: {r}" for i, r in enumerate(reasons))

        meta_prompt = self._SUMMARIZER_PROMPT.format(
            initial_prompt=current_prompt,
            reasons=reasons_text,
        )

        response = self._call_llm(meta_prompt)
        if not response:
            return []

        # Extract patterns from <START>...<END> tags
        raw_patterns = re.findall(r'<START>(.*?)<END>', response, flags=re.DOTALL)
        raw_patterns = [p.strip() for p in raw_patterns if p.strip()]

        if not raw_patterns:
            print("  AMPO Summarizer: no <START>...<END> tags found, using full response")
            raw_patterns = [response.strip()]

        # Parse importance scores from each pattern block
        scored_patterns = []
        for pat in raw_patterns:
            # Try to find score like "Important score: 8", "Score: 8/10", "Importance: 8"
            score_match = re.search(
                r'(?:important|importance)\s*(?:score)?\s*[:=]\s*(\d+)', pat, re.IGNORECASE
            )
            if not score_match:
                score_match = re.search(r'score\s*[:=]\s*(\d+)', pat, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 0
            scored_patterns.append((pat, score))

        # Sort by importance score descending (Algorithm line 14)
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        return scored_patterns

    def _revise(self, current_prompt, pattern_text):
        """Agent 3 (LLM-Revisor, Table 6): add/enhance branches.

        Returns (new_prompt, raw_response).
        """
        meta_prompt = self._REVISOR_PROMPT.format(
            initial_prompt=current_prompt,
            patterns=pattern_text,
        )

        response = self._call_llm(meta_prompt)
        if not response:
            return current_prompt, ""

        # The paper says "Please only output the optimized prompt without anything else."
        # So the full response IS the new prompt.
        new_prompt = response.strip()

        return new_prompt, response

    def _prune(self, optimized_prompt):
        """Separate pruning step (Algorithm line 17).

        The Revisor prompt includes inline pruning (step 6), but the algorithm
        specifies a distinct pruning pass to avoid overfitting.

        Returns pruned prompt string.
        """
        meta_prompt = self._PRUNER_PROMPT.format(
            optimized_prompt=optimized_prompt,
        )

        response = self._call_llm(meta_prompt)
        if not response:
            return optimized_prompt

        return response.strip()

    def step(self, current_prompt, records):
        """Single-candidate step for interface compatibility.

        When n_top_patterns=1 and the base class search() loop is used, this
        provides the standard step/eval/accept cycle.
        """
        errors = [r for r in records if not r["correct"]]
        if not errors:
            return {"prompt": current_prompt, "raw_response": ""}

        accuracy = sum(1 for r in records if r["correct"]) / len(records)

        # Line 9: Sample K failed cases
        k = min(self.k_failures, len(errors))
        sampled_errors = random.sample(errors, k)

        print(f"  AMPO: {len(errors)} errors at {accuracy:.0%}, sampling {k} for analysis")

        # Line 11: LLM-Analyzer
        reasons = self._analyze(sampled_errors, current_prompt)
        if not reasons:
            print("  AMPO: analyzer returned no reasons")
            return {"prompt": current_prompt, "raw_response": ""}
        print(f"  AMPO Analyzer: {len(reasons)} root causes identified")

        # Line 12: LLM-Summarizer
        scored_patterns = self._summarize(reasons, current_prompt)
        if not scored_patterns:
            print("  AMPO: summarizer returned no patterns")
            return {"prompt": current_prompt, "raw_response": ""}
        for i, (p, s) in enumerate(scored_patterns):
            print(f"  AMPO Pattern {i+1} (score={s}): {p[:100]}...")

        # Line 14: Select top N patterns
        top_patterns = scored_patterns[:self.n_top_patterns]

        # Line 15: LLM-Revisor (use top-1 for single-candidate step)
        pattern_text = top_patterns[0][0]
        new_prompt, raw_response = self._revise(current_prompt, pattern_text)

        # Line 17: LLM-Revisor prunes
        new_prompt = self._prune(new_prompt)

        print(f"  AMPO prompt ({len(new_prompt)} chars): {new_prompt[:150]}...")

        return {"prompt": new_prompt, "raw_response": raw_response, "rule": pattern_text}

    def search(self, init_prompt, eval_fn, iterations=3):
        """AMPO optimization loop (Algorithm 1).

        When n_top_patterns > 1, overrides the base class to generate N
        candidates per iteration, evaluate each, and select the best
        (Algorithm lines 15-19). When n_top_patterns == 1, this is equivalent
        to the base class greedy loop but with the separate pruning step.

        Args:
            init_prompt: str, the starting prompt
            eval_fn: callable(prompt) -> (accuracy: float, records: list[dict])
            iterations: int, number of optimization iterations (T)

        Returns:
            dict matching base class search() signature.
        """
        if self.n_top_patterns <= 1:
            # N=1: use base class greedy loop (step already handles full pipeline)
            return super().search(init_prompt, eval_fn, iterations=iterations)

        # N>1: generate N candidates per iteration, evaluate each, pick best
        current_prompt = init_prompt
        best_prompt = current_prompt
        best_accuracy = 0.0
        accuracy_history = []
        prompt_history = []
        all_records = []

        for iteration in range(iterations):
            print(f"\n--- AMPO Iteration {iteration} ---")
            print(f"Prompt ({len(current_prompt)} chars): {current_prompt[:120]}...")

            # Line 9: Evaluate P on Dtrain
            accuracy, records = eval_fn(current_prompt)
            accuracy_history.append(accuracy)
            prompt_history.append({
                "iteration": iteration,
                "prompt": current_prompt,
                "train_accuracy": accuracy,
            })
            all_records.extend(records)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_prompt = current_prompt

            errors = [r for r in records if not r["correct"]]
            if not errors:
                print("  No errors on train set -- stopping early")
                break

            # Line 9: Sample K failed cases
            k = min(self.k_failures, len(errors))
            sampled_errors = random.sample(errors, k)
            print(f"  AMPO: {len(errors)} errors at {accuracy:.0%}, sampling {k}")

            # Line 11: LLM-Analyzer
            reasons = self._analyze(sampled_errors, current_prompt)
            if not reasons:
                print("  AMPO: analyzer returned no reasons, skipping")
                continue
            print(f"  AMPO Analyzer: {len(reasons)} root causes")

            # Line 12: LLM-Summarizer
            scored_patterns = self._summarize(reasons, current_prompt)
            if not scored_patterns:
                print("  AMPO: summarizer returned no patterns, skipping")
                continue
            for i, (p, s) in enumerate(scored_patterns):
                print(f"  AMPO Pattern {i+1} (score={s}): {p[:100]}...")

            # Line 14: Select top N patterns
            top_patterns = scored_patterns[:self.n_top_patterns]

            # Lines 15-18: For each top pattern, revise -> prune -> evaluate
            best_candidate = None
            best_candidate_acc = -1.0
            best_candidate_pattern = None

            for j, (pattern_text, score) in enumerate(top_patterns):
                print(f"  AMPO: revising with pattern {j+1}/{len(top_patterns)} (score={score})")

                # Line 15: LLM-Revisor optimizes
                candidate, raw_resp = self._revise(current_prompt, pattern_text)

                # Line 17: LLM-Revisor prunes
                candidate = self._prune(candidate)

                # Line 18: Evaluate on Dval (here we use eval_fn)
                cand_acc, cand_records = eval_fn(candidate)
                all_records.extend(cand_records)
                print(f"    Candidate {j+1}: {cand_acc:.3f} ({len(candidate)} chars)")

                if cand_acc > best_candidate_acc:
                    best_candidate_acc = cand_acc
                    best_candidate = candidate
                    best_candidate_pattern = pattern_text

            # Line 19: Update P = P*
            if best_candidate is not None and best_candidate_acc > accuracy:
                print(f"  ACCEPTED best candidate (acc {accuracy:.3f} -> {best_candidate_acc:.3f})")
                current_prompt = best_candidate
                self.feedback(best_candidate_pattern, True, best_candidate_acc - accuracy)
            else:
                delta = best_candidate_acc - accuracy if best_candidate is not None else 0.0
                print(f"  REJECTED all candidates (best delta: {delta:+.3f})")
                if best_candidate_pattern is not None:
                    self.feedback(best_candidate_pattern, False, delta)

            if best_candidate_acc > best_accuracy and best_candidate is not None:
                best_accuracy = best_candidate_acc
                best_prompt = best_candidate

        return {
            "best_prompt": best_prompt,
            "best_accuracy": best_accuracy,
            "accuracy_history": accuracy_history,
            "prompt_history": prompt_history,
            "all_records": all_records,
        }