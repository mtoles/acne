"""DSPy MIPROv2 optimization support for auto_prompt.

Keeps DSPy-specific logic separate from the main optimization loop.
All shared helpers (eval_fn factories, data loading, inference) are imported from auto_prompt.
"""

from prompt_optimizers import DSPyOptimizer


def _odd_df_to_acne_columns(df):
    """Convert ODD DataFrame columns to ACNE-style columns for DSPy reuse."""
    acne_df = df.copy()
    acne_df["chunk"] = acne_df["context"]
    acne_df["found_keywords"] = ""
    acne_df["val_unified"] = acne_df["label"].map({1: "Yes", 0: "No"})
    return acne_df


def run_dspy_optimization(feature_name, train_df, eval_df, baseline_prompt, iterations, n_workers, eval_fn, model_id, dspy_train_df=None, dspy_eval_df=None):
    """Generic DSPy MIPROv2 optimization. eval_fn(df, prompt_text, desc) -> metrics_dict.

    dspy_train_df/dspy_eval_df: DataFrames with ACNE-style columns for DSPy (chunk, val_unified, found_keywords).
    If None, train_df/eval_df are used directly (assumed to already have the right columns).
    """
    print(f"\nProcessing {feature_name} with DSPy MIPROv2")
    print(f"Train: {len(train_df)}, Eval: {len(eval_df)}")

    dspy_optimizer = DSPyOptimizer(model_id=model_id, n_workers=n_workers)
    dspy_result = dspy_optimizer.optimize(
        dspy_train_df if dspy_train_df is not None else train_df,
        dspy_eval_df if dspy_eval_df is not None else eval_df,
        baseline_prompt=baseline_prompt,
        num_trials=iterations,
    )

    optimized_instruction = dspy_result["optimized_instruction"]
    dspy_eval_score = dspy_result["dspy_eval_score"]
    print(f"  DSPy eval accuracy: {dspy_eval_score:.1f}%")

    # Evaluate extracted instruction through native inference
    print("\n--- Evaluating extracted instruction through native inference ---")
    eval_metrics = eval_fn(eval_df, optimized_instruction, desc="Eval (DSPy instruction, native inference)")

    all_records = []
    for rec in eval_metrics["records"]:
        rec["feature"] = feature_name
        rec["split"] = "eval"
        rec["iteration"] = iterations - 1
        rec["prompt"] = optimized_instruction
    all_records.extend(eval_metrics["records"])

    # Build prompt_history from trial history
    trial_history = dspy_result.get("trial_history", [])
    prompt_history = []
    best_so_far = 0.0
    best_prompt_so_far = baseline_prompt
    for i, trial in enumerate(trial_history):
        score = trial["score"] if trial["score"] is not None else 0.0
        if score > 1.0:
            score = score / 100.0
        candidate_prompt = trial["instruction"] or ""
        accepted = score > best_so_far
        if accepted:
            best_so_far = score
            best_prompt_so_far = candidate_prompt
        prompt_history.append({
            "iteration": i,
            "prompt": best_prompt_so_far,
            "train_accuracy": best_so_far,
            "candidate_prompt": candidate_prompt,
            "candidate_accuracy": score,
            "accepted": accepted,
            "candidate_raw_response": "",
        })

    return {
        "feature_name": feature_name,
        "baseline_prompt": baseline_prompt,
        "train_chunks": len(train_df),
        "eval_chunks": len(eval_df),
        "train_accuracy_history": [h["train_accuracy"] for h in prompt_history],
        "best_train_accuracy": best_so_far if prompt_history else dspy_eval_score / 100.0,
        "eval_metrics": eval_metrics,
        "best_prompt": optimized_instruction,
        "prompt_history": prompt_history,
        "records": all_records,
    }
