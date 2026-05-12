"""DSPy MIPROv2 optimization support for auto_prompt.

Keeps DSPy-specific logic separate from the main optimization loop.
All shared helpers (eval_fn factories, data loading, inference) are imported from auto_prompt.
"""

from prompt_optimizers import DSPyOptimizer


def _binary_df_to_acne_columns(df):
    """Convert a binary-task DataFrame (context + label 0/1) to ACNE-style columns for DSPy reuse."""
    acne_df = df.copy()
    acne_df["chunk"] = acne_df["context"]
    acne_df["found_keywords"] = ""
    acne_df["val_unified"] = acne_df["label"].map({1: "Yes", 0: "No"})
    return acne_df


def _multichoice_df_to_acne_columns(df):
    """Convert an n-way multiple-choice DataFrame (context + letter answer) to ACNE-style columns."""
    acne_df = df.copy()
    acne_df["chunk"] = acne_df["context"]
    acne_df["found_keywords"] = ""
    acne_df["val_unified"] = acne_df["answer"]
    return acne_df


def run_dspy_optimization(feature_name, train_df, val_df, test_df, baseline_prompt,
                          iterations, n_workers, eval_fn, model_id,
                          dspy_train_df=None, dspy_val_df=None, dspy_test_df=None):
    """DSPy MIPROv2 optimization with val-gated selection and a final test pass.

    DSPy's compile() trains on train_df; its post-compile dev evaluation uses val_df
    (so the score MIPROv2 reports is val-set accuracy). The chosen optimized
    instruction is then evaluated on test_df via native inference.

    dspy_*_df: DataFrames in DSPy/ACNE column format (chunk, val_unified,
    found_keywords). Pass None to use the corresponding train/val/test_df directly.
    """
    print(f"\nProcessing {feature_name} with DSPy MIPROv2")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    dspy_optimizer = DSPyOptimizer(model_id=model_id, n_workers=n_workers)
    dspy_result = dspy_optimizer.optimize(
        dspy_train_df if dspy_train_df is not None else train_df,
        dspy_val_df if dspy_val_df is not None else val_df,
        baseline_prompt=baseline_prompt,
        num_trials=iterations,
    )

    optimized_instruction = dspy_result["optimized_instruction"]
    dspy_val_score = dspy_result["dspy_eval_score"]
    print(f"  DSPy val accuracy: {dspy_val_score:.1f}%")

    # Final test pass on the held-out test split with native inference.
    print("\n--- Test with optimized instruction (native inference) ---")
    test_metrics = eval_fn(test_df, optimized_instruction, desc="Test (DSPy instruction, native inference)")

    all_records = []
    for rec in test_metrics["records"]:
        rec["feature"] = feature_name
        rec["split"] = "test"
        rec["iteration"] = iterations - 1
        rec["prompt"] = optimized_instruction
    all_records.extend(test_metrics["records"])

    # Build prompt_history mirroring the non-dspy loop format. Each trial's
    # candidate is re-scored on val_df via native inference for a comparable
    # val-acc trajectory.
    trial_history = dspy_result.get("trial_history", [])
    prompt_history = []
    best_train_so_far = 0.0
    best_val_so_far = -1.0
    best_prompt_so_far = baseline_prompt
    for i, trial in enumerate(trial_history):
        train_score = trial["score"] if trial["score"] is not None else 0.0
        if train_score > 1.0:
            train_score = train_score / 100.0
        candidate_prompt = trial["instruction"] or ""
        if candidate_prompt:
            cand_val_metrics = eval_fn(val_df, candidate_prompt, desc=f"DSPy trial {i} val")
            candidate_val_accuracy = cand_val_metrics["combined"]
        else:
            candidate_val_accuracy = 0.0
        accepted = candidate_val_accuracy > best_val_so_far
        if accepted:
            best_train_so_far = train_score
            best_val_so_far = candidate_val_accuracy
            best_prompt_so_far = candidate_prompt
        prompt_history.append({
            "iteration": i,
            "prompt": best_prompt_so_far,
            "train_accuracy": best_train_so_far,
            "val_accuracy": best_val_so_far,
            "candidate_prompt": candidate_prompt,
            "candidate_train_accuracy": train_score,
            "candidate_val_accuracy": candidate_val_accuracy,
            "accepted": accepted,
            "candidate_raw_response": "",
        })

    return {
        "feature_name": feature_name,
        "baseline_prompt": baseline_prompt,
        "train_chunks": len(train_df),
        "val_chunks": len(val_df),
        "test_chunks": len(test_df),
        "train_accuracy_history": [h["train_accuracy"] for h in prompt_history],
        "val_accuracy_history": [h["val_accuracy"] for h in prompt_history],
        "best_val_accuracy": best_val_so_far if prompt_history else dspy_val_score / 100.0,
        "test_metrics": test_metrics,
        "best_prompt": optimized_instruction,
        "prompt_history": prompt_history,
        "records": all_records,
    }
