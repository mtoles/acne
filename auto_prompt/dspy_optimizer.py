"""DSPy MIPROv2 optimization support for auto_prompt.

Keeps DSPy-specific logic separate from the main optimization loop.
All shared helpers (eval_fn factories, data loading, inference) are imported from auto_prompt.
"""

from prompt_optimizers import (
    DSPyOptimizer, GEPAOptimizer,
    _make_choice_match, _make_dspy_classification_metric,
)


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
                          test_full_df=None,
                          dspy_train_df=None, dspy_val_df=None, dspy_test_df=None,
                          optimizer_name=None,
                          choice_task=False, choice_word_to_letter=None,
                          router=None):
    """DSPy-family (MIPROv2 or GEPA) optimization with a final test pass.

    Both optimizers run their own internal loop and return an optimized
    instruction plus a per-candidate trial_history. DSPy's compile() trains on
    train_df; the chosen instruction is evaluated on test_full_df via native
    inference for the final report, and each candidate is re-scored on val_df
    (native inference) for a comparable val-acc trajectory. test_df is used
    per-trial for the plotting curve.

    optimizer_name: "mipro" (MIPROv2) or "gepa". GEPA additionally uses val_df as
    its Pareto valset and derives its budget from (train + val) sizes.

    choice_task: True for lettered-legend tasks (binary Yes/No, multiple-choice)
    where the free-form classifier emits e.g. "A. Yes" against a "Yes" gold. Uses
    letter-aware matching for the DSPy-internal metric so MIPRO's search and GEPA's
    reflection get real signal instead of a uniformly-zero score. False for ACNE
    (strict exact-match, unchanged). choice_word_to_letter maps a full word answer
    to its option letter (e.g. {"yes": "A", "no": "B"}); pass None when the gold is
    already a letter (leading-letter extraction suffices).

    dspy_*_df: DataFrames in DSPy/ACNE column format (chunk, val_unified,
    found_keywords). Pass None to use the corresponding train/val/test_df directly.
    """
    if optimizer_name not in ("mipro", "gepa"):
        raise ValueError(f"optimizer_name must be 'mipro' or 'gepa', got {optimizer_name!r}")
    if router is None:
        raise ValueError("router (the shared MrModel) is required so DSPy load-balances "
                         "across the same vLLM endpoints as native inference")
    if test_full_df is None:
        test_full_df = test_df
    opt_label = "GEPA" if optimizer_name == "gepa" else "MIPROv2"
    print(f"\nProcessing {feature_name} with DSPy {opt_label}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test (per-iter): {len(test_df)}, Test (final): {len(test_full_df)}")

    opt_train_df = dspy_train_df if dspy_train_df is not None else train_df
    opt_val_df = dspy_val_df if dspy_val_df is not None else val_df

    # Scoring for the DSPy-internal loop: letter-aware for choice tasks, strict
    # exact-match for ACNE. Both optimizers share this so the comparison is fair.
    if choice_task:
        dspy_metric = _make_dspy_classification_metric(choice_word_to_letter)
        gepa_is_correct = _make_choice_match(choice_word_to_letter)
    else:
        dspy_metric = None          # DSPyOptimizer defaults to strict exact-match
        gepa_is_correct = None      # GEPAOptimizer defaults to strict exact-match

    if optimizer_name == "gepa":
        gepa_optimizer = GEPAOptimizer(router=router, n_workers=n_workers)
        dspy_result = gepa_optimizer.optimize(
            opt_train_df, opt_val_df,
            baseline_prompt=baseline_prompt,
            iterations=iterations,
            is_correct=gepa_is_correct,
        )
    elif optimizer_name == "mipro":
        dspy_optimizer = DSPyOptimizer(router=router, n_workers=n_workers)
        dspy_result = dspy_optimizer.optimize(
            opt_train_df, opt_val_df,
            baseline_prompt=baseline_prompt,
            num_trials=iterations,
            metric=dspy_metric,
        )

    optimized_instruction = dspy_result["optimized_instruction"]
    dspy_val_score = dspy_result["dspy_eval_score"]
    print(f"  DSPy val accuracy: {dspy_val_score:.1f}%")

    # Final test pass on the full held-out test split with native inference.
    print(f"\n--- Test with optimized instruction (native inference) on full test set ({len(test_full_df)}) ---")
    test_metrics = eval_fn(test_full_df, optimized_instruction, desc="Test (DSPy instruction, native inference, full)")

    all_records = []
    for rec in test_metrics["records"]:
        rec["feature"] = feature_name
        rec["split"] = "test"
        rec["iteration"] = iterations - 1
        rec["prompt"] = optimized_instruction
    all_records.extend(test_metrics["records"])

    # Build prompt_history mirroring the non-dspy loop format. Each trial's
    # candidate is re-scored on val_df via native inference for a comparable
    # val-acc trajectory. Test is also scored per trial purely for plotting
    # (never read by the optimizer).
    trial_history = dspy_result.get("trial_history", [])
    prompt_history = []
    best_train_so_far = 0.0
    best_val_so_far = -1.0
    best_test_so_far = 0.0
    best_prompt_so_far = baseline_prompt
    for i, trial in enumerate(trial_history):
        candidate_prompt = trial["instruction"] or ""
        if candidate_prompt:
            cand_train_metrics = eval_fn(train_df, candidate_prompt, desc=f"DSPy trial {i} train")
            candidate_train_accuracy = cand_train_metrics["combined"]
            cand_val_metrics = eval_fn(val_df, candidate_prompt, desc=f"DSPy trial {i} val")
            candidate_val_accuracy = cand_val_metrics["combined"]
            cand_test_metrics = eval_fn(test_df, candidate_prompt, desc=f"DSPy trial {i} test (plotting only)")
            candidate_test_accuracy = cand_test_metrics["combined"]
        else:
            candidate_train_accuracy = 0.0
            candidate_val_accuracy = 0.0
            candidate_test_accuracy = 0.0
        accepted = bool(candidate_val_accuracy > best_val_so_far)
        if accepted:
            best_train_so_far = candidate_train_accuracy
            best_val_so_far = candidate_val_accuracy
            best_test_so_far = candidate_test_accuracy
            best_prompt_so_far = candidate_prompt
        prompt_history.append({
            "iteration": i,
            "prompt": best_prompt_so_far,
            "train_accuracy": best_train_so_far,
            "val_accuracy": best_val_so_far,
            "test_accuracy": best_test_so_far,
            "candidate_prompt": candidate_prompt,
            "candidate_train_accuracy": candidate_train_accuracy,
            "candidate_val_accuracy": candidate_val_accuracy,
            "candidate_test_accuracy": candidate_test_accuracy,
            "accepted": accepted,
            "candidate_raw_response": "",
        })

    return {
        "feature_name": feature_name,
        "baseline_prompt": baseline_prompt,
        "train_chunks": len(train_df),
        "val_chunks": len(val_df),
        "test_chunks": len(test_full_df),
        "train_accuracy_history": [h["train_accuracy"] for h in prompt_history],
        "val_accuracy_history": [h["val_accuracy"] for h in prompt_history],
        "test_accuracy_history": [h["test_accuracy"] for h in prompt_history],
        "best_val_accuracy": best_val_so_far if prompt_history else dspy_val_score / 100.0,
        "test_metrics": test_metrics,
        "best_prompt": optimized_instruction,
        "prompt_history": prompt_history,
        "records": all_records,
    }
