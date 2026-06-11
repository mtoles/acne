"""Shared dataset loading/splitting for ACNE features.

Single source of truth for the train/val/test split, imported by BOTH the auto_prompt
optimizer (auto_prompt/auto_prompt.py) and the labeling UI (app/app.py) so the human-in-the-loop
UI uses byte-identical splits to the automated optimizer and the two cannot drift apart.

Kept dependency-light (only utils + pt_features) so importing it never drags in the optimizer
stack (dspy, prompt_optimizers, etc.).
"""
from pt_features import PtFeaturesMeta
from utils import get_dataset


def load_acne_feature_data(feature_name, data_source, downsample):
    """Load and 3-way split ACNE feature data.

    Returns (train_df, val_df, test_df, test_full_df, target_cls). The split is
    1/3 train, 1/3 val, 1/3 test of the per-feature corpus. train/val/test are
    downsampled per --downsample (test downsampling caps per-iter eval cost).
    test_full is always the un-downsampled test pool for the final report.
    """
    target_cls = PtFeaturesMeta.registry[feature_name]
    feature_metadata = {}
    if getattr(target_cls, "data_source_feature", None) or getattr(target_cls, "gt_column", "val_unified") != "val_unified":
        feature_metadata[feature_name] = {
            "data_source_feature": getattr(target_cls, "data_source_feature", None) or feature_name,
            "gt_column": getattr(target_cls, "gt_column", "val_unified"),
        }

    # NOTE: utils.get_dataset's `train_split` argument actually controls the
    # *eval* fraction (eval_size = int(total * train_split)). To get 2/3 in the
    # train pool and 1/3 in eval (which we'll use as test), we pass 1/3.
    datasets = get_dataset(
        data_source=data_source or "mgb",
        feature_names=[feature_name],
        train_split=1/3,
        downsample=None,
        random_state=42,
        feature_metadata=feature_metadata,
    )

    if feature_name not in datasets:
        raise ValueError(f"Dataset not found for feature: {feature_name}")

    train_pool = datasets[feature_name]["train"].copy()
    test_df = datasets[feature_name]["eval"].copy()

    # Carve val from train_pool: shuffle and split 50/50.
    train_pool = train_pool.sample(frac=1.0, random_state=42).reset_index(drop=True)
    half = len(train_pool) // 2
    val_df = train_pool.iloc[:half].copy()
    train_df = train_pool.iloc[half:].copy()

    test_full_df = test_df.copy()
    if downsample:
        if len(train_df) > downsample:
            train_df = train_df.sample(n=downsample, random_state=42).reset_index(drop=True)
        if len(val_df) > downsample:
            val_df = val_df.sample(n=downsample, random_state=42).reset_index(drop=True)
        if len(test_df) > downsample:
            test_df = test_df.sample(n=downsample, random_state=42).reset_index(drop=True)

    train_df = train_df[train_df["val_unified"] != "DROP"]
    val_df = val_df[val_df["val_unified"] != "DROP"]
    test_df = test_df[test_df["val_unified"] != "DROP"]
    test_full_df = test_full_df[test_full_df["val_unified"] != "DROP"]
    return train_df, val_df, test_df, test_full_df, target_cls
