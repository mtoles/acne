"""Post-pipeline sanity checks (unit-test style) for the acne antibiotic/cancer study.

Validates the three artifact layers end-to-end and exits non-zero if any CHECK fails:
  1. full_inference_out/records/*.jsonl   -- per-patient extraction
  2. full_inference_out/pooled_records.csv -- pooled feature matrix
  3. stats/eda_output/COX*.csv             -- Cox regression outputs

Run as part of stats/run_all.sh (after the R pipeline). Two severities:
  CHECK  -> hard assertion; any failure makes the script exit 1 (fails run_all.sh).
  WARN   -> something suspicious but not necessarily wrong (e.g. a cancer type with
            zero events, a degenerate HR from a sparse cell); reported, never fatal.

Usage:
  python sanity_check.py                 # scan ALL record files
  python sanity_check.py --records-sample 2000   # scan a random 2000 for speed
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from comorbidity_sheet import (
    load_comorbidity_sheet,
    cancer_specific_applicability,
    universal_conditions_by_category,
    slugify,
)

COHORT_JSON = REPO_ROOT / "cohort" / "empis.json"
# Pooled column the transplant feature lands in (folded into the immunosuppressed count).
TRANSPLANT_COL = "transplant__pre_index__STRUCTURED_DATA"

RECORDS_DIR = REPO_ROOT / "full_inference_out" / "records"
POOLED_CSV = REPO_ROOT / "full_inference_out" / "pooled_records.csv"
EDA_DIR = Path(__file__).resolve().parent / "eda_output"

# Feature_names that must appear at least once somewhere in the scanned records.
EXPECTED_FEATURES = {
    "index_date", "demographics", "disease", "cancer_cancer",
    "cancer_date_of_diagnosis", "smoking_status", "alcohol_status",
    "antibiotics", "contraceptives", "cancer_family_any", "transplant", "follow_up",
}
# Fields every structured disease row must carry (the new sheet-driven schema).
DISEASE_REQUIRED_FIELDS = ("Code", "disease", "condition", "category", "date", "prediction")
# Matching covariates that must NOT appear in any Cox model (balanced by cohort matching).
MATCHING_COVARS = {"Age", "Sex", "Race", "BMI.Category", "Smoking", "Alcohol"}
# Old comorbidity covariates that the new scheme replaced -- must NOT appear in any Cox model.
RETIRED_COVARS = {"Comorbidities", "Prior.Leuk.Lymph", "Prior.HIV", "Transplant"}

_FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _to_float(s):
    """Parse a numeric cell ('1.10', '10,900,739') to float, else None. No exceptions."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    return float(s) if _FLOAT_RE.match(s) else None


class Checker:
    """Minimal test harness: check() is a hard assertion, warn() is advisory."""

    def __init__(self):
        self.passed = 0
        self.failures = []
        self.warnings = []

    def check(self, cond, msg):
        if cond:
            self.passed += 1
        else:
            self.failures.append(msg)
            print(f"  FAIL: {msg}")
        return bool(cond)

    def warn(self, cond, msg):
        if not cond:
            self.warnings.append(msg)
            print(f"  WARN: {msg}")
        return bool(cond)

    def section(self, title):
        print(f"\n===== {title} =====")

    def finish(self):
        print(f"\n===== sanity_check summary =====")
        print(f"  checks passed: {self.passed}")
        print(f"  warnings:      {len(self.warnings)}")
        print(f"  FAILURES:      {len(self.failures)}")
        for m in self.failures:
            print(f"    - {m}")
        if self.failures:
            print("SANITY CHECK FAILED")
            sys.exit(1)
        print("SANITY CHECK PASSED")


# ---------------------------------------------------------------------------
# 1. records
# ---------------------------------------------------------------------------
def check_records(ck, sample):
    ck.section("1. full_inference records")
    files = sorted(RECORDS_DIR.glob("*.jsonl"))
    ck.check(len(files) > 0, f"no record files in {RECORDS_DIR}")
    if not files:
        return
    if sample and sample < len(files):
        # deterministic stride sample (no RNG -> reproducible)
        step = len(files) // sample
        files = files[::step][:sample]
    print(f"  scanning {len(files)} record files")

    sheet = load_comorbidity_sheet()["extraction"]
    known_conditions = {info["description"] for ct in ("ICD9", "ICD10") for info in sheet[ct].values()}
    known_categories = {info["category"] for ct in ("ICD9", "ICD10") for info in sheet[ct].values()}

    seen_features = set()
    bad_disease_fields = 0
    bad_disease_category = 0
    bad_disease_condition = 0
    files_without_index = 0
    files_multi_index = 0

    for f in tqdm(files):
        n_index = 0
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fn = rec["feature_name"]
            seen_features.add(fn)
            if fn == "index_date":
                n_index += 1
            if fn == "disease":
                if any(k not in rec for k in DISEASE_REQUIRED_FIELDS):
                    bad_disease_fields += 1
                else:
                    if rec["category"] not in known_categories:
                        bad_disease_category += 1
                    if rec["condition"] not in known_conditions:
                        bad_disease_condition += 1
        if n_index == 0:
            files_without_index += 1
        elif n_index > 1:
            files_multi_index += 1

    missing = EXPECTED_FEATURES - seen_features
    ck.check(not missing, f"feature_name(s) never seen in records: {sorted(missing)}")
    ck.check(files_without_index == 0, f"{files_without_index} record file(s) have no index_date")
    ck.check(bad_disease_fields == 0, f"{bad_disease_fields} disease row(s) missing required fields {DISEASE_REQUIRED_FIELDS}")
    ck.check(bad_disease_category == 0, f"{bad_disease_category} disease row(s) have a category not in the comorbidity sheet")
    ck.check(bad_disease_condition == 0, f"{bad_disease_condition} disease row(s) have a condition not in the comorbidity sheet")
    ck.warn(files_multi_index == 0, f"{files_multi_index} record file(s) have >1 index_date row")


# ---------------------------------------------------------------------------
# 2. pooled_records.csv
# ---------------------------------------------------------------------------
def check_pooled(ck, df):
    ck.section("2. pooled_records.csv")
    if df is None:
        ck.check(False, f"missing {POOLED_CSV}")
        return
    print(f"  {len(df)} rows x {len(df.columns)} cols")

    ck.check(len(df) > 0, "pooled_records.csv is empty")
    ck.check("EMPI" in df.columns, "no EMPI column")
    if "EMPI" in df.columns:
        ck.check(df["EMPI"].is_unique, "duplicate EMPIs in pooled_records.csv")

    cancer_cols = [c for c in df.columns if c.startswith("cancer_outcome__")]
    cs_cols = [c for c in df.columns if c.startswith("cancerspecific__pre_index__")]
    comorb_cols = [c for c in df.columns if c.startswith("comorbidity_n__pre_index__")]

    # required column families exist
    ck.check(len(cancer_cols) > 0, "no cancer_outcome__ columns")
    ck.check("comorbidity_n__pre_index__chronic_inflammation" in df.columns, "missing chronic_inflammation count column")
    ck.check("comorbidity_n__pre_index__immunosuppressed" in df.columns, "missing immunosuppressed count column")
    ck.check(len(cs_cols) > 0, "no cancerspecific__ columns")
    for col in ("demographics__age_at_index_date", "smoking_status__pre_index__pooled", "index_date"):
        ck.check(col in df.columns, f"missing expected column {col}")

    # cancerspecific column count matches the sheet-derived applicability map exactly
    applic = cancer_specific_applicability([c[len("cancer_outcome__"):] for c in cancer_cols])
    expected_cs = sum(len(v) for v in applic.values())
    ck.check(len(cs_cols) == expected_cs,
             f"cancerspecific columns ({len(cs_cols)}) != applicability pairs ({expected_cs})")

    # comorbidity counts: parseable non-negative ints, and not all-zero
    for col in comorb_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        ck.check(vals.notna().all(), f"{col} has non-numeric values")
        ck.check((vals.dropna() >= 0).all(), f"{col} has negative values")
        ck.check((vals.fillna(0) > 0).any(), f"{col} is zero for every patient (feature never occurs)")
        ck.warn((vals.dropna() < 15).all(), f"{col} has an implausibly large count (>=15)")

    # binary columns only contain Yes/No; flag any that never fire
    for col in cs_cols:
        uniq = set(df[col].dropna().unique())
        ck.check(uniq <= {"Yes", "No"}, f"{col} has values outside {{Yes,No}}: {uniq - {'Yes','No'}}")
        ck.warn((df[col] == "Yes").any(), f"cancer-specific covariate never 'Yes' (constant): {col}")

    # every cancer outcome column should have at least one case
    zero_cancer = [c for c in cancer_cols if not (df[c] == "Yes").any()]
    ck.warn(not zero_cancer, f"{len(zero_cancer)} cancer_outcome__ column(s) have zero cases: {zero_cancer[:5]}")

    # cohort balance: matched 1:1 so abx should be ~50%
    abx_cols = [c for c in df.columns if c.startswith("antibiotics__treatment__")]
    if abx_cols:
        has_abx = (df[abx_cols] == "Yes").any(axis=1)
        frac = has_abx.mean()
        ck.warn(0.3 < frac < 0.7, f"any-antibiotic fraction is {frac:.2f} (expected ~0.5 for matched cohort)")


# ---------------------------------------------------------------------------
# 3. cox outputs
# ---------------------------------------------------------------------------
def _cox_characteristics(df):
    """Set of covariate labels (the bold label rows, i.e. rows whose HR is blank)."""
    return set(df["**Characteristic**"].dropna().astype(str))


def check_cox(ck):
    ck.section("3. Cox outputs (eda_output)")
    expected = ["COX1_any_cancer.csv", "COX2_abx_classes.csv", "COX3_individual_abx.csv",
                "COX4_cancer_types.csv", "COX5_abx_dose.csv"]
    for name in expected:
        p = EDA_DIR / name
        ck.check(p.exists(), f"missing Cox output {name}")

    cox1 = EDA_DIR / "COX1_any_cancer.csv"
    if cox1.exists():
        df = pd.read_csv(cox1)
        for col in ("**Characteristic**", "**HR**", "**95% CI**", "**p-value**", "model"):
            ck.check(col in df.columns, f"COX1 missing column {col}")
        models = set(df["model"].unique())
        ck.check("Fully adjusted" in models, "COX1 has no 'Fully adjusted' model")
        ck.check("Age-adjusted" not in models, "COX1 still has an 'Age-adjusted' model (should be removed)")

        chars = _cox_characteristics(df)
        for v in ("Chronic.Inflammation", "Immunosuppressed", "Prior.Cancer", "Contraceptives", "Fam.Cancer"):
            ck.check(v in chars, f"COX1 fully-adjusted is missing expected covariate {v}")
        leaked = (MATCHING_COVARS | RETIRED_COVARS) & chars
        ck.check(not leaked, f"COX1 contains covariates that should be gone: {sorted(leaked)}")

        # the exposure HR must be present and finite
        exp_rows = df[(df["**Characteristic**"] == "Yes") & df["**HR**"].notna()]
        ck.check(len(exp_rows) > 0, "COX1 has no estimated HR rows")
        hrs = [_to_float(h) for h in df["**HR**"].dropna()]
        ck.warn(all(h is None or h > 0 for h in hrs), "COX1 has a non-positive HR")
        degenerate = df[df["**95% CI**"].astype(str).str.contains("Inf", na=False)]
        ck.warn(len(degenerate) == 0, f"COX1 has {len(degenerate)} row(s) with an infinite CI (separation)")

    cox4 = EDA_DIR / "COX4_cancer_types.csv"
    if cox4.exists():
        df = pd.read_csv(cox4)
        ck.check("cancer.type" in df.columns, "COX4 missing cancer.type column")
        if "cancer.type" in df.columns:
            ck.check(df["cancer.type"].nunique() > 0, "COX4 modeled zero cancer types")
        chars = _cox_characteristics(df)
        leaked = (MATCHING_COVARS | RETIRED_COVARS) & chars
        ck.check(not leaked, f"COX4 contains covariates that should be gone: {sorted(leaked)}")
        ck.check(any(str(c).startswith("cancerspecific__") for c in chars),
                 "COX4 has no cancer-specific covariates (cancerspecific__*)")
        ck.check("Age-adjusted" not in set(df["model"].unique()), "COX4 still has an 'Age-adjusted' model")


# ---------------------------------------------------------------------------
# 4. cross-layer reconciliation (recompute derived columns from their inputs)
# ---------------------------------------------------------------------------
def check_reconciliation(ck, df):
    ck.section("4. reconciliation (recompute from inputs)")
    if df is None:
        return
    n = len(df)
    transplant = (df[TRANSPLANT_COL] == "Yes") if TRANSPLANT_COL in df.columns else pd.Series(False, index=df.index)

    # 4a. comorbidity_n__<cat> == # distinct universal conditions present pre-index
    #     (+ transplant for immunosuppressed), recomputed from the disease_cond__ columns.
    for category, conds in universal_conditions_by_category().items():
        cols = [f"disease_cond__pre_index__{slugify(c)}" for c in conds]
        present = [c for c in cols if c in df.columns]
        recomputed = (sum((df[c] == "Yes").astype(int) for c in present)
                      if present else pd.Series(0, index=df.index))
        if category == "Immunosuppressed":
            recomputed = recomputed + transplant.astype(int)
        target = f"comorbidity_n__pre_index__{slugify(category)}"
        if ck.check(target in df.columns, f"reconcile: missing {target}"):
            stored = pd.to_numeric(df[target], errors="coerce").fillna(-1).astype(int)
            mism = int((stored.values != recomputed.values).sum())
            ck.check(mism == 0, f"reconcile: {target} != recompute for {mism}/{n} patients")

    # 4b. each cancerspecific__<cancer>__<cat> binary == OR of its applicable disease_cond columns.
    cancer_cols = [c[len("cancer_outcome__"):] for c in df.columns if c.startswith("cancer_outcome__")]
    applic = cancer_specific_applicability(cancer_cols)
    cs_mismatch = cs_checked = 0
    for cancer, by_cat in applic.items():
        for category, conds in by_cat.items():
            out_col = f"cancerspecific__pre_index__{cancer}__{slugify(category)}"
            if not ck.check(out_col in df.columns, f"reconcile: missing {out_col}"):
                continue
            cols = [f"disease_cond__pre_index__{slugify(c)}" for c in conds]
            present = [c for c in cols if c in df.columns]
            recomputed = ((df[present] == "Yes").any(axis=1)
                          if present else pd.Series(False, index=df.index))
            cs_mismatch += int(((df[out_col] == "Yes").values != recomputed.values).sum())
            cs_checked += 1
    ck.check(cs_mismatch == 0,
             f"reconcile: cancerspecific binaries != recompute ({cs_mismatch} cells over {cs_checked} cols)")

    # 4c. invariant: immunosuppressed count >= transplant indicator (transplant folds in).
    if "comorbidity_n__pre_index__immunosuppressed" in df.columns:
        imm = pd.to_numeric(df["comorbidity_n__pre_index__immunosuppressed"], errors="coerce").fillna(0)
        ck.check(bool((imm.values >= transplant.astype(int).values).all()),
                 "immunosuppressed count < transplant indicator for some patient")


# ---------------------------------------------------------------------------
# 5. cohort coverage (no silent patient loss across layers)
# ---------------------------------------------------------------------------
def check_cohort_coverage(ck, df):
    ck.section("5. cohort coverage (no silent loss)")
    if not ck.check(COHORT_JSON.exists(), f"missing {COHORT_JSON}"):
        return
    data = json.loads(COHORT_JSON.read_text())
    cohort = set(map(str, data["matched_case_empis"])) | set(map(str, data["matched_control_empis"]))
    records = {f.stem for f in RECORDS_DIR.glob("*.jsonl")}
    pooled = set(df["EMPI"].astype(str)) if (df is not None and "EMPI" in df.columns) else set()
    print(f"  cohort={len(cohort)}  records={len(records)}  pooled={len(pooled)}")

    missing_rec = cohort - records
    missing_pool = cohort - pooled
    ck.check(not missing_rec,
             f"{len(missing_rec)} cohort EMPI(s) have no record file (e.g. {sorted(missing_rec)[:3]})")
    ck.check(not missing_pool,
             f"{len(missing_pool)} cohort EMPI(s) absent from pooled_records (e.g. {sorted(missing_pool)[:3]})")
    ck.warn(not (pooled - cohort),
            f"{len(pooled - cohort)} pooled EMPI(s) not in cohort (e.g. {sorted(pooled - cohort)[:3]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-sample", type=int, default=0,
                    help="scan only this many record files (0 = all)")
    args = ap.parse_args()

    print("Running sanity checks...")
    ck = Checker()
    df = pd.read_csv(POOLED_CSV, dtype=str, low_memory=False) if POOLED_CSV.exists() else None
    check_records(ck, args.records_sample)
    check_pooled(ck, df)
    check_reconciliation(ck, df)
    check_cohort_coverage(ck, df)
    check_cox(ck)
    ck.finish()


if __name__ == "__main__":
    main()
