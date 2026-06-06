"""
Postprocess .jsonl records in full_inference/records. Ouput a csv file.

Because there may be multiple records for the same feature on a particular patient, we need to pool the results somehow. Each feature can have one version (column) for each period: pre-index, treatment assessment, and outcom assessment. Most features will only appear as one or two of these as defined in full_inference.py. Each feature also needs to have a version for each keyword. In the end, we want a csv with one row per patient, and one column per feature/period/keyword combination that can exist.

output file: full_inference/records/pooled_records.csv

This is how we will pool each feature:

Smoking status: majority case
Smoking amount: majority case
Alcohol status: majority case
Alcohol amount: majority case
Transplant: One boolean column per organ (keyword)
Transplant Date: One date column per organ (keyword) choosing the earliest date for that organ
Disease: One column per disease name
Cancer cancer: One column per cancer type keyword
Cancer stage at diagnosis: One column per cancer type keyword, choosing the highest stage
Cancer max stage: One column per cancer type keyword, choosing the highest stage
Cancer family any: One column per cancer type keyword
Antibiotic duration numeric: one feature per abx. Take the longest duration at each date, then summing across unique dates.
Contraceptives: One boolean column (pooled) – "Yes" if any contraceptive record exists, else absent.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
from utils import normalize_date
from ydata_profiling import ProfileReport
# --- Constants ---

OUTPUT_DIR = Path("full_inference_out")
OUTPUT_PATH = OUTPUT_DIR / "pooled_records.csv"

# Features pooled across keywords via majority vote
MAJORITY_FEATURES = {"smoking_status", "smoking_amount", "alcohol_status", "alcohol_amount"}

# Features with one boolean column per keyword (any "Yes" -> "Yes")
BOOLEAN_PER_KEYWORD_FEATURES = {"transplant", "cancer_cancer", "cancer_family_any", "antibiotics"}

# Stage features: one column per keyword, pick highest stage
STAGE_RANK = ["Stage 0", "Stage I", "Stage II or III", "Stage IV"]
STAGE_FEATURES = {"cancer_stage_at_diagnosis", "cancer_maximum_stage"}

# Date feature: one column per keyword, pick earliest date
DATE_FEATURES = {"cancer_date_of_diagnosis"}

# Antibiotic keyword -> class mapping
ABX_KW_TO_CLASS = {}
for _cls, _kws in {
    "TETRACYCLINE": [
        "tetracy", "tetra", "cycline", "doxy", "minocycl",
        "adoxa", "brodspec", "cleeravue", "declomycin", "doryx",
        "dynacin", "minocin", "nuzyra", "sumycin", "vibramycin",
    ],
    "TMP-SMX": [
        "trimethoprim sulfamethoxazole", "tmp", "smx", "bactrim", "septra",
        "smz", "sulfameth", "trimeth", "co-trim", "sxt",
    ],
    "AMOXICILLIN": [
        "amoxicillin", "amoxicot", "amoxil", "amox", "dispermox",
        "moxatag", "moxilin", "trimox",
    ],
    "CEPHALEXIN": [
        "cephalex", "keflex", "bio-cef", "panixine",
    ],
    "AZITHROMYCIN": [
        "azith", "zithro", "z-pak", "zpak", "z pak", "zmax",
        "z-max", "z max",
    ],
}.items():
    for _kw in _kws:
        ABX_KW_TO_CLASS[_kw] = _cls


def abx_keyword_to_class(kw):
    """Map an antibiotic keyword to its drug class. Returns kw unchanged if no match."""
    kw_lower = kw.lower()
    if kw_lower in ABX_KW_TO_CLASS:
        return ABX_KW_TO_CLASS[kw_lower]
    # Substring match as fallback
    for abx_kw, abx_class in ABX_KW_TO_CLASS.items():
        if abx_kw in kw_lower or kw_lower in abx_kw:
            return abx_class
    return kw


def abx_record_to_class(rec):
    """Get the abx class for a record, using Medication_Description for structured data."""
    kw = rec["keyword"]
    if kw == "STRUCTURED_DATA":
        med = rec.get("Medication_Description", "")
        return abx_keyword_to_class(med)
    return abx_keyword_to_class(kw)


# --- Custom pooling functions ---

def pool_majority(preds):
    """Return the most common prediction."""
    counts = Counter(preds)
    return counts.most_common(1)[0][0]


def pool_any_yes(preds):
    """Return 'Yes' if any prediction is 'Yes', else 'No'."""
    if "Yes" in preds:
        return "Yes"
    return "No"


def pool_highest_stage(preds):
    """Return the highest cancer stage from predictions."""
    best_rank = -1
    best_pred = preds[0]
    for p in preds:
        if p in STAGE_RANK:
            rank = STAGE_RANK.index(p)
            if rank > best_rank:
                best_rank = rank
                best_pred = p
    return best_pred


def pool_earliest_date(preds):
    """Return the earliest date (YYYYMMDD format). Ignores 'U' and 'X'."""
    dates = []
    for p in preds:
        if p not in ("U", "X"):
            # Handle MMDDYYYY when LLM swaps the format
            if len(p) == 8 and p.isdigit() and int(p[:4]) < 1900:
                swapped = p[4:] + p[:4]
                if int(swapped[:4]) >= 1900 and 1 <= int(swapped[4:6]) <= 12 and 1 <= int(swapped[6:8]) <= 31:
                    p = swapped
                else:
                    print(f"WARNING: Unparseable cancer date '{p}', skipping")
                    continue
            dates.append(pd.to_datetime(p, format="%Y%m%d"))
    if not dates:
        return "X"
    return min(dates).strftime("%Y-%m-%d")


def get_index_dates(jsonl_files):
    """Read index dates from the .jsonl record files.

    Returns dict: {empi_str: index_date_datetime}
    """
    index_dates = {}
    for jsonl_path in jsonl_files:
        empi = jsonl_path.stem
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["feature_name"] == "index_date":
                    index_dates[empi] = normalize_date(rec["prediction"])
                    break
    return index_dates


def classify_period(record_date, index_date, outcome_window_start):
    """Classify a record date into pre_index, treatment, or outcome period."""
    if record_date <= index_date:
        return "pre_index"
    elif record_date < outcome_window_start:
        return "treatment"
    else:
        return "outcome"


def pool_abx_duration_numeric(records):
    """Custom pooling for antibiotic_duration_numeric.

    Take the longest duration at each date, then sum across unique dates.
    """
    # Group by date
    by_date = defaultdict(list)
    for rec in records:
        by_date[rec["date"]].append(rec["prediction"])
    # For each date, take max numeric value
    total = 0
    for date_key, preds in by_date.items():
        numeric_preds = []
        for p in preds:
            if p != "F":
                numeric_preds.append(int(p))
        if numeric_preds:
            total += max(numeric_preds)
    if total == 0 and all(
        rec["prediction"] == "F" for rec in records
    ):
        return "F"
    return str(total)


def _group_by_period_and_keyword(feature_records, rec_key_fn=None):
    """Group records into {period: {keyword: [records]}}.

    Args:
        rec_key_fn: Optional function(record) -> key to use instead of rec["keyword"].
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in feature_records:
        kw = rec_key_fn(rec) if rec_key_fn else rec["keyword"]
        grouped[rec["_period"]][kw].append(rec)
    return grouped


def _group_by_period(feature_records):
    """Group records into {period: [records]}."""
    grouped = defaultdict(list)
    for rec in feature_records:
        grouped[rec["_period"]].append(rec)
    return grouped


def pool_patient_records(records, index_date):
    """Pool all records for a single patient into a dict of column_name -> value."""
    outcome_window_start = index_date + pd.Timedelta(days=365 * 2)
    result = {}

    # Parse dates and classify periods
    ABX_FEATURES = {"antibiotics", "antibiotic_duration_numeric"}
    TREATMENT_ONLY_FEATURES = {"contraceptives"}
    for rec in records:
        rec["_parsed_date"] = normalize_date(rec["date"])
        if rec["feature_name"] in ABX_FEATURES or rec["feature_name"] in TREATMENT_ONLY_FEATURES:
            rec["_period"] = "treatment"
        else:
            rec["_period"] = classify_period(rec["_parsed_date"], index_date, outcome_window_start)

    # Group by feature_name
    by_feature = defaultdict(list)
    for rec in records:
        by_feature[rec["feature_name"]].append(rec)

    for feature_name, feature_records in by_feature.items():

        # --- Majority vote features (smoking, alcohol): pool across all keywords ---
        if feature_name in MAJORITY_FEATURES:
            for period, period_recs in _group_by_period(feature_records).items():
                preds = [r["prediction"] for r in period_recs]
                col_name = f"{feature_name}__{period}__pooled"
                result[col_name] = pool_majority(preds)

        # --- Disease: one column per disease name, presence only ---
        elif feature_name == "disease":
            for period, by_kw in _group_by_period_and_keyword(feature_records).items():
                for kw, kw_recs in by_kw.items():
                    disease_name = kw_recs[0].get("disease", kw_recs[0].get("Code", "unknown"))
                    disease_name = disease_name.replace(",", ";")
                    col_name = f"disease__{period}__{disease_name}"
                    result[col_name] = "Yes"

        # --- Boolean per-keyword features: any "Yes" -> "Yes" ---
        elif feature_name in BOOLEAN_PER_KEYWORD_FEATURES:
            rec_key_fn = abx_record_to_class if feature_name == "antibiotics" else None
            for period, by_kw in _group_by_period_and_keyword(feature_records, rec_key_fn=rec_key_fn).items():
                for kw, kw_recs in by_kw.items():
                    preds = [r["prediction"] for r in kw_recs]
                    # Skip cancer_cancer keywords where no record says "Yes"
                    # (filters out noise columns from non-cancer ICD codes in existing data)
                    if feature_name == "cancer_cancer" and "Yes" not in preds:
                        continue
                    col_name = f"{feature_name}__{period}__{kw}"
                    result[col_name] = pool_any_yes(preds)
                    # Add ICD codes for cancer
                    if feature_name == "cancer_cancer":
                        icds = sorted({r["icd_code"] for r in kw_recs if r.get("icd_code")})
                        if icds:
                            result[f"cancer_cancer_icd__{period}__{kw}"] = "; ".join(icds)
                    # Add medication descriptions for antibiotics
                    if feature_name == "antibiotics":
                        meds = sorted({r["Medication_Description"] for r in kw_recs if r.get("Medication_Description")})
                        if meds:
                            result[f"antibiotics_meds__{period}__{kw}"] = "; ".join(meds)

        # --- Stage features: highest stage per keyword ---
        elif feature_name in STAGE_FEATURES:
            for period, by_kw in _group_by_period_and_keyword(feature_records).items():
                for kw, kw_recs in by_kw.items():
                    preds = [r["prediction"] for r in kw_recs]
                    col_name = f"{feature_name}__{period}__{kw}"
                    result[col_name] = pool_highest_stage(preds)

        # --- Date features: earliest date per keyword ---
        elif feature_name in DATE_FEATURES:
            for period, by_kw in _group_by_period_and_keyword(feature_records).items():
                for kw, kw_recs in by_kw.items():
                    preds = [r["prediction"] for r in kw_recs]
                    col_name = f"{feature_name}__{period}__{kw}"
                    result[col_name] = pool_earliest_date(preds)

        # --- Antibiotic duration numeric: max per date, then sum ---
        elif feature_name == "antibiotic_duration_numeric":
            for period, by_kw in _group_by_period_and_keyword(feature_records, rec_key_fn=abx_record_to_class).items():
                for kw, kw_recs in by_kw.items():
                    col_name = f"antibiotic_duration_numeric__{period}__{kw}"
                    result[col_name] = pool_abx_duration_numeric(kw_recs)

        # --- Demographics: one column per keyword, take the value directly ---
        elif feature_name == "demographics":
            for period, by_kw in _group_by_period_and_keyword(feature_records).items():
                for kw, kw_recs in by_kw.items():
                    col_name = f"demographics__{kw}"
                    result[col_name] = kw_recs[0]["prediction"]

        # --- Contraceptives: boolean presence (any record -> "Yes") ---
        elif feature_name == "contraceptives":
            for period, period_recs in _group_by_period(feature_records).items():
                col_name = f"contraceptives__{period}__pooled"
                result[col_name] = "Yes"
                meds = sorted({r["Medication_Description"] for r in period_recs if r.get("Medication_Description")})
                if meds:
                    result[f"contraceptives_meds__{period}__pooled"] = "; ".join(meds)

        elif feature_name == "index_date":
            result["index_date"] = str(index_date.date())

        else:
            print(f"WARNING: Unknown feature_name '{feature_name}', skipping")

    # Earliest abx date across all antibiotic records
    if "antibiotics" in by_feature:
        abx_dates = [r["_parsed_date"] for r in by_feature["antibiotics"]]
        if abx_dates:
            result["earliest_abx_date"] = str(min(abx_dates).date())

    return result


def main():
    # Step 1: Discover all patient files
    jsonl_files = sorted((OUTPUT_DIR / "records").glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} patient files in {OUTPUT_DIR}")

    # Step 2: Get index dates from .jsonl records
    print("Reading index dates from record files...")
    index_dates = get_index_dates(jsonl_files)
    print(f"Got index dates for {len(index_dates)}/{len(jsonl_files)} patients")

    # Step 3: Load, classify, pool
    rows = []
    skipped = 0
    for jsonl_path in tqdm(jsonl_files):
        empi = jsonl_path.stem
        if empi not in index_dates:
            skipped += 1
            continue

        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            continue

        patient_row = pool_patient_records(records, index_dates[empi])
        patient_row["EMPI"] = empi
        rows.append(patient_row)

    if skipped:
        print(f"Skipped {skipped} patients (no index date found)")

    # Step 4: Build DataFrame and write CSV
    df = pd.DataFrame(rows)
    df = df.set_index("EMPI")
    df = df.reindex(sorted(df.columns), axis=1)

    df.to_csv(OUTPUT_PATH)
    print(f"Wrote {len(df)} patients x {len(df.columns)} columns to {OUTPUT_PATH}")

    # --- Cancer rate by abx status ---
    abx_cols = [c for c in df.columns if c.startswith("antibiotics__")]
    cancer_cols = [c for c in df.columns if c.startswith("cancer_cancer__") and not c.startswith("cancer_cancer_icd__")]
    has_abx = (df[abx_cols] == "Yes").any(axis=1) if abx_cols else pd.Series(False, index=df.index)
    has_cancer = (df[cancer_cols] == "Yes").any(axis=1) if cancer_cols else pd.Series(False, index=df.index)

    n_abx = has_abx.sum()
    n_no_abx = (~has_abx).sum()
    cancer_abx = (has_abx & has_cancer).sum()
    cancer_no_abx = (~has_abx & has_cancer).sum()
    pct_abx = 100 * cancer_abx / n_abx if n_abx else 0
    pct_no_abx = 100 * cancer_no_abx / n_no_abx if n_no_abx else 0
    print(f"\nCancer rate by abx status:")
    print(f"  Abx patients:     {cancer_abx}/{n_abx} ({pct_abx:.1f}%) have cancer")
    print(f"  Non-abx patients: {cancer_no_abx}/{n_no_abx} ({pct_no_abx:.1f}%) have cancer")

    # Generate pandas profiling report
    profile_path = OUTPUT_DIR / "pooled_records_profile.html"
    print(f"Generating profile report...")
    profile = ProfileReport(df, title="Pooled Records Profile")
    profile.to_file(profile_path)
    print(f"Wrote profile report to {profile_path}")


if __name__ == "__main__":
    main()
