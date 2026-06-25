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
Cancer cancer: Each cancer Diagnosis description (col D of ICD_Cancer_Codes_Grouped.xlsx,
    mapped from each record's ICD code) is categorized by its earliest diagnosis date into
    preexisting (before the outcome window) or outcome (first diagnosed in the outcome window =
    incident), producing one boolean column per diagnosis: cancer_preexisting__<dx> / cancer_outcome__<dx>.
Cancer stage at diagnosis: One column per cancer type keyword, choosing the highest stage
Cancer max stage: One column per cancer type keyword, choosing the highest stage
Cancer family any: One column per cancer type keyword
Antibiotic duration numeric: one feature per abx. Take the longest duration at each date, then summing across unique dates.
Contraceptives: One boolean column (pooled) – "Yes" if any contraceptive record exists, else absent.
"""

import json
from enum import StrEnum
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
from utils import normalize_date
# from data_profiling import ProfileReport
# --- Constants ---

OUTPUT_DIR = Path("full_inference_out")
OUTPUT_PATH = OUTPUT_DIR / "pooled_records.csv"

# Smoking and alcohol are the majority vote across all records dated ON OR BEFORE the
# index date (no record on/before index -> column omitted -> "unknown" downstream).
# Emitted as <feature>__pre_index__pooled.
ONBEFORE_MAJORITY_FEATURES = {"smoking_status", "smoking_amount", "alcohol_status", "alcohol_amount"}

# Features with one boolean column per keyword (any "Yes" -> "Yes")
# (cancer_cancer is handled separately: split into cancer_preexisting / cancer_outcome by dx date)
BOOLEAN_PER_KEYWORD_FEATURES = {"transplant", "cancer_family_any", "antibiotics"}

# Stage features: one column per keyword, pick highest stage
STAGE_RANK = ["Stage 0", "Stage I", "Stage II or III", "Stage IV"]
STAGE_FEATURES = {"cancer_stage_at_diagnosis"}

# Date feature: one column per keyword, pick earliest date (structured, from the dia table)
DATE_FEATURES = {"cancer_date_of_diagnosis"}

# --- Cancer ICD code -> Diagnosis description (col D of the grouped codes file) ---
# Cancers are reported by their Diagnosis description (col D), NOT the broader Cancer
# Group (col A). Each cancer_cancer record carries its raw ICD code; we prefix-match it
# to the codes in this file to recover the Diagnosis.
CANCER_CODES_PATH = Path("labeled_data/ICD_Cancer_Codes_Grouped.xlsx")
_cancer_code_df = pd.read_excel(CANCER_CODES_PATH)
ICD_CODE_TO_DIAGNOSIS = {
    str(row["Code"]).strip(): row["Diagnosis"] for _, row in _cancer_code_df.iterrows()
}
# Longest code first so the most specific prefix wins (all codes are 3 chars today, but
# this keeps matching robust if longer/overlapping codes are ever added).
_CANCER_CODE_PREFIXES = sorted(ICD_CODE_TO_DIAGNOSIS, key=len, reverse=True)

# Leukemia/lymphoma preexisting category: diagnoses (col D) whose text contains "leukemia"
# or "lymphoma". These are reported as their OWN preexisting category and excluded from
# the "other" preexisting-cancer category below. Stored in the same column-name form used
# for cancer_preexisting__<dx> columns (commas -> ';', spaces -> '_').
_LEUK_LYMPH_MASK = _cancer_code_df["Diagnosis"].str.contains("leukemia|lymphoma", case=False, na=False)
LEUK_LYMPH_DX_COLS = {
    str(d).replace(",", ";").replace(" ", "_")
    for d in _cancer_code_df.loc[_LEUK_LYMPH_MASK, "Diagnosis"].dropna().unique()
}


def icd_to_diagnosis(icd_code):
    """Map a raw ICD code (e.g. '173.3') to its cancer Diagnosis description by prefix match."""
    for prefix in _CANCER_CODE_PREFIXES:
        if icd_code.startswith(prefix):
            return ICD_CODE_TO_DIAGNOSIS[prefix]
    raise ValueError(f"ICD code {icd_code!r} not found in {CANCER_CODES_PATH}")

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


class CancerPeriod(StrEnum):
    """The two buckets a cancer diagnosis date can fall into, relative to the outcome window.
    A StrEnum so members interpolate directly into column names as their string value while
    still constraining equality comparisons to these two options."""

    PREEXISTING = "preexisting"  # diagnosed before the outcome window
    OUTCOME = "outcome"  # first diagnosed in the outcome window [index+2yr, ...) = incident


def classify_cancer_period(dx_date, outcome_window_start):
    """Categorize a cancer diagnosis date as preexisting (diagnosed before the outcome
    window) or outcome (first diagnosed in the outcome window [index+2yr, ...) = incident)."""
    return CancerPeriod.OUTCOME if dx_date >= outcome_window_start else CancerPeriod.PREEXISTING


def pool_abx_duration_numeric(records):
    """Custom pooling for antibiotic_duration_numeric.

    De-duplicate by VALUE within one abx class: identical duration values are assumed to be
    the same course re-mentioned across notes (counted once), while different values are
    assumed to be different courses (summed). No time window is needed because abx records
    are already restricted to the 2-year treatment window. Returns the sum of the distinct
    positive durations; "F" if the drug was taken but no duration could be determined; else "0".
    """
    distinct = {int(rec["prediction"]) for rec in records if rec["prediction"] != "F"}
    distinct.discard(0)
    if distinct:
        return str(sum(distinct))
    if all(rec["prediction"] == "F" for rec in records):
        return "F"
    return "0"


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
    # Cancer occurrence/date are split into preexisting/outcome by the diagnosis date.
    CANCER_DATE_FEATURES = {"cancer_cancer"} | DATE_FEATURES
    for rec in records:
        rec["_parsed_date"] = normalize_date(rec["date"])
        fn = rec["feature_name"]
        if fn in ABX_FEATURES or fn in TREATMENT_ONLY_FEATURES:
            rec["_period"] = "treatment"
        elif fn in STAGE_FEATURES:
            # Stage is only extracted for outcome-window cancers (see full_inference._cancer_rows).
            rec["_period"] = "outcome"
        elif fn in CANCER_DATE_FEATURES:
            rec["_period"] = classify_cancer_period(rec["_parsed_date"], outcome_window_start)
        else:
            rec["_period"] = classify_period(rec["_parsed_date"], index_date, outcome_window_start)

    # Group by feature_name
    by_feature = defaultdict(list)
    for rec in records:
        by_feature[rec["feature_name"]].append(rec)

    for feature_name, feature_records in by_feature.items():

        # --- Smoking & alcohol: majority vote across all records dated ON OR BEFORE index ---
        if feature_name in ONBEFORE_MAJORITY_FEATURES:
            on_before = [r["prediction"] for r in feature_records
                         if pd.notna(r["_parsed_date"]) and r["_parsed_date"] <= index_date
                         and r["prediction"] not in (None, "")]
            if on_before:
                result[f"{feature_name}__pre_index__pooled"] = pool_majority(on_before)

        # --- Disease: one column per DISTINCT disease, presence only ---
        # Group by the disease name, NOT the keyword: structured disease rows all share
        # keyword "STRUCTURED_DATA", so grouping by keyword collapsed every disease a patient
        # has into a single column (the first one). Keying on the disease name gives one
        # column per distinct disease, so multi-comorbidity patients are preserved.
        elif feature_name == "disease":
            by_period_dx = _group_by_period_and_keyword(
                feature_records,
                rec_key_fn=lambda r: r.get("disease", r.get("Code", "unknown")),
            )
            for period, by_dx in by_period_dx.items():
                for dx_name, dx_recs in by_dx.items():
                    dn = str(dx_name).replace(",", ";").replace(" ", "_")
                    result[f"disease__{period}__{dn}"] = "Yes"

        # --- Boolean per-keyword features: any "Yes" -> "Yes" ---
        elif feature_name in BOOLEAN_PER_KEYWORD_FEATURES:
            rec_key_fn = abx_record_to_class if feature_name == "antibiotics" else None
            for period, by_kw in _group_by_period_and_keyword(feature_records, rec_key_fn=rec_key_fn).items():
                for kw, kw_recs in by_kw.items():
                    preds = [r["prediction"] for r in kw_recs]
                    col_name = f"{feature_name}__{period}__{kw}".replace(" ", "_")
                    result[col_name] = pool_any_yes(preds)
                    # Add medication descriptions for antibiotics
                    # if feature_name == "antibiotics":
                    #     meds = sorted({r["Medication_Description"] for r in kw_recs if r.get("Medication_Description")})
                    #     if meds:
                    #         result[f"antibiotics_meds__{period}__{kw}".replace(" ", "_")] = "; ".join(meds)

        # --- Cancer occurrence: report by Diagnosis description (col D), splitting each ---
        # --- diagnosis into preexisting vs outcome by its EARLIEST diagnosis date, so ---
        # --- prevalent cancers don't count as incident outcomes. ---
        elif feature_name == "cancer_cancer":
            by_dx = defaultdict(list)
            for rec in feature_records:
                by_dx[icd_to_diagnosis(rec["icd_code"])].append(rec)
            for dx, dx_recs in by_dx.items():
                preds = [r["prediction"] for r in dx_recs]
                # Skip diagnoses where no record says "Yes"
                # (filters out noise columns from non-cancer ICD codes in existing data)
                if "Yes" not in preds:
                    continue
                earliest_dx = min(r["_parsed_date"] for r in dx_recs)
                cat = classify_cancer_period(earliest_dx, outcome_window_start)  # CancerPeriod
                dx_col = dx.replace(",", ";").replace(" ", "_")  # diagnoses contain commas/spaces; keep CSV columns clean
                result[f"cancer_{cat}__{dx_col}"] = pool_any_yes(preds)

        # --- Stage features: highest stage per keyword ---
        elif feature_name in STAGE_FEATURES:
            for period, by_kw in _group_by_period_and_keyword(feature_records).items():
                for kw, kw_recs in by_kw.items():
                    preds = [r["prediction"] for r in kw_recs]
                    col_name = f"{feature_name}__{period}__{kw}".replace(" ", "_")
                    result[col_name] = pool_highest_stage(preds)

        # --- Cancer date of diagnosis: group by Diagnosis description and split into ---
        # --- preexisting/outcome by earliest dx date, exactly like cancer_cancer, so each ---
        # --- date column aligns 1:1 with its cancer_preexisting / cancer_outcome column. ---
        elif feature_name in DATE_FEATURES:
            by_dx = defaultdict(list)
            for rec in feature_records:
                by_dx[icd_to_diagnosis(rec["icd_code"])].append(rec)
            for dx, dx_recs in by_dx.items():
                preds = [r["prediction"] for r in dx_recs]
                earliest_dx = min(r["_parsed_date"] for r in dx_recs)
                cat = classify_cancer_period(earliest_dx, outcome_window_start)  # CancerPeriod
                dx_col = dx.replace(",", ";").replace(" ", "_")
                result[f"{feature_name}__{cat}__{dx_col}"] = pool_earliest_date(preds)

        # --- Antibiotic duration numeric: sum of distinct durations per class (value-dedup) ---
        elif feature_name == "antibiotic_duration_numeric":
            for period, by_kw in _group_by_period_and_keyword(feature_records, rec_key_fn=abx_record_to_class).items():
                for kw, kw_recs in by_kw.items():
                    col_name = f"antibiotic_duration_numeric__{period}__{kw}".replace(" ", "_")
                    result[col_name] = pool_abx_duration_numeric(kw_recs)

        # --- Demographics: one column per keyword, take the value directly ---
        elif feature_name == "demographics":
            for period, by_kw in _group_by_period_and_keyword(feature_records).items():
                for kw, kw_recs in by_kw.items():
                    col_name = f"demographics__{kw}".replace(" ", "_")
                    result[col_name] = kw_recs[0]["prediction"]

        # --- Contraceptives: boolean presence (any record -> "Yes") ---
        elif feature_name == "contraceptives":
            for period, period_recs in _group_by_period(feature_records).items():
                col_name = f"contraceptives__{period}__pooled"
                result[col_name] = "Yes"
                # meds = sorted({r["Medication_Description"] for r in period_recs if r.get("Medication_Description")})
                # if meds:
                #     result[f"contraceptives_meds__{period}__pooled"] = "; ".join(meds)

        elif feature_name == "index_date":
            result["index_date"] = str(index_date.date())

        # --- Follow-up: per-patient single values computed in full_inference.py ---
        # (followup_start, followup_end, end_reason, last_activity_date, death_date,
        # cancer_date, follow_up_time_years). Passed through as one column each.
        elif feature_name == "follow_up":
            for rec in feature_records:
                result[rec["keyword"]] = rec["prediction"]

        else:
            print(f"WARNING: Unknown feature_name '{feature_name}', skipping")

    # Earliest abx date across all antibiotic records
    if "antibiotics" in by_feature:
        abx_dates = [r["_parsed_date"] for r in by_feature["antibiotics"]]
        if abx_dates:
            result["earliest_abx_date"] = str(min(abx_dates).date())

    # --- Preexisting condition categories (derived from the per-type columns above) ---
    # Split preexisting cancers into leukemia/lymphoma (its own category) vs other, and
    # flag preexisting HIV. Leukemia/lymphoma is identified from col D of the grouped ICD
    # table (LEUK_LYMPH_DX_COLS) and is NOT counted among "other" preexisting cancers. HIV
    # comes from the pre-index HIV disease columns.
    pre_leuk_lymph = "No"
    pre_other_cancer = "No"
    pre_hiv = "No"
    for col, val in result.items():
        if val != "Yes":
            continue
        if col.startswith("cancer_preexisting__"):
            dx_col = col[len("cancer_preexisting__"):]
            if dx_col in LEUK_LYMPH_DX_COLS:
                pre_leuk_lymph = "Yes"
            else:
                pre_other_cancer = "Yes"
        elif col.startswith("disease__pre_index__") and "human_immunodeficiency" in col.lower():
            pre_hiv = "Yes"
    result["cancer_preexisting_group__leukemia_lymphoma"] = pre_leuk_lymph
    result["cancer_preexisting_group__other"] = pre_other_cancer
    result["preexisting__hiv"] = pre_hiv

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

    # Step 4: Build DataFrame
    df = pd.DataFrame(rows)
    df = df.set_index("EMPI")

    # Follow-up time + window components (followup_start/end, end_reason, last_activity_date,
    # death_date, cancer_date, follow_up_time_years) are now computed per patient in
    # full_inference.py and pooled in above as the follow_up__* records -> plain columns here.
    # This replaces the old single pass over the full ~254GB source DB.
    if "end_reason" in df.columns:
        print("Follow-up end reason counts:")
        print(df["end_reason"].value_counts())

    df = df.reindex(sorted(df.columns), axis=1)
    df.to_csv(OUTPUT_PATH)
    print(f"Wrote {len(df)} patients x {len(df.columns)} columns to {OUTPUT_PATH}")

    # --- Cancer OUTCOME rate by abx status (cancers in the outcome window) ---
    abx_cols = [c for c in df.columns if c.startswith("antibiotics__")]
    cancer_cols = [c for c in df.columns if c.startswith("cancer_outcome__")]
    has_abx = (df[abx_cols] == "Yes").any(axis=1) if abx_cols else pd.Series(False, index=df.index)
    has_cancer = (df[cancer_cols] == "Yes").any(axis=1) if cancer_cols else pd.Series(False, index=df.index)

    n_abx = has_abx.sum()
    n_no_abx = (~has_abx).sum()
    cancer_abx = (has_abx & has_cancer).sum()
    cancer_no_abx = (~has_abx & has_cancer).sum()
    pct_abx = 100 * cancer_abx / n_abx if n_abx else 0
    pct_no_abx = 100 * cancer_no_abx / n_no_abx if n_no_abx else 0
    print(f"\nCancer outcome rate by abx status:")
    print(f"  Abx patients:     {cancer_abx}/{n_abx} ({pct_abx:.1f}%) have a cancer outcome")
    print(f"  Non-abx patients: {cancer_no_abx}/{n_no_abx} ({pct_no_abx:.1f}%) have a cancer outcome")

    # # Generate pandas profiling report
    # profile_path = OUTPUT_DIR / "pooled_records_profile.html"
    # print(f"Generating profile report...")
    # profile = ProfileReport(df, title="Pooled Records Profile")
    # profile.to_file(profile_path)
    # print(f"Wrote profile report to {profile_path}")


if __name__ == "__main__":
    main()
