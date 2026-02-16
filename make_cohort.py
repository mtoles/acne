import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from make_db import db_url
from pt_features import get_rows_by_icd, cancer_icd9s, cancer_icd10s, smoking_status, alcohol_status
import json
from pathlib import Path
import numpy as np
import argparse
from itertools import combinations

# Parse command line arguments
parser = argparse.ArgumentParser(description="Create matched case-control cohort for antibiotic study")
parser.add_argument("--n-limit", type=int, default=None, help="Limit to first N patients (N/2 cases, N/2 controls). Default: None (use all)")
parser.add_argument("--min-match-features", type=int, default=6, help="Minimum number of matching features (out of 6: sex, race, BMI, smoking, alcohol, age±5). Default: 6 (exact match on all)")
args = parser.parse_args()

N_LIMIT = args.n_limit  # use first N/2 from each cohort for quick test; set to None for full run
BATCH_SIZE = 1000  # For SQL queries to avoid memory issues


def get_cancer_dx_date(df):
    """return earliest data of cancer diagnosis, or None if no cancer diagnosis"""
    cancer_df = get_rows_by_icd(df, cancer_icd9s, cancer_icd10s)
    if cancer_df.empty:
        return None
    return pd.to_datetime(cancer_df["Date"], format="%m/%d/%Y").min()


def batch_sql_query(conn, empis, table_name, batch_size=BATCH_SIZE):
    """Query database in batches to avoid memory issues with large IN clauses"""
    all_results = []
    empis_list = list(empis)

    for i in tqdm(range(0, len(empis_list), batch_size), desc=f"Querying {table_name} in batches"):
        batch = empis_list[i:i + batch_size]
        empis_str = ",".join([f"'{str(e).replace(chr(39), chr(39)+chr(39))}'" for e in batch])
        query = text(f"SELECT * FROM {table_name} WHERE EMPI IN ({empis_str})")
        result = conn.execute(query)
        batch_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        all_results.append(batch_df)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


# Connect to database
print("Connecting to database...")
db = create_engine(db_url)

print("Loading antibiotic medication list...")
med_list_df = pd.read_csv("labeled_data/abx_med_code_list_v2.csv")
# med_list = med_list_df[med_list_df["a"] == 1]["MedicationID"]
med_list = med_list_df[med_list_df["include_consensus"] == True][["Medication", "Code_Type", "Code"]]
abx_pairs = med_list[["Code_Type", "Code"]].drop_duplicates()
abx_filter_sql = " OR ".join(
    f"(Code_Type = '{row['Code_Type']}' AND Code = '{row['Code']}')"
    for _, row in abx_pairs.iterrows()
)

# Step 1: Get all patients with acne diagnosis and their earliest acne date (= index date)
print("Querying acne diagnosis dates...")
with db.connect() as conn:
    query = text("""
        SELECT EMPI, MIN(Date) as acne_date FROM dia
        WHERE Code LIKE '%L70.0%' OR Code LIKE '%L70.8%' OR Code LIKE '%L70.9%'
            OR Code LIKE '%L70.1%' OR Code LIKE '%706.1%' OR Code LIKE '%acne%'
        GROUP BY EMPI
    """)
    result = conn.execute(query)
    acne_dates_df = pd.DataFrame(result.fetchall(), columns=result.keys())

acne_dates_df["acne_date"] = pd.to_datetime(acne_dates_df["acne_date"])
print(f"Found {len(acne_dates_df)} patients with acne diagnosis")

# Step 2: Get ABX medication records for acne patients
print("Querying antibiotic records for acne patients...")
acne_empis = acne_dates_df["EMPI"].tolist()
abx_med_batches = []
with db.connect() as conn:
    for i in tqdm(range(0, len(acne_empis), BATCH_SIZE), desc="Querying ABX med records"):
        batch = acne_empis[i:i + BATCH_SIZE]
        empis_str = ",".join([f"'{str(e).replace(chr(39), chr(39)+chr(39))}'" for e in batch])
        query = text(f"SELECT EMPI, Medication_Date FROM med WHERE EMPI IN ({empis_str}) AND ({abx_filter_sql})")
        result = conn.execute(query)
        batch_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        abx_med_batches.append(batch_df)

abx_med_df = pd.concat(abx_med_batches, ignore_index=True) if abx_med_batches else pd.DataFrame(columns=["EMPI", "Medication_Date"])
abx_med_df["Medication_Date"] = pd.to_datetime(abx_med_df["Medication_Date"])

# Step 3: Determine took_abx = had ABX within [acne_date, acne_date + 1 year)
abx_with_acne = abx_med_df.merge(acne_dates_df, on="EMPI")
abx_with_acne["in_window"] = (
    (abx_with_acne["Medication_Date"] >= abx_with_acne["acne_date"]) &
    (abx_with_acne["Medication_Date"] < abx_with_acne["acne_date"] + pd.Timedelta(days=365))
)

took_abx_in_window = set(abx_with_acne[abx_with_acne["in_window"]]["EMPI"].unique())
took_abx_any_time = set(abx_med_df["EMPI"].unique())
took_abx_outside_window_only = took_abx_any_time - took_abx_in_window

all_acne_empis = set(acne_dates_df["EMPI"].unique())
took_abx_empis = np.array(sorted(took_abx_in_window))
no_abx_empis = np.array(sorted(all_acne_empis - took_abx_in_window))

print(f"Patients with acne diagnosis: {len(all_acne_empis)}")
print(f"  Took ABX within treatment window (cases): {len(took_abx_empis)}")
print(f"  Did not take ABX within treatment window (controls): {len(no_abx_empis)}")
print(f"  Had ABX but only outside treatment window (reclassified as control): {len(took_abx_outside_window_only)} ({len(took_abx_outside_window_only)/len(all_acne_empis)*100:.1f}%)")

# Downsample BEFORE querying dem/phy tables
if N_LIMIT is not None:
    n_each = N_LIMIT // 2
    took_abx_empis = took_abx_empis[:n_each]
    no_abx_empis = no_abx_empis[:n_each]
    print(f"Limited to {n_each} cases and {n_each} controls ({N_LIMIT} pts total)")

# Index date = acne diagnosis date for all patients
acne_date_map = acne_dates_df.set_index("EMPI")["acne_date"].to_dict()
index_date_map = {}
for empi in list(took_abx_empis) + list(no_abx_empis):
    index_date_map[empi] = acne_date_map[empi]

# Get demographic info for both cohorts using batched queries
print("Querying demographic data...")
with db.connect() as conn:
    took_abx_dem_df = batch_sql_query(conn, took_abx_empis, "dem")
    no_abx_dem_df = batch_sql_query(conn, no_abx_empis, "dem")

# Stack the dataframes with differentiation column
took_abx_dem_df["took_abx"] = True
no_abx_dem_df["took_abx"] = False
dem_df = pd.concat([took_abx_dem_df, no_abx_dem_df], ignore_index=True)

print("Processing demographic data...")
dem_df["study_age"] = dem_df["Age"].astype(int)

# Bin age into 10-year categories for categorical matching
def categorize_age(age):
    if pd.isna(age):
        return None
    age_int = int(age)
    # Create bins: 0-9, 10-19, 20-29, ..., 90-99, 100+
    if age_int < 10:
        return "0-9"
    elif age_int >= 100:
        return "100+"
    else:
        decade = (age_int // 10) * 10
        return f"{decade}-{decade+9}"

dem_df["study_age_bin"] = dem_df["study_age"].apply(categorize_age)

dem_df["study_sex"] = dem_df["Sex_At_Birth"]
mask = ~dem_df["study_sex"].astype(str).str.lower().isin(["male", "female"])
dem_df.loc[mask, "study_sex"] = dem_df.loc[mask, "Gender_Legal_Sex"]
dem_df["study_sex"] = dem_df["study_sex"].astype(str).str.lower().str.strip()
dem_df["study_race"] = dem_df["Race_Group"]
mask = dem_df["Ethnic_Group"] == "HISPANIC"
dem_df.loc[mask, "study_race"] = "Hispanic"
dem_df["study_race"] = dem_df["study_race"].apply(lambda x: str(x).lower() if pd.notna(x) else "")

# Query physical health data using batched queries
print("Querying physical health data...")
with db.connect() as conn:
    all_empis = list(took_abx_empis) + list(no_abx_empis)
    phy_df = batch_sql_query(conn, all_empis, "phy")

# Parse dates and filter to records within 1 year before index date
print("Filtering phy records to closest to index date...")
phy_df["Date"] = pd.to_datetime(phy_df["Date"], format="%m/%d/%Y", errors="coerce")
phy_df["index_date"] = phy_df["EMPI"].map(index_date_map)

# Keep records within 1 year before index date
phy_df["days_before_index"] = (phy_df["index_date"] - phy_df["Date"]).dt.days
phy_df = phy_df[(phy_df["days_before_index"] >= 0) & (phy_df["days_before_index"] <= 365)]

print(f"Filtered phy records: {len(phy_df)} records within 1 year before index date")

# Extract BMI (use most recent measurement within the window)
print("Processing BMI data...")
bmi_df = phy_df[phy_df["Code"] == "BMI"].copy()
bmi_df["Result"] = pd.to_numeric(bmi_df["Result"], errors="coerce")
# Sort by days_before_index to get closest to index date (smallest value = closest)
bmi_df = bmi_df.sort_values("days_before_index")
bmi_by_empi = bmi_df.groupby("EMPI")["Result"].first()
dem_df["study_bmi"] = dem_df["EMPI"].map(bmi_by_empi)

# Categorize BMI into 4 categories
def categorize_bmi(bmi):
    if pd.isna(bmi):
        return None
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal weight"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"

dem_df["study_bmi_category"] = dem_df["study_bmi"].apply(categorize_bmi)


def smoking_status_from_structured_wrapper(phy_subset):
    """Wrapper for smoking_status.option_from_structured to work with grouped EMPI data."""
    results = {}
    for empi, group in phy_subset.groupby("EMPI"):
        records = group[["Concept_Name", "Result"]].to_dict("records")
        option = smoking_status.option_from_structured(records)
        # Map options to descriptive labels
        if option == "C":
            results[empi] = "Current Smoker"
        elif option == "B":
            results[empi] = "Former Smoker"
        elif option == "A":
            results[empi] = "Never Smoker"
        else:
            results[empi] = "Unknown"
    return pd.Series(results)


def alcohol_status_from_structured_wrapper(phy_subset):
    """Wrapper for alcohol_status.option_from_structured to work with grouped EMPI data."""
    results = {}
    for empi, group in phy_subset.groupby("EMPI"):
        records = group[["Concept_Name", "Result"]].to_dict("records")
        option = alcohol_status.option_from_structured(records)
        # Map options to descriptive labels
        if option == "A":
            results[empi] = "Currently Drinks"
        elif option == "B":
            results[empi] = "Does Not Currently Drink"
        else:
            results[empi] = "Unknown"
    return pd.Series(results)


# Use structured data detection via option_from_structured wrapper
print("Processing smoking status...")
smoking_by_empi = smoking_status_from_structured_wrapper(phy_df)
dem_df["study_smoking_status"] = dem_df["EMPI"].map(smoking_by_empi).fillna("Unknown")

# Use structured data detection via option_from_structured wrapper
print("Processing alcohol status...")
alcohol_by_empi = alcohol_status_from_structured_wrapper(phy_df)
dem_df["study_alcohol_status"] = dem_df["EMPI"].map(alcohol_by_empi).fillna("Unknown")

# Free up memory
del phy_df, bmi_df


print(dem_df.head())

# Add index date to dem_df for reference
dem_df["index_date"] = dem_df["EMPI"].map(index_date_map)

dem_df = dem_df[["EMPI", "took_abx", "index_date", "study_age", "study_sex", "study_race", "study_bmi", "study_bmi_category", "study_smoking_status", "study_alcohol_status"]]

# Filter patients by visit note coverage requirements
# Require: at least one visit note >5 years after index date AND >6 months before index date
print("\nFiltering patients by visit note coverage...")
all_empis_for_filter = dem_df["EMPI"].tolist()
vis_date_batches = []
with db.connect() as conn:
    for i in tqdm(range(0, len(all_empis_for_filter), BATCH_SIZE), desc="Querying vis date ranges"):
        batch = all_empis_for_filter[i:i + BATCH_SIZE]
        empis_str = ",".join([f"'{str(e).replace(chr(39), chr(39)+chr(39))}'" for e in batch])
        query = text(f"SELECT EMPI, MIN(Report_Date_Time) as earliest_visit, MAX(Report_Date_Time) as latest_visit FROM vis WHERE EMPI IN ({empis_str}) GROUP BY EMPI")
        result = conn.execute(query)
        batch_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        vis_date_batches.append(batch_df)

vis_dates_df = pd.concat(vis_date_batches, ignore_index=True) if vis_date_batches else pd.DataFrame()
vis_dates_df["earliest_visit"] = pd.to_datetime(vis_dates_df["earliest_visit"])
vis_dates_df["latest_visit"] = pd.to_datetime(vis_dates_df["latest_visit"])
vis_dates_df["index_date"] = vis_dates_df["EMPI"].map(index_date_map)

# At least one visit note more than 5 years after index date
vis_dates_df["has_late_visit"] = vis_dates_df["latest_visit"] > (vis_dates_df["index_date"] + pd.Timedelta(days=5 * 365))
# At least one visit note more than 6 months before index date
# vis_dates_df["has_early_visit"] = vis_dates_df["earliest_visit"] < (vis_dates_df["index_date"] - pd.Timedelta(days=183))

qualifying_empis = set(vis_dates_df[vis_dates_df["has_late_visit"]]["EMPI"])

pre_filter_count = len(dem_df)
dem_df = dem_df[dem_df["EMPI"].isin(qualifying_empis)].reset_index(drop=True)
print(f"Visit coverage filter: {pre_filter_count} -> {len(dem_df)} patients")
print(f"Patients who took ABX: {(dem_df['took_abx'] == True).sum()}")
print(f"Patients who did not take ABX: {(dem_df['took_abx'] == False).sum()}")

# Separate cases and controls
print("\nPreparing for case-control matching...")
cases_df = dem_df[dem_df["took_abx"] == True].copy()
controls_df = dem_df[dem_df["took_abx"] == False].copy()

# Sort cases for consistent matching order
cases_df = cases_df.sort_values("EMPI").reset_index(drop=True)
controls_df = controls_df.reset_index(drop=True)

# Matching features: 5 categorical + age (±5 years)
MATCH_CATEGORICAL_FEATURES = ["study_sex", "study_race", "study_bmi_category", "study_smoking_status", "study_alcohol_status"]

def build_key_from_features(row, feature_list):
    return "_".join(str(row[f]) for f in feature_list)

matches_list = []
used_control_indices = set()
total_cases = len(cases_df)

print(f"\nMatching cases to controls (min features: {args.min_match_features}/6)...")

for n_required in range(6, args.min_match_features - 1, -1):
    matched_case_empis_set = {pair[0] for pair in matches_list}
    unmatched_cases = cases_df[~cases_df["EMPI"].isin(matched_case_empis_set)]

    if len(unmatched_cases) == 0:
        print(f"  Round {n_required}/6: all cases already matched, skipping")
        break

    round_matches = 0

    # Generate all feature subsets for this round
    # Each subset is (list_of_categorical_features, check_age_bool)
    subsets = []
    for include_age in [True, False]:
        n_cat = n_required - (1 if include_age else 0)
        if n_cat < 0 or n_cat > len(MATCH_CATEGORICAL_FEATURES):
            continue
        for cat_subset in combinations(MATCH_CATEGORICAL_FEATURES, n_cat):
            subsets.append((list(cat_subset), include_age))

    for cat_features, check_age in subsets:
        # Refresh unmatched cases (may have changed from previous subset)
        matched_case_empis_set = {pair[0] for pair in matches_list}
        unmatched_cases = cases_df[~cases_df["EMPI"].isin(matched_case_empis_set)]

        if len(unmatched_cases) == 0:
            break

        # Build control index for this feature subset (only available controls)
        controls_by_key = {}
        for idx, row in controls_df.iterrows():
            if idx in used_control_indices:
                continue
            key = build_key_from_features(row, cat_features)
            if key not in controls_by_key:
                controls_by_key[key] = []
            controls_by_key[key].append(idx)

        for case_idx, case_row in unmatched_cases.iterrows():
            case_key = build_key_from_features(case_row, cat_features)
            if case_key not in controls_by_key:
                continue
            candidate_indices = controls_by_key[case_key]

            for ctrl_idx in candidate_indices:
                if ctrl_idx in used_control_indices:
                    continue

                if check_age:
                    if abs(controls_df.iloc[ctrl_idx]["study_age"] - case_row["study_age"]) > 5:
                        continue

                matches_list.append((case_row["EMPI"], controls_df.iloc[ctrl_idx]["EMPI"]))
                used_control_indices.add(ctrl_idx)
                round_matches += 1
                break

    total_matched = len(matches_list)
    print(f"  Round {n_required}/6: {round_matches} new matches, {total_matched}/{total_cases} total ({total_matched/total_cases*100:.1f}%)")

matched_count = len(matches_list)
unmatched_count = total_cases - matched_count
unmatched_percentage = (unmatched_count / total_cases) * 100 if total_cases > 0 else 0

print(f"\nMatched {matched_count} out of {total_cases} cases ({100 - unmatched_percentage:.2f}%)")
print(f"Unmatched: {unmatched_count} cases ({unmatched_percentage:.2f}%)")

# Save EMPIs to JSON format in cohort directory
print("\nSaving results...")
cohort_dir = Path("cohort")
cohort_dir.mkdir(exist_ok=True)

# Get matched EMPIs (both cases and controls from matches)
matched_case_empis = [pair[0] for pair in matches_list]
matched_control_empis = [pair[1] for pair in matches_list]

# Create JSON structure
empis_data = {
    "all_empis": sorted([str(e) for e in dem_df["EMPI"].tolist()]),
    "case_empis": sorted([str(e) for e in cases_df["EMPI"].tolist()]),
    "control_empis": sorted([str(e) for e in controls_df["EMPI"].tolist()]),
    "matched_case_empis": sorted([str(e) for e in matched_case_empis]),
    "matched_control_empis": sorted([str(e) for e in matched_control_empis]),
    "matched_pairs": [[str(pair[0]), str(pair[1])] for pair in matches_list],
    "stats": {
        "total_cases": int(total_cases),
        "total_controls": len(controls_df),
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "match_percentage": float(100 - unmatched_percentage)
    }
}

# Save to JSON file
json_path = cohort_dir / "empis.json"
with open(json_path, "w") as f:
    json.dump(empis_data, f, indent=2)

print(f"\nSaved EMPIs to {json_path}")
print(f"  Total EMPIs: {len(empis_data['all_empis'])}")
print(f"  Case EMPIs: {len(empis_data['case_empis'])}")
print(f"  Control EMPIs: {len(empis_data['control_empis'])}")
print(f"  Matched pairs: {len(matches_list)}")

# Save match statistics to text file
txt_path = cohort_dir / "match_stats.txt"
with open(txt_path, "w") as f:
    f.write(f"Total matched: {matched_count}\n")
    f.write(f"Match percentage: {100 - unmatched_percentage:.2f}%\n")
    f.write(f"Total cases: {total_cases}\n")
    f.write(f"Unmatched: {unmatched_count} ({unmatched_percentage:.2f}%)\n")

print(f"Saved match statistics to {txt_path}")

# Create pie charts for each categorical variable
print("\nCreating pie charts for cohort variables...")
import matplotlib.pyplot as plt

tmp_dir = Path("tmp")
tmp_dir.mkdir(exist_ok=True)

# Get matched patients only
matched_empis = list(set(matched_case_empis)) + list(set(matched_control_empis)) 
matched_dem_df = dem_df[dem_df["EMPI"].isin(matched_empis)]

# Variables to plot
variables = {
    "took_abx": "Antibiotic Use",
    "study_sex": "Sex",
    "study_race": "Race/Ethnicity",
    "study_bmi_category": "BMI Category",
    "study_smoking_status": "Smoking Status",
    "study_alcohol_status": "Alcohol Status"
}

for var_name, var_label in variables.items():
    counts = matched_dem_df[var_name].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"{var_label} Distribution in Matched Cohort (N={len(matched_dem_df)})")
    plt.axis('equal')

    output_path = tmp_dir / f"{var_name}_pie_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")

print("\nDone!")
