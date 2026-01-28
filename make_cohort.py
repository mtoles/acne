import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from make_db import db_url
from pt_features import get_rows_by_icd, cancer_icd9s, cancer_icd10s, smoking_status, alcohol_status
import json
from pathlib import Path
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Create matched case-control cohort for antibiotic study")
parser.add_argument("--n-limit", type=int, default=None, help="Limit to first N patients (N/2 cases, N/2 controls). Default: None (use all)")
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
med_list_df = pd.read_csv("stores/abx_med_code_list.csv")
med_list = med_list_df[med_list_df["a"] == 1]["MedicationID"]
abx_list = list(med_list.unique())

# Get distinct EMPIs only, not all med records
print("Querying for patients who took antibiotics...")
with db.connect() as conn:
    # Get EMPIs who took ABX (medication IDs are in Code column)
    abx_list_str = ",".join([f"'{str(x)}'" for x in abx_list])
    query = text(f"SELECT DISTINCT EMPI FROM med WHERE Code IN ({abx_list_str})")
    result = conn.execute(query)
    took_abx_empis = pd.Series([row[0] for row in result.fetchall()]).unique()

    print("Querying for patients who did not take antibiotics...")
    # Get EMPIs who did not take ABX using LEFT JOIN (SQLite-compatible)
    query = text(f"""
        SELECT DISTINCT m.EMPI
        FROM med m
        LEFT JOIN (
            SELECT DISTINCT EMPI
            FROM med
            WHERE Code IN ({abx_list_str})
        ) abx ON m.EMPI = abx.EMPI
        WHERE abx.EMPI IS NULL
    """)
    result = conn.execute(query)
    no_abx_empis = pd.Series([row[0] for row in result.fetchall()]).unique()

print(f"Found {len(took_abx_empis)} patients who took ABX")
print(f"Found {len(no_abx_empis)} patients who did not take ABX")

# Downsample BEFORE querying dem/phy tables
if N_LIMIT is not None:
    n_each = N_LIMIT // 2
    took_abx_empis = took_abx_empis[:n_each]
    no_abx_empis = no_abx_empis[:n_each]
    print(f"Limited to {n_each} cases and {n_each} controls ({N_LIMIT} pts total)")

# Get index dates for cases (first antibiotic date)
print("Determining index dates for cases...")
with db.connect() as conn:
    abx_list_str = ",".join([f"'{str(x)}'" for x in abx_list])
    empis_str = ",".join([f"'{str(e).replace(chr(39), chr(39)+chr(39))}'" for e in took_abx_empis])
    query = text(f"""
        SELECT EMPI, MIN(Medication_Date) as index_date
        FROM med
        WHERE EMPI IN ({empis_str}) AND Code IN ({abx_list_str})
        GROUP BY EMPI
    """)
    result = conn.execute(query)
    index_dates_df = pd.DataFrame(result.fetchall(), columns=result.keys())

index_dates_df["index_date"] = pd.to_datetime(index_dates_df["index_date"])

# For controls, use the median index date from cases
median_index_date = index_dates_df["index_date"].median()
print(f"Median index date for cases: {median_index_date}")

# Create index date mapping for all patients
index_date_map = index_dates_df.set_index("EMPI")["index_date"].to_dict()
for empi in no_abx_empis:
    index_date_map[empi] = median_index_date

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

print(f"Smoking status values: {dem_df['study_smoking_status'].value_counts().to_dict()}")
print(f"Alcohol status values: {dem_df['study_alcohol_status'].value_counts().to_dict()}")
print(f"Patients with same status: {(dem_df['study_smoking_status'] == dem_df['study_alcohol_status']).sum()} / {len(dem_df)}")

print(dem_df.head())

print(f"Patients who took ABX: {len(took_abx_empis)}")
print(f"Patients who did not take ABX: {len(no_abx_empis)}")

# Add index date to dem_df for reference
dem_df["index_date"] = dem_df["EMPI"].map(index_date_map)

dem_df = dem_df[["EMPI", "took_abx", "index_date", "study_age", "study_sex", "study_race", "study_bmi", "study_bmi_category", "study_smoking_status", "study_alcohol_status"]]

# Separate cases and controls
print("\nPreparing for case-control matching...")
cases_df = dem_df[dem_df["took_abx"] == True].copy()
controls_df = dem_df[dem_df["took_abx"] == False].copy()

# Sort cases for consistent matching order
cases_df = cases_df.sort_values("EMPI").reset_index(drop=True)
controls_df = controls_df.reset_index(drop=True)

# Create index for faster lookups
print("Creating matching index...")
controls_df["original_idx"] = controls_df.index
used_control_indices = set()
matches_list = []

# More efficient matching using pre-computed control groups
# Group controls by characteristics for faster matching
print("Pre-indexing controls by matching criteria...")
controls_df["match_key"] = (
    controls_df["study_sex"].astype(str).str.lower() + "_" +
    controls_df["study_race"].astype(str) + "_" +
    controls_df["study_bmi_category"].astype(str) + "_" +
    controls_df["study_smoking_status"].astype(str) + "_" +
    controls_df["study_alcohol_status"].astype(str)
)

# Group controls by match key for faster lookup
controls_by_key = {}
for idx, row in controls_df.iterrows():
    key = row["match_key"]
    if key not in controls_by_key:
        controls_by_key[key] = []
    controls_by_key[key].append(idx)

# Efficient matching
print("Matching cases to controls...")
for case_idx, case_row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Matching cases"):
    # Build match key for this case
    case_key = (
        str(case_row["study_sex"]).lower() + "_" +
        str(case_row["study_race"]) + "_" +
        str(case_row["study_bmi_category"]) + "_" +
        str(case_row["study_smoking_status"]) + "_" +
        str(case_row["study_alcohol_status"])
    )

    # Get candidate controls with same key
    candidate_indices = controls_by_key.get(case_key, [])

    # Find first available control within age range
    for ctrl_idx in candidate_indices:
        if ctrl_idx in used_control_indices:
            continue

        control_row = controls_df.iloc[ctrl_idx]

        # Check age match
        if abs(control_row["study_age"] - case_row["study_age"]) <= 5:
            matches_list.append((case_row["EMPI"], control_row["EMPI"]))
            used_control_indices.add(ctrl_idx)
            break

# Print match statistics
matched_count = len(matches_list)
total_cases = len(cases_df)
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
    "all_empis": sorted([str(e) for e in set(took_abx_empis.tolist() + no_abx_empis.tolist())]),
    "case_empis": sorted([str(e) for e in took_abx_empis.tolist()]),
    "control_empis": sorted([str(e) for e in no_abx_empis.tolist()]),
    "matched_case_empis": sorted([str(e) for e in matched_case_empis]),
    "matched_control_empis": sorted([str(e) for e in matched_control_empis]),
    "matched_pairs": [[str(pair[0]), str(pair[1])] for pair in matches_list],
    "stats": {
        "total_cases": int(total_cases),
        "total_controls": len(no_abx_empis),
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
matched_empis = set(matched_case_empis + matched_control_empis)
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
