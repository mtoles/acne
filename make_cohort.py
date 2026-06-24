import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from make_db import db_url
from pt_features import get_rows_by_icd, cancer_icd9s, cancer_icd10s, compute_sex, compute_race, categorize_age, categorize_bmi, compute_bmi_from_phy, compute_smoking_status_demographic, compute_alcohol_status_demographic, COVARIATE_MAX_DAYS_AFTER_INDEX
import json
from pathlib import Path
import numpy as np
import argparse
from itertools import combinations
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import linear_sum_assignment

# Parse command line arguments
parser = argparse.ArgumentParser(description="Create matched case-control cohort for antibiotic study")
parser.add_argument("--n-limit", type=int, default=None, help="Limit to first N patients (N/2 cases, N/2 controls). Default: None (use all)")
parser.add_argument("--min-match-features", type=int, default=5, help="Minimum number of matching features (out of 6: sex, race, BMI, smoking, alcohol, age±5). Default: 6 (exact match on all)")
parser.add_argument("--min-age", type=float, default=12, help="Minimum age (inclusive) at index date. Default: 12")
parser.add_argument("--max-age", type=float, default=50, help="Maximum age (inclusive) at index date. Default: 45")
parser.add_argument("--n-jobs", type=int, default=8, help="Parallel workers for per-patient phy feature extraction. 1 = serial. Default: 8")
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
med_list = med_list_df[med_list_df["include_final"] == True][["Medication", "Code_Type", "Code"]]
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

# Step 3: Determine took_abx = had ABX within [acne_date, acne_date + 2 years)
abx_with_acne = abx_med_df.merge(acne_dates_df, on="EMPI")
abx_with_acne["in_window"] = (
    (abx_with_acne["Medication_Date"] >= abx_with_acne["acne_date"]) &
    (abx_with_acne["Medication_Date"] < abx_with_acne["acne_date"] + pd.Timedelta(days= 365 * 2))
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
dem_df["Date_of_Birth"] = pd.to_datetime(dem_df["Date_of_Birth"], errors="coerce")
dem_df["index_date"] = dem_df["EMPI"].map(index_date_map)
dem_df["study_age"] = (dem_df["index_date"] - dem_df["Date_of_Birth"]).dt.days / 365.25

# Filter by age at index date
pre_age_filter = len(dem_df)
in_range = dem_df["study_age"].between(args.min_age, args.max_age, inclusive="left")
dem_df = dem_df[in_range].reset_index(drop=True)
print(
    f"Age filter [{args.min_age}, {args.max_age}]: {pre_age_filter} -> {len(dem_df)} patients "
    f"(cases: {(dem_df['took_abx']==True).sum()}, controls: {(dem_df['took_abx']==False).sum()})"
)
index_date_map = {empi: index_date_map[empi] for empi in dem_df["EMPI"].tolist()}

dem_df["study_age_bin"] = dem_df["study_age"].apply(categorize_age)
dem_df["study_sex"] = dem_df.apply(compute_sex, axis=1)
dem_df["study_race"] = dem_df.apply(compute_race, axis=1)

# Query physical health data using batched queries
print("Querying physical health data...")
with db.connect() as conn:
    all_empis = dem_df["EMPI"].tolist()
    phy_df = batch_sql_query(conn, all_empis, "phy")

# Process BMI, smoking, alcohol per patient.
# Parse phy dates ONCE here (vectorized) instead of re-parsing inside each helper:
# format="mixed" infers per-element and previously ran 3x per patient group.
print("Processing BMI, smoking, and alcohol status...")
phy_df["Date"] = pd.to_datetime(phy_df["Date"], format="mixed", errors="coerce")


def _compute_phy_features(empi, group, idx_date):
    bmi_val, bmi_cat = compute_bmi_from_phy(group, idx_date, max_days_after=COVARIATE_MAX_DAYS_AFTER_INDEX)
    smoking = compute_smoking_status_demographic(group, idx_date)
    alcohol = compute_alcohol_status_demographic(group, idx_date)
    return empi, bmi_val, bmi_cat, smoking, alcohol


groups = phy_df.groupby("EMPI")
if args.n_jobs == 1:
    results = [_compute_phy_features(empi, group, index_date_map[empi]) for empi, group in tqdm(groups)]
else:
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_compute_phy_features)(empi, group, index_date_map[empi])
        for empi, group in tqdm(groups)
    )

bmi_by_empi = {}
bmi_cat_by_empi = {}
smoking_by_empi = {}
alcohol_by_empi = {}
for empi, bmi_val, bmi_cat, smoking, alcohol in results:
    bmi_by_empi[empi] = bmi_val
    bmi_cat_by_empi[empi] = bmi_cat
    smoking_by_empi[empi] = smoking
    alcohol_by_empi[empi] = alcohol

dem_df["study_bmi"] = dem_df["EMPI"].map(bmi_by_empi)
dem_df["study_bmi_category"] = dem_df["EMPI"].map(bmi_cat_by_empi)
dem_df["study_smoking_status"] = dem_df["EMPI"].map(smoking_by_empi).fillna("Unknown")
dem_df["study_alcohol_status"] = dem_df["EMPI"].map(alcohol_by_empi).fillna("Unknown")

# Free up memory
del phy_df


print(dem_df.head())

dem_df = dem_df[["EMPI", "took_abx", "index_date", "study_age", "study_sex", "study_race", "study_bmi", "study_bmi_category", "study_smoking_status", "study_alcohol_status"]]

# Filter patients by note coverage: require at least one note >5 years after index date.
# Use ALL free-text note types that full inference reads (vis + hnp + prg + dis + lno),
# not just vis, so coverage reflects the data full_inference.py actually uses.
# Keep in sync with EXTRA_NOTE_TABLES in full_inference.py.
#   iso_sortable=True  -> 'YYYY-MM-DD ...', so SQL MAX() is chronologically correct.
#   iso_sortable=False -> 'M/D/YYYY ...' (lno), so we must parse before taking the max.
# date_format: explicit parse format (None = let pandas infer the ISO format, fast).
NOTE_DATE_TABLES = [
    ("vis", "Report_Date_Time", True,  None),
    ("hnp", "Report_Date_Time", True,  None),
    ("prg", "Report_Date_Time", True,  None),
    ("dis", "Report_Date_Time", True,  None),
    ("lno", "LMRNote_Date",     False, "%m/%d/%Y %I:%M:%S %p"),
]

print("\nFiltering patients by note coverage (vis + hnp + prg + dis + lno)...")
all_empis_for_filter = dem_df["EMPI"].tolist()


def latest_note_date_by_empi(conn, empis, table, datecol, iso_sortable, date_format):
    """Return a Series indexed by EMPI of the latest parsed note date in `table`.
    For ISO-sortable date strings SQL MAX() is used; otherwise all rows are pulled
    and the max is taken after parsing (string MAX would be chronologically wrong)."""
    parts = []
    for i in tqdm(range(0, len(empis), BATCH_SIZE), desc=f"Querying {table} dates"):
        batch = empis[i:i + BATCH_SIZE]
        empis_str = ",".join([f"'{str(e).replace(chr(39), chr(39)+chr(39))}'" for e in batch])
        if iso_sortable:
            query = text(f"SELECT EMPI, MAX({datecol}) AS latest FROM {table} WHERE EMPI IN ({empis_str}) GROUP BY EMPI")
        else:
            query = text(f"SELECT EMPI, {datecol} AS latest FROM {table} WHERE EMPI IN ({empis_str})")
        parts.append(pd.DataFrame(conn.execute(query).fetchall(), columns=["EMPI", "latest"]))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["EMPI", "latest"])
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")
    df["latest"] = pd.to_datetime(df["latest"], format=date_format, errors="coerce")
    return df.dropna(subset=["latest"]).groupby("EMPI")["latest"].max()


with db.connect() as conn:
    per_table_latest = [
        latest_note_date_by_empi(conn, all_empis_for_filter, tbl, col, iso, fmt)
        for tbl, col, iso, fmt in NOTE_DATE_TABLES
    ]

# Combine across note types: latest note date per EMPI over all tables.
latest_note_by_empi = pd.concat(per_table_latest).groupby(level=0).max()

vis_dates_df = pd.DataFrame({"EMPI": all_empis_for_filter})
vis_dates_df["latest_visit"] = vis_dates_df["EMPI"].map(latest_note_by_empi)
vis_dates_df["index_date"] = vis_dates_df["EMPI"].map(index_date_map)

# At least one note more than 5 years after index date
vis_dates_df["has_late_visit"] = vis_dates_df["latest_visit"] > (vis_dates_df["index_date"] + pd.Timedelta(days=5 * 365))

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

# Output directory for per-round matched-cohort distribution diagnostics
dist_dir = Path("tmp") / "match_distributions"
dist_dir.mkdir(parents=True, exist_ok=True)

# (column, human label) for the 5 categorical matching features
CAT_PLOT_FEATURES = [
    ("study_sex", "Sex"),
    ("study_race", "Race/Ethnicity"),
    ("study_bmi_category", "BMI Category"),
    ("study_smoking_status", "Smoking Status"),
    ("study_alcohol_status", "Alcohol Status"),
]
CASE_COLOR = "#d62728"
CONTROL_COLOR = "#1f77b4"


def smd_continuous(case_vals, ctrl_vals):
    """Standardized mean difference and Welch t-test p-value for a continuous variable."""
    c = np.asarray(case_vals, dtype=float)
    t = np.asarray(ctrl_vals, dtype=float)
    pooled_sd = np.sqrt((c.var(ddof=1) + t.var(ddof=1)) / 2)
    smd = (c.mean() - t.mean()) / pooled_sd if pooled_sd > 0 else 0.0
    p = stats.ttest_ind(c, t, equal_var=False).pvalue
    return smd, p


def smd_categorical(case_series, ctrl_series):
    """Multi-category standardized mean difference (Yang & Dalton 2012) and chi-square
    p-value of independence. Reduces to the usual binary SMD for 2-level variables."""
    cats = sorted(set(case_series.astype(str)) | set(ctrl_series.astype(str)))
    if len(cats) < 2:
        return 0.0, float("nan")
    cc = case_series.astype(str).value_counts().reindex(cats, fill_value=0)
    tc = ctrl_series.astype(str).value_counts().reindex(cats, fill_value=0)
    pc = (cc / cc.sum()).to_numpy()
    pt = (tc / tc.sum()).to_numpy()

    # Use k-1 categories (last is redundant) for the covariance form
    pc1, pt1 = pc[:-1], pt[:-1]

    def cov(p):
        s = -np.outer(p, p)
        s[np.diag_indices_from(s)] = p * (1 - p)
        return s

    S = (cov(pc1) + cov(pt1)) / 2
    diff = (pc1 - pt1).reshape(-1, 1)
    smd = float(np.sqrt(max(0.0, (diff.T @ np.linalg.pinv(S) @ diff).item())))

    table = np.vstack([cc.to_numpy(), tc.to_numpy()])
    table = table[:, table.sum(axis=0) > 0]  # drop categories absent in both
    p = stats.chi2_contingency(table).pvalue
    return smd, p


def plot_match_distributions(matches_list, n_required, cases_df, controls_df, out_dir, min_age, max_age):
    """Plot/save case-vs-control distributions of the 6 matching features for the
    currently matched cohort (after `n_required`/6 round). Writes a PNG with one
    subplot per feature and a markdown file with the same data in tabular form."""
    if len(matches_list) == 0:
        print(f"  Round {n_required}/6: no matches yet, skipping distribution plot")
        return

    matched_case_empis = {p[0] for p in matches_list}
    matched_control_empis = {p[1] for p in matches_list}
    mcases = cases_df[cases_df["EMPI"].isin(matched_case_empis)]
    mcontrols = controls_df[controls_df["EMPI"].isin(matched_control_empis)]
    n_pairs = len(matches_list)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    md_lines = [
        f"# Matched cohort feature distributions — round {n_required}/6",
        "",
        f"Allowing matches on {n_required} of 6 features. "
        f"Cumulative matched pairs: **{n_pairs}** (cases={len(mcases)}, controls={len(mcontrols)}).",
        "",
    ]

    # Balance summary: standardized mean difference (SMD) and p-value per feature.
    # |SMD| < 0.1 is the usual rule of thumb for good balance.
    balance_rows = []
    for col, label in CAT_PLOT_FEATURES:
        smd_v, p_v = smd_categorical(mcases[col], mcontrols[col])
        balance_rows.append((label, smd_v, p_v))
    age_smd, age_p = smd_continuous(mcases["study_age"], mcontrols["study_age"])
    balance_rows.append(("Age", age_smd, age_p))

    md_lines.append("## Balance summary (SMD & p-value)")
    md_lines.append("")
    md_lines.append("| Feature | SMD | p-value | balanced (|SMD|<0.1) |")
    md_lines.append("|---|---|---|---|")
    for label, smd_v, p_v in balance_rows:
        md_lines.append(f"| {label} | {smd_v:.4f} | {p_v:.4g} | {'yes' if abs(smd_v) < 0.1 else 'NO'} |")
    md_lines.append("")

    print(f"  Round {n_required}/6 balance (SMD | p-value):")
    for label, smd_v, p_v in balance_rows:
        flag = "" if abs(smd_v) < 0.1 else "  <-- imbalanced"
        print(f"      {label:18s} SMD={smd_v:+.4f}  p={p_v:.4g}{flag}")

    # 5 categorical features: grouped bar chart of category proportions
    for ax, (col, label) in zip(axes, CAT_PLOT_FEATURES):
        cats = sorted(set(mcases[col].astype(str)) | set(mcontrols[col].astype(str)))
        case_prop = mcases[col].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0)
        ctrl_prop = mcontrols[col].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0)
        x = np.arange(len(cats))
        w = 0.4
        ax.bar(x - w / 2, case_prop.values, w, label="Treatment", color=CASE_COLOR)
        ax.bar(x + w / 2, ctrl_prop.values, w, label="Controls", color=CONTROL_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylabel("Proportion")
        ax.set_title(label)
        ax.legend()

        md_lines.append(f"## {label}")
        md_lines.append("")
        md_lines.append("| Category | Treatment | Controls |")
        md_lines.append("|---|---|---|")
        for c in cats:
            md_lines.append(f"| {c} | {case_prop[c]:.3f} | {ctrl_prop[c]:.3f} |")
        md_lines.append("")

    # Age: continuous, so use an overlaid density histogram with shared bins
    ax = axes[5]
    bins = np.linspace(min_age, max_age, 16)
    ax.hist(mcases["study_age"], bins=bins, density=True, alpha=0.5, label="Treatment", color=CASE_COLOR)
    ax.hist(mcontrols["study_age"], bins=bins, density=True, alpha=0.5, label="Controls", color=CONTROL_COLOR)
    ax.set_xlabel("Age at index date")
    ax.set_ylabel("Density")
    ax.set_title("Age (±5y match)")
    ax.legend()

    md_lines.append("## Age (at index date)")
    md_lines.append("")
    md_lines.append("| Statistic | Treatment | Controls |")
    md_lines.append("|---|---|---|")
    for stat, fn in [("n", len), ("mean", lambda s: s.mean()), ("std", lambda s: s.std()),
                     ("median", lambda s: s.median()), ("min", lambda s: s.min()), ("max", lambda s: s.max())]:
        cv = fn(mcases["study_age"])
        tv = fn(mcontrols["study_age"])
        md_lines.append(f"| {stat} | {cv:.2f} | {tv:.2f} |")
    md_lines.append("")

    fig.suptitle(f"Matched cohort feature distributions — round {n_required}/6 (treatment vs controls, n_pairs={n_pairs})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = out_dir / f"round_{n_required}_distributions.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    md_path = out_dir / f"round_{n_required}_distributions.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"  Round {n_required}/6: saved distribution plot -> {png_path} and table -> {md_path}")


print(f"\nMatching cases to controls (min features: {args.min_match_features}/6)...")

# Precompute once: per-feature string columns (for fast vectorized key building) and
# numpy arrays (for O(1) scalar access instead of repeated .iloc[] lookups in the loop).
control_str = {f: controls_df[f].astype(str) for f in MATCH_CATEGORICAL_FEATURES}
case_str = {f: cases_df[f].astype(str) for f in MATCH_CATEGORICAL_FEATURES}
control_ages = controls_df["study_age"].to_numpy()
control_empis_arr = controls_df["EMPI"].to_numpy()
case_ages = cases_df["study_age"].to_numpy()
case_empis_arr = cases_df["EMPI"].to_numpy()

# Cost assigned to age-incompatible (|diff| > 5y) pairs in optimal assignment.
# Must dominate any feasible total cost (max feasible edge = 5y) so the solver
# maximizes the number of in-window matches before minimizing total age distance.
AGE_FORBIDDEN = 1e9


def _build_keys(str_cols, feature_list, index):
    """Vectorized '_'-joined key Series over feature_list (matches build_key_from_features)."""
    if not feature_list:
        return pd.Series("", index=index)
    keys = str_cols[feature_list[0]].copy()
    for f in feature_list[1:]:
        keys = keys + "_" + str_cols[f]
    return keys


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

        # Build control index for this feature subset (only available controls).
        # keys built vectorized; dict built in ascending-index order to preserve
        # the original candidate ordering.
        control_keys = _build_keys(control_str, cat_features, controls_df.index)
        controls_by_key = {}
        for idx, key in control_keys.items():
            if idx in used_control_indices:
                continue
            if key not in controls_by_key:
                controls_by_key[key] = []
            controls_by_key[key].append(idx)

        case_keys = _build_keys(case_str, cat_features, cases_df.index).to_numpy()
        matched_case_empis_set = {pair[0] for pair in matches_list}

        if check_age:
            # Optimal assignment per categorical-key group: minimize total age distance
            # subject to |age diff| <= 5y. Forbidden (>5y) pairs get a huge cost so the
            # solver first maximizes the number of in-window matches (order-independent,
            # unlike greedy), then minimizes total age distance among those.
            cases_by_key = {}
            for i in range(len(cases_df)):
                if case_empis_arr[i] in matched_case_empis_set:
                    continue
                k = case_keys[i]
                if k in controls_by_key:  # only keys with available controls can match
                    cases_by_key.setdefault(k, []).append(i)

            for key, case_idxs in cases_by_key.items():
                ctrl_idxs = controls_by_key[key]
                diff = np.abs(case_ages[case_idxs][:, None] - control_ages[ctrl_idxs][None, :])
                cost = np.where(diff <= 5, diff, AGE_FORBIDDEN)
                row_ind, col_ind = linear_sum_assignment(cost)
                for ri, ci in zip(row_ind, col_ind):
                    if diff[ri, ci] > 5:  # forbidden pair the solver was forced into; skip
                        continue
                    gi = case_idxs[ri]
                    gj = ctrl_idxs[ci]
                    matches_list.append((case_empis_arr[gi], control_empis_arr[gj]))
                    used_control_indices.add(gj)
                    round_matches += 1
        else:
            # No age constraint: every available control sharing the key is an equally
            # valid match, so greedy first-available is already optimal.
            for i in range(len(cases_df)):
                if case_empis_arr[i] in matched_case_empis_set:
                    continue
                case_key = case_keys[i]
                if case_key not in controls_by_key:
                    continue
                chosen_idx = None
                for ctrl_idx in controls_by_key[case_key]:
                    if ctrl_idx not in used_control_indices:
                        chosen_idx = ctrl_idx
                        break
                if chosen_idx is None:
                    continue
                matches_list.append((case_empis_arr[i], control_empis_arr[chosen_idx]))
                used_control_indices.add(chosen_idx)
                round_matches += 1

    total_matched = len(matches_list)
    print(f"  Round {n_required}/6: {round_matches} new matches, {total_matched}/{total_cases} total ({total_matched/total_cases*100:.1f}%)")

    # Plot/save how well the (cumulative) matched cohort is balanced across all 6 features
    plot_match_distributions(matches_list, n_required, cases_df, controls_df, dist_dir, args.min_age, args.max_age)

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
