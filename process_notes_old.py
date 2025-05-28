import pandas as pd
import os
import time  # Add time module import

from tqdm import tqdm

from make_db import store_dir as file_store_parent_dir
from pt_features import PtFeaturesMeta, PtFeatureBase
from utils import chunk_text, has_keyword
from models import MrModel, DummyModel

# df cols
# 'chunk', 'id', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number',
#        'Report_Date_Time', 'Report_Description', 'Report_Status',
#        'Report_Type', 'Report_Text', 'has_kw', 'preds'],

DOWNSAMPLE_SIZE = 10

tqdm.pandas()

# model = MrModel(use_cache=False)
model = DummyModel()

patient_store_parent_dir = "stores/patient_stores"


dfs = {}

for root, dirs, files in os.walk(file_store_parent_dir):
    for dir in dirs:
        if dir.startswith("rpdr.parquet"):
            suffix = dir.split("_")[-1]
            dfs[suffix] = pd.read_parquet(os.path.join(root, dir))

pt_ids = dfs["Vis"].index
pts_out = []

pt_ids = pd.Index(pt_ids).to_series().sample(n=DOWNSAMPLE_SIZE, random_state=42)

feat_to_df = {}

# loop over features
for name, ft in PtFeaturesMeta.registry.items():
    all_rows = []
    if name not in [
        # "antibiotics"
        "smoking_status"
    ]:
        continue
    # if LlmFeatureBase not in ft.__bases__:
    if not hasattr(ft, "query"):
        continue
    print(f"Features: {name}")

    # Start timing
    start_time = time.time()

    # loop over patients
    for i, id in tqdm(enumerate(pt_ids), desc="Patient ID", total=len(pt_ids)):
        pt_df = {
            k: v.loc[[id]] if id in v.index else pd.DataFrame(columns=v.columns)
            for k, v in dfs.items()
        }["Vis"]

        # loop over reports (visits)
        for report in pt_df.iloc:
            # new_chunks = report["Report_Text"].apply(chunk_text).explode() # flatten series of lists into single series
            new_chunks = chunk_text(report["Report_Text"])
            rows = [{"chunk": chunk, "id": id} for chunk in new_chunks]
            # Update each row with all columns from pt_df
            for row in rows:
                row.update({col: report[col] for col in report.index})
            all_rows.extend(rows)

    # End timing and print results
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to process patients: {elapsed_time:.2f} seconds")
    print(f"Average time per patient: {elapsed_time/len(pt_ids):.2f} seconds")
    
    chunk_df = pd.DataFrame(all_rows)

    # Filter by keyword
    chunk_df["chunk_has_kw"] = False  # Initialize all to False
    print("Filtering by keyword")
    kw_df = chunk_df[chunk_df["chunk"].progress_apply(has_keyword, keywords=ft.keywords)]
    chunk_df.loc[kw_df.index, "chunk_has_kw"] = True

    # run all chunks through LLM
    kw_df["histories"] = model.format_chunk_qs(ft.query, kw_df["chunk"])

    # Calculate statistics for patient MRNs and record numbers
    pts = len(chunk_df["id"].unique())
    chunks_per_pt = len(chunk_df) / pts
    records_per_pt = len(chunk_df["Report_Number"].unique()) / pts
    print(f"Feature: {name}")
    print(f"Patients: {pts}")
    print(f"Chunks per patient: {chunks_per_pt}")
    print(f"Records per patient: {records_per_pt}")
    print(f"Chunks per record: {chunks_per_pt / records_per_pt}")

    

    kw_df["chunk_pred"] = model.predict(kw_df["histories"])

    chunk_df["chunk_pred"] = False
    chunk_df.loc[kw_df.index, "chunk_pred"] = kw_df["chunk_pred"]
    # (df[~df["chunk_pred"] & df["chunk_has_kw"]][["Report_Text"]].iloc[0])

    # print(df[~df["chunk_pred"] & df["chunk_has_kw"]]["chunk"].iloc[1])
    print(chunk_df["chunk_pred"].value_counts())
    feat_to_df[name] = chunk_df

    chunk_df["rn_pred"] = chunk_df["chunk_pred"].groupby(chunk_df["Report_Number"]).transform("any")
    chunk_df["rn_has_kw"] = chunk_df["chunk_has_kw"].groupby(chunk_df["Report_Number"]).transform("any")

    rn_df = chunk_df[["id", "EPIC_PMRN", "MRN", "Report_Number", "Report_Date_Time", "rn_has_kw", "rn_pred"]].drop_duplicates()
    assert len(set(rn_df["Report_Number"])) == len(rn_df)

    # save 5 MRNs that are predicted true and 5 that are predicted false to a txt named by the feature
    val_study_save_dir = f"redcap_mrns"
    os.makedirs(val_study_save_dir, exist_ok=True)

    # save the references
    rn_df.to_csv(f"{val_study_save_dir}/rn_{name}.csv", index=False)


    # Sample 5 random rows from each prediction group and save to validation file
    true_sample = rn_df[rn_df["rn_pred"]].sample(n=5, random_state=42)
    false_sample = rn_df[~rn_df["rn_pred"]].sample(n=5, random_state=42)
    val_df = pd.concat([true_sample, false_sample]).drop(columns=["rn_pred", "rn_has_kw"])
    val_df.to_csv(f"{val_study_save_dir}/rn_{name}_val.csv", index=False)


    # save the df to a parquet file
    chunk_df[["id", "EPIC_PMRN", "MRN", "Report_Number", "Report_Date_Time", "chunk_has_kw", "rn_has_kw", "rn_pred"]].to_parquet(f"{val_study_save_dir}/df_{name}.parquet")


# chunk_df = pd.DataFrame(data={"chunk_results": chunk_results}, index=pt_chunks)
# condense results per patient
# group chunks by patient and set True if any chunk was True
# results_df[name] = chunk_df.any(axis=1)
# store results
pass

# for i, id in tqdm(enumerate(pt_ids[:100])):
#     pt_df = {
#         k: v.loc[[id]] if id in v.index else pd.DataFrame(columns=v.columns)
#         for k, v in dfs.items()
#     }
#     # for suffix, df in dfs.items():
#     pt_out = {}
#     for name, ft in PtFeaturesMeta.registry.items():
#         if name in ["antibiotics"]:
#         # if True:
#             pt_out[name] = ft.compute(pt_df)
#         # print(df)
#     pts_out.append(pt_out)
# results_df = pd.DataFrame(pts_out)
# print
