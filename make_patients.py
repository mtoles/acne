import pandas as pd
import os

from tqdm import tqdm

from make_store import store_dir as file_store_parent_dir
from pt_features import PtFeaturesMeta, LlmFeatureBase
from utils import chunk_text
from models import MrModel


model = MrModel()

patient_store_parent_dir = "stores/patient_stores"


dfs = {}

for root, dirs, files in os.walk(file_store_parent_dir):
    for dir in dirs:
        if dir.startswith("rpdr.parquet"):
            suffix = dir.split("_")[-1]
            dfs[suffix] = pd.read_parquet(os.path.join(root, dir))

pt_ids = dfs["Vis"].index
pts_out = []


pt_ids = pt_ids[:2]


feat_to_df = {}

# loop over features
for name, ft in PtFeaturesMeta.registry.items():
    all_rows = []
    if name not in ["antibiotics"]:
        continue
    if LlmFeatureBase not in ft.__bases__:
        continue


    # loop over patients
    for i, id in tqdm(enumerate(pt_ids)):
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
    df = pd.DataFrame(all_rows)

    # run all chunks through LLM
    df["histories"] = model.format_chunk_qs(ft.query, df["chunk"])

    df["predictions"] = model.predict(df["histories"])
    feat_to_df[name] = df

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
