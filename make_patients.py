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


    # @staticmethod
    # def compute(dfs: dict):
    #     df = dfs["Vis"]
    #     query = "Does this medical record excerpt indicate that the patient took any of the following antibiotics: amoxicillin, cephalexin, azithromycin, or tmp-smx?"
    #     # Apply chunking to each row and flatten the list
    #     all_chunks = []
    #     for chunks in df['Report_Text'].apply(chunk_text):
    #         all_chunks.extend(chunks)
    #     histories = model.format_chunk_qs(query, all_chunks)
    #     chunk_results = model.predict(histories)
    #     return any(chunk_results)

pt_ids = pt_ids[:2]

results_df = pd.DataFrame(index=pt_ids)

histories = []
for name, ft in PtFeaturesMeta.registry.items():
    if name not in ["antibiotics"]:
        continue
    if LlmFeatureBase not in ft.__bases__:
        continue

    # queue up and chunk all patient data
    all_chunks = []
    for i, id in tqdm(enumerate(pt_ids)):
        pt_df = {
            k: v.loc[[id]] if id in v.index else pd.DataFrame(columns=v.columns)
            for k, v in dfs.items()
        }
        all_chunks.extend(pt_df["Vis"]["Report_Text"].apply(chunk_text))

    # run all chunks through LLM
    histories.extend(model.format_chunk_qs(ft.query, all_chunks))

chunk_results = model.predict(histories)
chunk_df = pd.DataFrame(data={"chunk_results": chunk_results}, index=pt_ids)
# condense results per patient
# group chunks by patient and set True if any chunk was True
results_df[name] = chunk_df.any(axis=1)
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
