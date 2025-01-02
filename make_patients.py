import pandas as pd
import os

from tqdm import tqdm

from make_store import store_dir as file_store_parent_dir
from pt_features import PtFeaturesMeta


patient_store_parent_dir = "stores/patient_stores"


dfs = {}

for root, dirs, files in os.walk(file_store_parent_dir):
    for dir in dirs:
        if dir.startswith("rpdr.parquet"):
            suffix = dir.split("_")[-1]
            dfs[suffix] = pd.read_parquet(os.path.join(root, dir))

pt_ids = dfs["Dem"].index
pts_out = []
for i, id in tqdm(enumerate(pt_ids[:10])):
    pt_df = {
        k: v.loc[[id]] if id in v.index else pd.DataFrame(columns=v.columns)
        for k, v in dfs.items()
    }
    # for suffix, df in dfs.items():
    pt_out = {}
    for name, ft in PtFeaturesMeta.registry.items():
        # if name in ["bmi"]:
        if True:
            pt_out[name] = ft.compute(pt_df)
        # print(df)
    pts_out.append(pt_out)
results_df = pd.DataFrame(pts_out)
print
