from make_store import store_path, acceptable_suffixes
from pt_features import PtFeaturesMeta

from tqdm import tqdm
import pandas as pd



def get_user_data(empi, store):
    tables = {}
    for suf in acceptable_suffixes:
        tables[suf] = store[suf].loc[empi]
    return tables

### Read the stores
print
# close the store if it is already open

suffixes = ["Vis"]
stores = {}
for suf in suffixes:
    store_path_suffix = f"{store_path}_{suf}"
    # Read the Parquet file instead of HDF
    stores[suf] = pd.read_parquet(store_path_suffix)

ready_fts = ["antibiotics"]

empis = stores[suffixes[0]].index  # Enterprise Master Patient Index

ds_list = []

for name, ft in PtFeaturesMeta.registry.items():
    if ft.__name__ not in ready_fts:
        continue
    # if name in ready_fns:
    for empi in tqdm(empis[:2]):
        user_data = get_user_data(empi, stores)
        # TESTING
        user_data["Vis"].iloc[0]["Report_Text"] = "This patient was proscribed 100mg of amoxicillin"
        ds_list.append(ft.compute(user_data))


print


## define function name


#
