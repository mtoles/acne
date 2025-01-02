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

store = pd.HDFStore(store_path, "r")

ready_fns = ["sex"]

empis = store["/Dem"].index  # Enterprise Master Patient Index

ds_list = []

for name, ft in PtFeaturesMeta.registry.items():
    # if name in ready_fns:
    for empi in tqdm(empis[:10]):
        user_data = get_user_data(empi, store)
        ds_list.append(ft.compute(user_data, store))


print


## define function name


#
