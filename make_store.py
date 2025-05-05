import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
import shutil

data_dir = "rpdr_dumps/rpdr_latest/8"
store_dir = "stores/file_stores"
acceptable_suffixes = [
    # "All",  # allergy
    # "Dem",  # demographics
    # "Dia",  # diagnosis
    # "Enc",  # encounter
    # "Med",  # medications
    # "Phy",  # ?
    # "Prc",  # procedure?
    # "Rdt",  # radiology
    # "Rfv",  # refill
    # "Dis",  # disease
    # "End",  # endocrinology
    # "Hnp",  # history and physical
    # "Lno",  # letters
    # "Mic",  # micribiology
    # "Mrn",  # medical record number
    # "Opn",  # operation
    # "Pat",  # patient
    # "Prg",  # progress
    # "Pul",  # pulmonary
    # "Rad",  # radiology
    "Vis",  # visit
]

store_path = os.path.join(store_dir, "rpdr.parquet")


def list_all_files(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def parse_pipe_delimited_file(
    suffix_paths,
    store_path_suffix,
    #   min_itemsize,
    suffix,
):
    # Read the files
    rows = []
    for filepath in tqdm(suffix_paths):
        with open(filepath, "r", encoding="utf-8") as f:
            f.seek(0)
            headers = f.readline().strip().split("|")
            content = f.read()
            if suffix in [
                "Dem",
                "Dia",
                "Med",
                "Mrn",
                "All",
                "Enc",
                "Phy",
                "Prc",
                "Rdt",
                "Rfv",
            ]:
                content = content.replace("\n", "|")
            elif suffix in ["Mic"]:
                content = content.replace("[report_end]", "|")
            else:
                content = content.replace("[report_end]\n", "|")
            cells = content.split("|")  # [: len(headers) * 1000]
            rows.extend(
                [
                    cells[i : i + len(headers)]
                    for i in range(0, len(cells), len(headers))
                ]
            )

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    first_col = df.columns[0]
    df[first_col] = df[first_col].apply(lambda x: x.strip())
    # print(df)
    # drop rows where EMPI is ''
    df = df.dropna(subset=[first_col])
    df = df[df[first_col] != ""]
    df = df.set_index(first_col)

    # Append to HDF5 store
    # /*************  ✨ Codeium Command 🌟  *************/
    # dfchunks = np.array_split(df, 100)
    # for i, dfchunk in enumerate(dfchunks):
    #     store.put(f"{suffix}/{i}", dfchunk, format="table", complib="blosc")
    # store.put(suffix, df, format="table", complib="blosc")
    # print(df)
    chunksize = 10**100
    # Delete the parquet dataset at the specified path
    if os.path.exists(store_path_suffix):
        shutil.rmtree(store_path_suffix, ignore_errors=True)
    for i in tqdm(range(0, len(df), chunksize)):
        table = pa.Table.from_pandas(df.iloc[i : i + chunksize])
        # df_chunk = df.iloc[i : i + chunksize]
        pq.write_to_dataset(table, root_path=store_path_suffix)
        # pq.write_table(table, store_path, compression="snappy")
    # /******  9b6bf1e4-998f-414b-933e-e4d7f7045877  *******/
    parquet_df = pq.read_table(f"{store_path_suffix}").to_pandas()
    assert len(parquet_df) == len(df)
    print(parquet_df)
    print


if __name__ == "__main__":

    # for root, dirs, files in os.walk(data_dir):
    if os.path.exists(store_path):
        shutil.rmtree(store_path, ignore_errors=True)
    files = list_all_files(data_dir)
    suffix_paths = []
    for suffix in acceptable_suffixes:
        store_path_suffix = f"{store_path}_{suffix}"
        for file in files:
            if file.endswith(suffix + ".txt"):
                # if file.endswith("1_" + suffix + ".txt"):
                suffix_paths.append(file)
        parse_pipe_delimited_file(
            suffix_paths,
            store_path_suffix,
            # min_itemsize=min_itemsize,
            suffix=suffix,
        )

        # store_head = store.select(suffix, where="index=2")
        # print
