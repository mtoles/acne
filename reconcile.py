import pandas as pd

edw_df = pd.read_csv("edw_dumps/abx_duration_all.tsv", sep="\t")
rpdr_df = pd.read_parquet("stores/file_stores/rpdr.parquet_Dem")

total_users = len(set(edw_df.index).union(set(rpdr_df.index)))
edw_only = len(set(edw_df.index) - set(rpdr_df.index))
rpdr_only = len(set(rpdr_df.index) - set(edw_df.index))
both = len(set(edw_df.index).intersection(set(rpdr_df.index)))
neither = total_users - edw_only - rpdr_only - both
print(f"EDW only: {edw_only / total_users * 100:.2f}%")
print(f"RPDR only: {rpdr_only / total_users * 100:.2f}%")
print(f"Both: {both / total_users * 100:.2f}%")
print(f"Neither: {neither / total_users * 100:.2f}%")
