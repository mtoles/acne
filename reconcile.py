import pandas as pd

edw_df = pd.read_csv("edw_dumps/abx_duration_with_empi_HEADER.csv", sep=",")
edw_df.EMPI = edw_df.EMPI.astype(float)
edw_df = edw_df[~edw_df.EMPI.isna()]
edw_df.EMPI = edw_df.EMPI.astype(int).astype(str)
rpdr_df = pd.read_parquet("stores/file_stores/rpdr.parquet_Dem")


total_users = len(set(edw_df.EMPI).union(set(rpdr_df.index)))
edw_only = len(set(edw_df.EMPI) - set(rpdr_df.index))
rpdr_only = len(set(rpdr_df.index) - set(edw_df.EMPI))
both = len(set(edw_df.EMPI).intersection(set(rpdr_df.index)))
neither = total_users - edw_only - rpdr_only - both
print(f"EDW only: {edw_only / total_users * 100:.2f}%")
print(f"RPDR only: {rpdr_only / total_users * 100:.2f}%")
print(f"Both: {both / total_users * 100:.2f}%")
print(f"Neither: {neither / total_users * 100:.2f}%")

# write all EMPIs from both EDW and RPDR to a file
with open("edw_empis.txt", "w") as f:
    for empi in set(edw_df.EMPI) - set(rpdr_df.index):
        f.write(empi + "\n")