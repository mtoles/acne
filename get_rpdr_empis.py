from make_db import parse_pipe_delimited_files
import os
import pandas as pd

target_dir = "/home/mtoles/acne/rpdr_dumps/rpdr_notes"

# read every _Vis.txt file in the target dir


dfs = []
paths = []
for root, _, files in os.walk(target_dir):
    for file in files:
        if file.endswith("_Vis.txt"):
            filepath = os.path.join(root, file)
            paths.append(filepath)

for path in paths:
    df = parse_pipe_delimited_files([path], "Vis")
    dfs.append(df)


df = pd.concat(dfs)

# save the df to a txt file, prepending each empi with "EMPI:"
empis = df.index.unique()
with open("reconciliation/rpdr_empis.txt", "w") as f:
    for empi in empis:
        f.write(f"EMPI:{empi}\n")

