import pandas as pd
import os
from tqdm import tqdm


def parse_pipe_delimited_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
        col_names = header.split("|")
        n_cols = len(col_names)
        # read the rest of the file until you have a complete row
        first_row = ""
        while True:
            first_row += f.readline()
            n_pipes = first_row.count("|")
            if n_pipes > n_cols:
                break
        first_row_cells = first_row.split("|")[:n_cols]
    return col_names, first_row_cells


rpdr_dir = '/Users/matt/Partners HealthCare Dropbox/Matthew Toles/RPDR for Dorsa/May 29 RPDR/2023P001442 - "All patients diagnosed w: acne before 5:31:2019-1"'
output_path = "rpdr_summaries/summary.md"

report_entries = []
for file in tqdm(os.listdir(rpdr_dir)):
    file_type = file.split("_")[-1].split(".")[0]
    if file_type in ["Log", "Let"]:
        continue
    file_path = os.path.join(rpdr_dir, file)
    col_names, first_row = parse_pipe_delimited_file(file_path)
    cell_dict = dict(zip(col_names, first_row))
    file_entry = (
        f"# {file_type}:\n"
        + "\n\n".join([f"## {k}: \n{v}" for k, v in cell_dict.items()])
        + "\n\n"
        + "\n\n================\n\n"
    )
    report_entries.append(file_entry)
    print


with open(output_path, "w") as f:
    f.write("\n".join(report_entries))
