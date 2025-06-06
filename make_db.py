import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from sqlalchemy import create_engine, text
import shutil
from pt_features import PtFeaturesMeta, PtFeatureBase

# data_dir = "rpdr_dumps/rpdr_latest/8"
notes_dir = "rpdr_dumps/rpdr_notes/"
structured_dir = "rpdr_dumps/rpdr_structured/"
store_dir = "stores/file_stores"
db_url = "sqlite:///stores/rpdr.db"  # Using SQLite for simplicity, can be changed to other databases
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



def list_all_files(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def parse_pipe_delimited_files(
    suffix_paths,
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

    # Convert Report_Date_Time to datetime if the column exists
    if 'Report_Date_Time' in df.columns:
        df['Report_Date_Time'] = pd.to_datetime(df['Report_Date_Time'], format='%m/%d/%Y %I:%M:%S %p')

    return df

    # write each structured




if __name__ == "__main__":

    # # for root, dirs, files in os.walk(data_dir):
    # files = list_all_files(notes_dir) + list_all_files(structured_dir)
    # suffix_paths = []
    # for suffix in [
    #     "Vis",
    #     "Dem",
    #     "Dia",
    #     "Med"
    # ]:
    #     for file in files:
    #         if file.endswith(suffix + ".txt"):
    #             # if file.endswith("1_" + suffix + ".txt"):
    #             suffix_paths.append(file)
    #     df = parse_pipe_delimited_files(
    #         suffix_paths,
    #         suffix=suffix,
    #     )
    #     # Create SQL engine and write to database
    #     engine = create_engine(db_url)
    #     table_name = f"{suffix.lower()}"
        
    #     # Write to database in chunks to handle large datasets
    #     chunksize = 10000
    #     for i in tqdm(range(0, len(df), chunksize)):
    #         df_chunk = df.iloc[i:i + chunksize]
    #         df_chunk.to_sql(
    #             name=table_name,
    #             con=engine,
    #             if_exists='append' if i > 0 else 'replace',
    #             index=True
    #         )

    ### Create a table containing key MRs corresponding to key dates ###

    ## Final visit date
    # Read the vis table into a dataframe
    engine = create_engine(db_url)
    with engine.connect() as conn:
        query = text("SELECT * FROM vis")
        result = conn.execute(query)
        vis_df = pd.DataFrame(result.fetchall(), columns=result.keys())

    vis_df["Report_Date_Time"] = pd.to_datetime(vis_df["Report_Date_Time"])
    # Calculate the date and Report_Number of the final visit for each patient
    final_visit_date = vis_df.groupby("EMPI")["Report_Date_Time"].max()
    final_visit_report_number = vis_df.groupby("EMPI")["Report_Number"].max()

    # Create a new table with the final visit date and report number
    important_dates_df = pd.DataFrame({
        "EMPI": final_visit_date.index,
        "final_visit_date": final_visit_date,
        "final_visit_report_number": final_visit_report_number
    })



    ## Acne dianoses
    # Read the dia (diagnosis) db and get the date of the first acne diagnosis

    with engine.connect() as conn:
        query = text("SELECT * FROM dia WHERE Code IN ('L70.0', 'L70.8', 'L70.9')")
        result = conn.execute(query)
        acne_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        acne_df["Date"] = pd.to_datetime(acne_df["Date"], format="%m/%d/%Y")

    # Calculate earliest acne diagnosis date for each patient
    earliest_acne_date = acne_df.groupby("EMPI")["Date"].min()

    # Add earliest acne date to important dates dataframe
    important_dates_df["first_acne_date"] = important_dates_df["EMPI"].map(earliest_acne_date)


    ## 2 Year Followup Date
    # Calculate the record closest to the 2 year followup date
    # raise Exception("None of this will work until we have the correct RPDR data")
    ideal_two_year_visit_date = important_dates_df["final_visit_date"] + pd.Timedelta(days=365*2)
    time_since_ideal_two_year_visit_date = vis_df["Report_Date_Time"] - ideal_two_year_visit_date
    closest_two_year_visit_date = vis_df.loc[time_since_ideal_two_year_visit_date.abs().idxmin()] if not time_since_ideal_two_year_visit_date.empty else None
    important_dates_df["two_year_followup_date"] = closest_two_year_visit_date["Report_Date_Time"]




    # Write to a new table in the database
    important_dates_df.to_sql(
        name="final_visit",
        con=engine,
        if_exists="replace",
        index=False
    )

    ### Calculate the 


    print(important_dates_df)