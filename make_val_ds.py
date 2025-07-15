# go through each target feature and get N Report_Texts that contain any keyword from the feature. 
# save the relevant data to a txt file named along with the feature name

from pt_features import PtFeaturesMeta, PtFeatureBase, PtDateFeatureBase
from utils import *
from sqlalchemy import create_engine, text
from make_db import db_url
import pandas as pd
from tqdm import tqdm
import os
tqdm.pandas()

db = create_engine(db_url)
BATCH_SIZE = 1000
MAX_DS_SIZE = 100



for cls in PtFeaturesMeta.registry.values():
    # if not issubclass(cls, PtDateFeatureBase): # temp, to fill in date features we didn't do yet
        # continue
    if not cls.val_var:
        continue
    if not cls.__name__ == "antibiotic_duration":
        print(f"Skipping {cls.__name__}")
        continue
    # check each record if it has the keywords
    print(f"Checking {cls.__name__} for keywords...")



    with db.connect() as conn:
        query = text(f"SELECT * FROM vis ORDER BY RANDOM() LIMIT 10000")
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys()).sample(frac=1)


    dfs = []
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[start:start+BATCH_SIZE].copy()
        batch["has_kw"] = batch["Report_Text"].progress_apply(has_keyword, keywords=cls.keywords)
        batch = batch[batch["has_kw"]]
        batch["included_kw"] = batch["Report_Text"].apply(lambda x: [kw for kw in cls.keywords if kw.lower() in x.lower()])
        dfs.append(batch)

        total_len = sum([len(x) for x in dfs])
        if total_len >= MAX_DS_SIZE:
            break

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=df.columns)
    df = df.iloc[:MAX_DS_SIZE]

    # save the df to a csv file
    # include cols: EMPI,EPIC_PMRN,MRN,Report_Number,Report_Date_Time
    df = df[["EMPI", "EPIC_PMRN", "MRN", "Report_Number", "Report_Date_Time", "included_kw"]]
    os.makedirs("val_ds", exist_ok=True)
    df.to_csv(f"val_ds/{cls.__name__}.csv", index=False)

    # print metadata to a md file
    with open(f"val_ds/{cls.__name__}.md", "w") as f:
        f.write(f"# {cls.__name__}\n")
        f.write(f"## Query\n")
        f.write(f"{cls.query}\n")
        f.write(f"## Keywords\n")
        f.write(f"{cls.keywords}\n")
        # f.write(f"## Number of records with keyword: {len(df[df['included_kw'].notna()])}\n")
        # f.write(f"## Number of records: {len(df)}\n")