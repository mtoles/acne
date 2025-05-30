# go through each target feature and get N Report_Texts that contain any keyword from the feature. 
# save the relevant data to a txt file named along with the feature name

from pt_features import PtFeaturesMeta, PtFeatureBase
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
    if not cls.val_var:
        continue

    with db.connect() as conn:
        query = text(f"SELECT * FROM rpdr_vis")
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys()).sample(frac=1)

    # check each record if it has the keywords
    print(f"Checking {cls.__name__} for keywords...")

    dfs = []
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[start:start+BATCH_SIZE].copy()
        batch["has_kw"] = batch["Report_Text"].progress_apply(has_keyword, keywords=cls.keywords)
        batch = batch[batch["has_kw"]]
        batch["included_kw"] = None
        for kw in cls.keywords[::-1]:
            batch.loc[batch["included_kw"].isna(), "included_kw"] = batch["Report_Text"].apply(lambda x: kw if kw.lower() in x.lower() else None)
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
        f.write(f"## Number of records with keyword: {len(df[df['included_kw'].notna()])}\n")
        f.write(f"## Number of records: {len(df)}\n")