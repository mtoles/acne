Annotated Social Determinants of Health Dataset for Adverse Pregnancy Outcomes
Dataset Structure
Two annotated CSV files are included:
* MIMICIII_annotations_PregnancySDoH.csv
* MIMICIV_annotations_PregnancySDoH.csv
Both files contain the following fields:
ColumnDescriptionsubject_idAnonymized identifier for the patient.textDischarge summary text from clinical note.complicationBinary indicator of pregnancy outcome (1 = adverse complication, 0 = normal pregnancy).social_supportBinary indicator of social support (1 = present, 0 = absent or undocumented).occupationBinary indicator of active employment (1 = employed, 0 = unemployed or undocumented).substance_useBinary indicator of substance use (1 = present, 0 = absent or undocumented).Note: Column headers in MIMIC-III use uppercase, while MIMIC-IV uses lowercase. Functionality remains consistent across files.
Setup and Access
* Files are provided in standard CSV format.
* Users can load the datasets using any data processing tool (e.g., Python pandas, R).
* Example (Python):
import pandas as pd

# Load MIMIC-III annotations
df_mimiciii = pd.read_csv('MIMICIII_annotations_PregnancySDoH.csv')

# Load MIMIC-IV annotations
df_mimiciv = pd.read_csv('MIMICIV_annotations_PregnancySDoH.csv')
