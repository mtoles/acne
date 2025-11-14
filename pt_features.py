import pandas as pd

# from models import MrModel
from utils import chunk_text
from collections import Counter, defaultdict
from datetime import datetime
from models import MrModel, retry_with_validation
from datetime import timedelta
import math
from utils import OptionType

### General Functions ###

cancer_icd9s = [
    str(i)
    for sublist in [
        list(range(140, 150)),
        list(range(150, 160)),
        list(range(160, 166)),
        list(range(170, 177)),
        list(range(179, 190)),
        list(range(190, 200)),
        list(range(200, 210)),
    ]
    for i in sublist
]
# fmt: off
cancer_icd10s = ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C40", "C41", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C77", "C78", "C79", "C80", "C7A", "C7B", "C81", "C82", "C83", "C84", "C85", "C86", "C87", "C88", "C89", "C90", "C91", "C92", "C93", "C94", "C95", "C96",]

SPECIFIC_CANCERS = ["Oropharyngeal cancer", "Nasopharyngeal cancer", "Hypopharyngeal cancer", "Esophageal cancer", "Stomach cancer", "Small intestine cancer", "Colorectal cancer", "Anal cancer", "Liver cancer", "Intrahepatic bile duct cancer", "Gallbladder cancer", "Extrahepatic biliary tract cancer", "Pancreatic cancer", "Splenic cancer", "Other intestinal tract cancer", "Retroperitoneal cancer", "Peritoneal cancer", "Nasal cavity and paranasal sinus cancer", "Middle ear cancer", "Laryngeal cancer", "Tracheal cancer", "Bronchial cancer", "Lung cancer", "Thymic cancer", "Heart cancer", "Mediastinal cancer", "Pleural cancer", "Bone cancer", "Melanoma", "Non-melanoma skin cancer", "Merkel cell carcinoma", "Mesothelioma", "Kaposi sarcoma", "Ewing sarcoma", "Rhabdomyosarcoma", "Osteosarcoma", "Chondrosarcoma", "Fibrosarcoma", "Soft tissue sarcoma", "Angiosarcoma", "Liposarcoma", "Breast cancer", "Vulvar cancer", "Vaginal cancer", "Cervical cancer", "Uterine cancer", "Ovarian cancer", "Placental cancer", "Penile cancer", "Prostate cancer", "Testicular cancer", "Kidney cancer", "Urethral cancer", "Ureteral cancer", "Bladder cancer", "Eye cancer", "Brain cancer", "Spinal cord cancer", "Meningeal cancer", "Cranial nerve cancer", "Thyroid cancer", "Adrenal cancer", "Carcinoid tumor", "Neuroendocrine tumor", "Parathyroid cancer", "Pituitary cancer", "Hodgkin lymphoma", "Non-Hodgkin lymphoma", "Leukemia", "Multiple myeloma", "Malignant mast cell tumor", "Malignant histiocytosis", "Myelodysplastic syndrome", "Choriocarcinoma", "Polycythemia vera", "Essential thrombocythemia", "Myelofibrosis", "Plasmacytoma", "Salivary gland cancer", "Appendiceal cancer", "Gliomas", "Glioblastoma", "Astrocytoma", "Oligodendroglioma", "Leiomyosarcoma", "Synovial sarcoma"]

EXCLUDED_TRANSPLANTS = ["corneal", "skin graft", "hair", "osteochondral", "cartilage", "bone", "valve", "autograft", "hip", "shoulder", "tendon", "fecal", "skin" ]
INCLUDED_TRANSPLANTS = ["liver", "kidney", "pancreas", "heart", "lung", "intestine", "middle ear", "skin", "bone", "bone marrow", "heart valve", "connective tissue", "vascular composite allograft"]
CANCER_STAGE_SYNTHETIC_KEYWORDS = [
        "in situ",
        "non-invasive",
        "non invasive",
        "stage 0",
        "stage 1",
        "stage 2",
        "stage 3",
        "stage 4",
        "stage I",
        "stage II",
        "stage III",
        "stage IV",
        "metastatic",
    ]
# fmt: on

KEYWORD_ADDITIONAL_INFO = defaultdict(
    lambda: None,
    {
        "amoxicillin": "amoxicillin (DO NOT INCLUDE amoxicillin clavulanate, which is sometimes first mentioned as amoxicillin clavulanate, then mentioned as amoxicillin)",
        "doxycycline": "doxycycline (BE SURE TO INCLUDE doxycycline hyclate and doxycycline monohydrate, which are sometimes mentioned as doxycycline)",
    },
)

# used for questions about meidcations
ENTRY_FORMAT_STR = """Entries in this medical record typically adhere to the following format, though the header may be missing:
###
                                                   Disp             Refills           Start               End

cetirizine (ZYRTEC) 10 MG tablet (Taking)          30 tablet        1                 01/30/2020          02/30/2021

Sig - Route: Take 1 tablet (10 mg total) by mouth daily. Take       daily during allergy season. -  Oral
###
Note that some fields may be missing, and the format may not be perfect. If you see the meidcation in this format or under the (Taking) header, assume the patient took the medication, even if Disp/Refills/Start/End are missing. Note that other formats, such as long form notes, may also indicate medication use.
"""
# strings

NOT_CANCER_STR = "Pre-cancer, atypia, dysplasia, pre-malignant, non-malignant, or benign conditions are not considered cancer. Cancer screenings do not necessarily indicate cancer."
FAMILY_HISTORY_STR = (
    "Ignore family history of cancer (e.g. mother, father, sister, brother, etc.)."
)
DATE_INTERPOLATION_STR = "If only the month and year are given, assume the date was the 15th. If only the year is given, assume the date was July 2. Answer in the format YYYYMMDD, including leading zeros."



# helper functions
def get_rows_by_icd(df: pd.DataFrame, icd9_list: list, icd10_list: list):
    icd10_df = df[df["Code"].apply(lambda x: any(x.startswith(y) for y in icd9_list))]
    icd9_df = df[df["Code"].apply(lambda x: any(x.startswith(y) for y in icd10_list))]
    out_df = pd.concat([icd10_df, icd9_df])
    return out_df

### Patient Features ###

class PtFeaturesMeta(type):
    registry = {}

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if name not in ["PtFeatureBase", "LlmFeatureBase"]:
            PtFeaturesMeta.registry[name] = cls
        return cls


class PtFeatureBase(metaclass=PtFeaturesMeta):
    val_var = False  # whether the variable is checked in the validation study
    max_tokens = 1  # default max tokens for model prediction

    @classmethod
    def forward(cls, chunk: str, keyword: str, model, inference_type="logit", **kwargs):
        """
        Forward pass through the model for a given chunk and keyword from the query.
        This method encapsulates the boilerplate logic for query generation,
        history formatting, and prediction.

        Args:
            chunk: The text chunk to process
            keyword: The keyword that was found in the chunk
            model: The model instance to use for prediction
            inference_type: Type of inference to use ('logit' or 'cot')
            **kwargs: Additional arguments to pass to the query method

        Returns:
            str: The model prediction
        """
        keyword = KEYWORD_ADDITIONAL_INFO[keyword] or keyword
        # Generate query using the class's query method

        if "custom_query" in kwargs:
            query = kwargs["custom_query"]
        else:
            query = cls.query(chunk=chunk, keyword=keyword, **kwargs)

        # Format the query and chunk for the model
        history = model.format_chunk_qs(q=query, chunk=chunk, options=cls.options)

        if inference_type == "cot":
            # Use chain of thought for multiple choice questions
            pred = model.predict_with_cot(
                history,
                options=cls.options,
                max_answer_tokens=cls.max_tokens,
                sample=True,
            )
        else:
            # Use logit trick for multiple choice questions (default)
            pred = model.predict_single_with_logit_trick(
                history, output_choices=set(cls.options)
            )

        return {cls.__name__: pred}


class PtDateFeatureBase(PtFeatureBase):
    def pooling_fn_latest(preds: list):
        # return the most recent date
        dates = [pd.to_datetime(pred, format="%Y%m%d") for pred in preds if pred != "X"]
        if not dates:
            return "X"
        return dates.max().strftime("%Y-%m-%d")

    def pooling_fn_earliest(preds: list):
        # return the earliest date
        dates = [pd.to_datetime(pred, format="%Y%m%d") for pred in preds if pred != "X"]
        if not dates:
            return "X"
        return dates.min().strftime("%Y-%m-%d")

    options = OptionType.DATE
    max_tokens = 8


class bmi(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Phy"]
        point = df[df["Code"] == "BMI"]["Result"].astype(float).mean()
        return point


class zip_code(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        # zip = store["Dem"].loc[pt_id, "Zip_code"]
        return zip


class smoking_status(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Phy"]
        smoking_statuses = df[df["Concept_Name"] == "Smoking status"]["Result"]
        if smoking_statuses.empty:
            return None
        # get the majority case
        smoking_status = smoking_statuses.value_counts().index[0]
        return smoking_status

    @classmethod
    def query(cls, **kwargs):
        """Return the query for smoking status."""
        return "What is the smoking status of this patient? If it just says 'smoker: no', consider it as 'never smoked'. Options are: A. Never smoked, B. Former smoker, C. Current smoker, D. Unknown"

    options = ["A", "B", "C", "D"]
    keywords = ["smokes", "smoker", "smoking", "tobacco"]
    val_var = True

    def pooling_fn(preds: list):
        # return the most common smoking status
        preds = [x for x in preds if x in ["A", "B", "C", "D"]]  # drop NO_KW
        counts = Counter(preds)
        return counts.most_common(1)[0][0]


class smoking_amount(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for smoking amount."""
        return "How many packs per week does this patient smoke? If a range is given, take the upper bound. Ignore results from questionaires. Round all values up to the nearest integer (i.e. 0.1 -> 1). Do not attempt to infer the number of packs from questionnaire scores (e.g. TAPS, AUDIT-C). Options are: A. 0 (does not smoke), B. 1-2, C. 3-5, D. 6+, E. Smoker but unknown quantity, F. No indication of smoking status"

    options = ["A", "B", "C", "D", "E", "F"]
    keywords = smoking_status.keywords
    val_var = True

    def pooling_fn(preds: list):
        # Get counts of A-D options
        abcd_counts = Counter([p for p in preds if p in ["A", "B", "C", "D"]])
        if abcd_counts:
            return abcd_counts.most_common(1)[0][0]
        # If no A-D, check for E
        elif "E" in preds:
            return "E"
        # Otherwise return F
        else:
            return "F"


class alcohol_status(PtFeatureBase):
    pass

    @classmethod
    def query(cls, **kwargs):
        """Return the query for alcohol status."""
        return "What is the alcohol status of this patient? (Note: ETOH is the abbreviation for ethanol/drinking/alcohol use.) If the patient abuses alcohol, return A. If no information is given, return C. DO NOT use questionnaire scores (e.g. AUDIT-C, TAPS) to determine alcohol status. Options are: A. Currently Drinks, B. Does not currently drink, C. No indication of alcohol status"

    options = ["A", "B", "C"]
    keywords = ["alcohol", "drinks", "etoh"]
    val_var = True

    def pooling_fn(preds: list):
        # return the most common alcohol status among A, B, C
        abc_counts = Counter([p for p in preds if p in ["A", "B"]])
        if abc_counts:
            return abc_counts.most_common(1)[0][0]
        # If no A-C, return D
        return "C"


class alcohol_amount(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for alcohol amount."""
        return "How many drinks per week does this patient drink? Round all values up to the nearest integer (i.e. 0.1 -> 1), using the largest value if a range is given. Only use stated numbers; do not guess based on words like 'rare'. DO NOT use questionnaire scores (e.g. AUDIT-C, TAPS) to determine alcohol status. Options are: A. 0 (sober or does not drink), B. 1-2, C. 3-5, D. 6+, E. Drinker but unknown quantity, F. No indication of alcohol status"

    options = ["A", "B", "C", "D", "E", "F"]
    keywords = alcohol_status.keywords
    val_var = True

    def pooling_fn(preds: list):
        if "D" in preds:
            return "D"
        elif "C" in preds:
            return "C"
        elif "B" in preds:
            return "B"
        elif "A" in preds:
            return "A"
        elif "E" in preds:
            return "E"
        else:
            return "F"


class alcohol_amount_drinking_index(PtFeatureBase):
    pass


class transplant(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for transplant status."""
        return f"Does this medical record indicate that the patient received a transplant, solid organ transplant, stem cell transplant, or bone marrow transplant? Do not include transplant types: {', '.join(EXCLUDED_TRANSPLANTS)}. Options are: A. Yes, B. No"

    options = ["A", "B"]
    keywords = [
        "transplant",
        "transplantation",
    ]
    val_var = True

    def pooling_fn(preds: list):
        if "A" in preds:
            return "A"
        return "B"


class transplant_date(PtDateFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for transplant date."""
        return f"What was the date of the patient's most recent organ transplant? Do not include transplant types: {', '.join(EXCLUDED_TRANSPLANTS)}. If the patient received a transplant but no date is specified, return U. If the record gives no indication of transplant, return X {DATE_INTERPOLATION_STR}"

    keywords = transplant.keywords
    synthetic_keywords = (
        [
            # "transplant on",
        ]
        + [f"status post {x} transplant" for x in INCLUDED_TRANSPLANTS]
        + [f"s/p {x} transplant" for x in INCLUDED_TRANSPLANTS]
    )
    val_var = True
    pooling_fn = PtDateFeatureBase.pooling_fn_latest


class immunosuppressed_disease(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for immunosuppressed disease status."""
        return "Does this medical record indicate that the patient had an immunosuppressed disease, including leukemia, lymphoma, HIV, AIDS, or immune deficiency? Options are: A. Yes, B. No"

    options = ["A", "B"]
    keywords = [
        "leukemia",
        "lymphoma",
        "HIV",
        "immune deficiency",
        "immunodeficiency",
        "immunosuppressed",
        "immunocompromised",
        "AIDS",
    ]
    synthetic_keywords = [
        # "he is immunosuppressed",
        # "she is immunosuppressed",
        "is immunosuppressed",
        "was immunosuppressed",
    ]
    val_var = True

    def pooling_fn(preds: list):
        if "A" in preds:
            return "A"
        return "B"


class immunosuppressed_transplant_organ_name(PtFeatureBase):
    pass


class immunosuppressed_transplant_organ_date(PtDateFeatureBase):
    pooling_fn = PtDateFeatureBase.pooling_fn_latest


class immunosuppressed_medication_medication_name(PtFeatureBase):
    pass


class immunosuppressed_disease_disease_name(PtFeatureBase):
    pass


class sex(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        # sex = store["Dem"].loc[pt_id, "Sex_At_Birth"]
        df = dfs["Dem"]
        sexes = df["Sex_At_Birth"]
        if sexes.empty:
            return None
        sex = sexes.value_counts().index[0]
        return sex


class race(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        # race = store["Dem"].loc[pt_id, "Race_Group"]
        df = dfs["Dem"]
        races = df["Race_Group"]
        if races.empty:
            return None
        race = races.value_counts().index[0]
        return race


class ethnicity(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        # eth = store["Dem"].loc[pt_id, "Ethnic_Group"]
        df = dfs["Dem"]
        ethnicities = df["Ethnic_Group"]
        if ethnicities.empty:
            return None
        eth = ethnicities.value_counts().index[0]
        return eth


class language(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dem"]
        languages = df["Language_group"]
        if languages.empty:
            return None
        language = languages.value_counts().index[0]
        return language


class insurance(PtFeatureBase):
    pass


class military(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dem"]
        mil = df["Is_a_veteran"]
        if mil.empty:
            return None
        mil = mil.value_counts().index[0]
        return mil

    @classmethod
    def query(cls, **kwargs):
        """Return the query for military service status."""
        return "Does this medical record indicate that the patient served in the military? Options are: A. Yes, B. No"

    keywords = [
        "military",
        "military",
        "veteran",
        "marine corps",
        "army",
        "navy",
        "air force",
        "coast guard",
    ]
    # val_var = True


class military_years(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for military years of service."""
        return "How many years did this patient serve in the military? Answer with a number, or if the patient is not a veteran, return X"

    keywords = military.keywords
    # val_var = True


class military_retirement_date(PtDateFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for military retirement date."""
        return "What was the date of the patient's military retirement? If the patient is not a veteran, return X {DATE_INTERPOLATION_STR}"

    keywords = military.keywords
    pooling_fn = PtDateFeatureBase.pooling_fn_latest


class military_agent_orange(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for Agent Orange exposure."""
        return "Does this medical record indicate that the patient was exposed to Agent Orange? Options are: A. Yes, B. No"

    options = ["A", "B"]
    keywords = ["agent orange"]
    # val_var = True


class military_oef_oif(PtFeatureBase):
    pass


class military_camp_lejeune(PtFeatureBase):
    pass


class pregnancy(PtFeatureBase):
    pass


class pregnancy_age(PtFeatureBase):
    pass


class pregnancy_count(PtFeatureBase):
    pass


class early_menstruation(PtFeatureBase):
    pass


class menopause(PtFeatureBase):
    pass


class menopause_late(PtFeatureBase):
    pass


class birth_control_copper_iud(PtFeatureBase):
    # structured data text search
    pass


class birth_control_hormonal_iud(PtFeatureBase):
    # structured data text search
    pass


class birth_control_implant(PtFeatureBase):
    # structured data text search
    pass


class birth_control_shot(PtFeatureBase):
    # structured data text search
    pass


class birth_control_patch(PtFeatureBase):
    # structured data text search
    pass


class birth_control_ring(PtFeatureBase):
    # structured data text search
    pass


class birth_control_pill(PtFeatureBase):
    # structured data text search
    pass


class carcinogen_arsenic(PtFeatureBase):
    pass


class carcinogen_asbestos(PtFeatureBase):
    pass


class carcinogen_benzene(PtFeatureBase):
    pass


class carcinogen_beryllium(PtFeatureBase):
    pass


class carcinogen_cadmium(PtFeatureBase):
    pass


class carcinogen_chromium(PtFeatureBase):
    pass


class carcinogen_ethylene_oxide(PtFeatureBase):
    pass


class carcinogen_nickel(PtFeatureBase):
    pass


class carcinogen_radon(PtFeatureBase):
    pass


class carcinogen_vinyl_chloride(PtFeatureBase):
    pass


class carcinogen_smoke(PtFeatureBase):
    pass


class carcinogen_gasoline(PtFeatureBase):
    pass


class carcinogen_formaldehyde(PtFeatureBase):
    pass


class carcinogen_hair_dyes(PtFeatureBase):
    pass


class carcinogen_soot(PtFeatureBase):
    pass


class cancer_cancer(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        # cancer_df = get_rows_with_cancer(df)
        cancer_df = get_rows_by_icd(df, cancer_icd9s, cancer_icd10s)
        if cancer_df.empty:
            return None
        return True

    @classmethod
    def query(cls, **kwargs):
        """Return the query for cancer history."""
        return f"Does this medical record indicate that the patient has a history of cancer? Only consider tumors as cancer if they are malignant. {NOT_CANCER_STR} {FAMILY_HISTORY_STR} Options are: A. Yes, B. No"

    options = ["A", "B"]
    keywords = [
        "cancer",
        "carcinoma",
        "melanoma",
        "mesothelioma",
        "sarcoma",
        "lymphoma",
        "leukemia",
        "myeloma",
        "malignant",
        "tumor",
        "myelodysplastic",
    ]
    val_var = True

    def pooling_fn(preds: list):
        if "A" in preds:
            return "A"
        return "B"


class cancer_date_of_diagnosis(PtDateFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        cancer_df = get_rows_by_icd(df, cancer_icd9s, cancer_icd10s)
        if cancer_df.empty:
            return None
        dates = pd.to_datetime(cancer_df["Date"], format="%m/%d/%Y")
        return dates.min()

    @classmethod
    def query(cls, **kwargs):
        """Return the query for cancer diagnosis date."""
        return f"What was the date of the patient's most recent {kwargs['keyword']} diagnosis? If the patient has cancer but no diagnosis date is specified, return U. If the record gives no indication of cancer, return X {DATE_INTERPOLATION_STR}"

    keywords = SPECIFIC_CANCERS
    synthetic_keywords = CANCER_STAGE_SYNTHETIC_KEYWORDS
    val_var = True
    pooling_fn = PtDateFeatureBase.pooling_fn_earliest


class cancer_stage_at_diagnosis(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for cancer stage at diagnosis."""
        return f"What was the stage of the patient's most recent {kwargs['keyword']} diagnosis? {NOT_CANCER_STR} {FAMILY_HISTORY_STR} Stage 0 is defined as in situ, non-invasive, or carcinoma in situ. Only answer stage 0 if one of these terms is explicitly mentioned in the diagnosis. Stage I is defined as localized. Stage II and III are defined as locally advanced. Stage IV is defined as metastatic. In-situ or non-invasive melanoma are stage 0. <=2mm melanoma are stage I. >2mm melanoma are stage II or III. Options are: A. Stage 0, B. Stage I, C. Stage II or III, D. Stage IV, E. Patient has cancer but stage unknown, F. Patient does not have cancer."

    options = ["A", "B", "C", "D", "E", "F"]
    keywords = SPECIFIC_CANCERS
    synthetic_keywords = CANCER_STAGE_SYNTHETIC_KEYWORDS

    val_var = True

    def pooling_fn(preds: list):
        # return latest stage
        if "D" in preds:
            return "D"
        if "C" in preds:
            return "C"
        if "B" in preds:
            return "B"
        if "A" in preds:
            return "A"
        if "F" in preds:
            return "F"
        return "E"


class cancer_maximum_stage(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for cancer maximum stage."""
        return f"What was the maximum stage of the patient's most recent {kwargs['keyword']} diagnosis? {NOT_CANCER_STR} {FAMILY_HISTORY_STR} Stage 0 is defined as in situ, non-invasive, or carcinoma in situ. Stage I is defined as localized. Stage II and III are defined as locally advanced. Stage IV is defined as metastatic. Options are: A. Stage 0, B. Stage I, C. Stage II or III, D. Stage IV, E. Cancer present but maximum stage unknown, F. Patient does not have cancer"

    options = ["A", "B", "C", "D", "E", "F"]
    keywords = SPECIFIC_CANCERS
    synthetic_keywords = CANCER_STAGE_SYNTHETIC_KEYWORDS
    val_var = True

    def pooling_fn(preds: list):
        # return latest stage
        if "D" in preds:
            return "D"
        if "C" in preds:
            return "C"
        if "B" in preds:
            return "B"
        if "A" in preds:
            return "A"
        if "F" in preds:
            return "F"
        return "E"


### decided we dont need this feature

# class cancer_status_at_last_follow_up(PtFeatureBase):
#     @classmethod
#     def query(cls, **kwargs):
#         """Return the query for cancer status at last follow-up."""
#         # return f"What was the status of the patient's {kwargs['keyword']} at their last follow-up? {NOT_CANCER_STR} {FAMILY_HISTORY_STR} Assume resection is clearance (cancer free) unless otherwise stated. Options are: A. Having previously had cancer, now they are cancer free, in remission, complete response, no evidence of disease, or disease free, B. Stable disease, C. Progressive disease, D. Cancer present but status at last follow-up unknown, E. No indication of patient's cancer status, F. Patient has never had cancer" # E+F. No indication that patient has cancer
#         return f"What was the status of the patient's {kwargs['keyword']} according the latest available information? {NOT_CANCER_STR} {FAMILY_HISTORY_STR} Assume resection is clearance (cancer free) unless otherwise stated. Options are: A. Having previously had cancer, now they are cancer free, in remission, complete response, no evidence of disease, or disease free, B. Stable disease, C. Progressive disease, D. No indication that the patient has had cancer"
#         # example chart: The patient was diagnosed with progressive cancer in 2004.

#     options = ["A", "B", "C", "D"]
#     keywords = SPECIFIC_CANCERS
#     synthetic_keywords = [
#         "cancer free",
#         "in remission",
#         "no evidence of disease",
#         "metastasis",
#         "stable disease",
#         "progressive disease",
#         "no spread",
#     ]
#     val_var = True

#     def pooling_fn(preds: list):
#         # return the highest status seen
#         if "C" in preds:
#             return "C"
#         if "B" in preds:
#             return "B"
#         if "A" in preds:
#             return "A"
#         if "D" in preds:
#             return "D"
#         if "F" in preds:
#             return "F"
#         return "E"


class cancer_date_free(PtDateFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for cancer free date."""
        return f"What is the date the patient was declared cancer free of {kwargs['keyword']}? {NOT_CANCER_STR} {FAMILY_HISTORY_STR} Cancer free can be defined as: cancer free, in remission, complete response, no evidence of disease, or disease free. If the patient had cancer but no cancer free date is specified, return U. If the record gives no indication of cancer, return X {DATE_INTERPOLATION_STR}"

    keywords = SPECIFIC_CANCERS
    synthetic_keywords = [
        "cancer free",
        "in remission",
        "no evidence of disease",
    ]
    val_var = True
    pooling_fn = PtDateFeatureBase.pooling_fn_latest


class cancer_treatment_radiation(PtFeatureBase):
    pass


class cancer_treatment_radiation(PtFeatureBase):
    pass


class cancer_treatment_chemo(PtFeatureBase):
    pass


class cancer_treatment_other(PtFeatureBase):
    pass


class cancer_family_any(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        """Return the query for family cancer history."""
        return f"Does this medical record indicate that any member of the patient's family has a history of cancer? {NOT_CANCER_STR} Options are: A. Yes, B. No"

    options = ["A", "B"]
    keywords = cancer_cancer.keywords
    val_var = True


class cancer_family_cancer_type(PtFeatureBase):
    pass


class cancer_family_mother(PtFeatureBase):
    pass


class cancer_family_father(PtFeatureBase):
    pass


class cancer_family_brother(PtFeatureBase):
    pass


class cancer_family_sister(PtFeatureBase):
    pass


class cancer_family_paternal_grandparent(PtFeatureBase):
    pass


class cancer_family_maternal_grandparent(PtFeatureBase):
    pass


class cancer_family_cancer(PtFeatureBase):
    pass


class medical_radiation_exposure_type(PtFeatureBase):
    pass


class medical_radiation_exposure_date(PtFeatureBase):
    pass


class hereditary_li_fraumeni_syndrome(PtFeatureBase):
    # ICD-10: Z15.0 = genetic susceptibility to malignant neoplasm, Z15.01 = susceptibility to malignant neoplasm of breast, Z15.02 = genetic susceptibility to malignant neoplasm of ovary, Z15.03 = genetic susceptibility to malignant neoplasm of prostate, Z15.04 = genetic susceptibility to malignant neoplasm of endometrium. ICD-9: V84.0 = genetic susceptibility to malignant neoplasm, V84.01 = genetic susceptibility to malignant neoplasm of breast, V84.02 = genetic susceptibility to malignant neoplasm of ovary, V84.03 = genetic susceptibility to malignant neoplasm of prostate, V84.04 = genetic susceptibility to malignant neoplasm of endometrium, V84.09 = genetic susceptibility to other malignant neoplasm
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["Z15.0", "Z15.01", "Z15.02", "Z15.03", "Z15.04"]
        icd9s = ["V84.0", "V84.01", "V84.02", "V84.03", "V84.04", "V84.09"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class hereditary_gastrointestinal_stromal_tumors(PtFeatureBase):
    pass


class hereditary_paraganglioma_pheochromocytoma_syndrome(PtFeatureBase):
    pass


class hereditary_gi_cancer_any(PtFeatureBase):
    pass


class hereditary_gi_cancer_lynch_syndrome(PtFeatureBase):
    pass


class hereditary_gi_cancer_polyposis_syndromes(PtFeatureBase):
    pass


class hereditary_gi_cancer_peutz_jeghers_syndrome(PtFeatureBase):
    pass


class hereditary_gi_cancer_myh_associated_polyposis(PtFeatureBase):
    pass


class hereditary_gi_cancer_pancreatic_cancer(PtFeatureBase):
    pass


class hereditary_brca1_mutation(PtFeatureBase):
    pass


class hereditary_brca2_mutation(PtFeatureBase):
    pass


class hereditary_von_hippel_lindau_disease(PtFeatureBase):
    pass


class hereditary_cowden_syndrome(PtFeatureBase):
    pass


class infection_history_hpv_human_papillomavirus(PtFeatureBase):
    @staticmethod
    # icd10: B97.7, R87.81, R87.82, R85.81, R85.82, B07
    # icd9:  079.4, 078.1, 796.75, 796.05, 796.15, 796.79, 795.09, 795.19
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["B97.7", "R87.81", "R87.82", "R85.81", "R85.82"]
        icd9s = [
            "079.4",
            "078.1",
            "796.75",
            "796.05",
            "796.15",
            "796.79",
            "795.09",
            "795.19",
        ]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_ebv_epstein_barr_virus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        # icd10: B27
        # icd9:  075
        df = dfs["Dia"]
        icd10s = ["B27"]
        icd9s = ["075"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_hbv_hepatitis_b_virus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        # icd10: B16, B18.0, B18.1
        # icd9:  070.3, 070.2
        df = dfs["Dia"]
        icd10s = ["B16", "B18.0", "B18.1"]
        icd9s = ["070.3", "070.2"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_hcv_hepatitis_c_virus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        # icd10: B18.2
        # icd9:  070.41, 070.44, 070.51, 070.54, 070.7
        df = dfs["Dia"]
        icd10s = ["B18.2"]
        icd9s = ["070.41", "070.44", "070.51", "070.54", "070.7"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_hiv_human_immunodeficiency_virus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["B20", "Z21"]
        icd9s = ["V08", "042"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_hhv_8_kaposis_sarcoma_associated_herpesvirus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["B10.89", "C46"]
        icd9s = ["058.89", "176"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_htlv_1_human_t_lymphotropic_virus_1(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["B97.33", "Z22.6"]
        icd9s = ["079.51"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


# class infection_history_mcv_merkel_cell_polyomavirus(PtFeatureBase):
#     @staticmethod
#     def compute(dfs: dict):
#         pass


class infection_history_covid_19(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["U07.1"]
        icd9s = []
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_helicobacter_pylori(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["B96.81"]
        icd9s = ["041.86"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_chlamydia_trachomatis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["A74", "A56"]
        icd9s = ["079.98", "099.41"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_salmonella_typhi(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["A01.0"]
        icd9s = ["002.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


# missing from sheet
# class infection_history_streptococcus_bovis_gallolyticus(PtFeatureBase):
#     @staticmethod
#     def compute(dfs: dict):
#         pass


class infection_history_mycobacterium_tuberculosis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["A15", "A16", "A17", "A18", "A19"]
        icd9s = ["010", "011", "012", "013", "014", "015", "016", "017", "018"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


# class infection_history_nontuberculous_mycobacteria(PtFeatureBase):
#     @staticmethod
#     def compute(dfs: dict):
#         pass


class infection_history_opisthorchis_viverrini(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["B66.0"]
        icd9s = ["121.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_schistosoma_hematobium(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["B65.0"]
        icd9s = ["120.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_aspergillus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["B44"]
        icd9s = ["117.3"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_uti_any(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["Z87.440", "N39.0"]
        icd9s = ["599.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_uti_cystitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["N30"]
        icd9s = ["595"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_uti_pyelonephritis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["N10"]
        icd9s = ["590"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_uti_urethritis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["N34"]
        icd9s = ["597"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class infection_history_uti_count(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["Z87.440", "N39.0", "N30", "N10", "N34"]
        icd9s = ["599.0", "595", "590", "597"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        dates = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        unique_dates = dates.unique()
        return unique_dates.size


class hrt(PtFeatureBase):
    # structured data text search
    pass


class hrt_type(PtFeatureBase):
    pass


class hrt_duration(PtFeatureBase):
    pass


class hrt_pre_post_menopause(PtFeatureBase):
    pass


class autoimmune_rheumatoid_arthritis_seropositive(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["M05"]
        icd9s = ["714.0", " 714.1", "714.2"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


# class autoimmune_psoriasis(PtFeatureBase):
#     @staticmethod
#     def compute(dfs: dict):
#         pass


class autoimmune_type_1_diabetes(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["E10"]
        icd9s = [
            "250.01",
            "250.03",
            "250.11",
            "250.13",
            "250.21",
            "250.23",
            "250.31",
            "250.33",
            "250.41",
            "250.43",
            "250.51",
            "250.53",
            "250.61",
            "250.63",
            "250.71",
            "250.73",
            "250.81",
            "250.83",
            "250.91",
            "250.93",
        ]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_hashimotos_thyroiditis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["E06.3"]
        icd9s = ["245.2"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_graves_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["E05.00"]
        icd9s = ["242.00"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_multiple_sclerosis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["G35"]
        icd9s = ["340"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_systemic_lupus_erythematosus_sle(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["M32"]
        icd9s = ["710.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_celiac_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["K90.0"]
        icd9s = ["579.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


# class autoimmune_inflammatory_bowel_disease_ulcerative_colitis(PtFeatureBase):
#     @staticmethod
#     def compute(dfs: dict):
#         pass


# class autoimmune_inflammatory_bowel_disease_crohns_disease(PtFeatureBase):
#     @staticmethod
#     def compute(dfs: dict):
#         pass


class autoimmune_sjogrens_syndrome_sicca_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["M35.0"]
        icd9s = ["710.2"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_vitiligo(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L80"]
        icd9s = ["709.01"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_polymyalgia_rheumatica(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["M35.3"]
        icd9s = ["725"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_ankylosing_spondylitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["M45"]
        icd9s = ["720.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_pernicious_anemia(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["D51.0"]
        icd9s = ["281.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_autoimmune_hepatitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["K75.4"]
        icd9s = ["571.42"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_autoimmune_adrenalitis_addison_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["E27.1"]
        icd9s = ["255.41"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


# Alopecia areata is already noted as included in comorbid data,
# but the original code has a separate class, so we fill it in:
class autoimmune_alopecia_areata(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L63"]
        icd9s = ["704.01"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_guillain_barre_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["G61.0"]
        icd9s = []
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_goodpasture_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["M31.0"]
        icd9s = ["446.21"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_primary_biliary_cirrhosis_primary_biliary_cholangitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["K74.3"]
        icd9s = ["571.6"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_behcets_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["M35.2"]
        icd9s = ["136.1"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_systemic_sclerosis_scleroderma(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["M34"]
        icd9s = ["710.1"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class autoimmune_myasthenia_gravis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["G70.0"]
        icd9s = ["358.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_crohns_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["K50"]
        icd9s = ["555"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_ulcerative_colitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["K51"]
        icd9s = ["556"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_kidney_stones(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["N20.0"]
        icd9s = ["592.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_kidney_disease_acute_chronic_unspecified(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        # ICD-10 codes for acute, chronic, and unspecified kidney disease
        icd10s = ["N17", "N18", "N19"]
        # ICD-9 codes for acute (584), chronic (585), and unspecified (586)
        icd9s = ["584", "585", "586"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_chronic_pancreatitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["K86.0", "K86.1"]
        icd9s = ["577.1"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_dense_breast_tissue(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["R92.30"]
        icd9s = ["793.82"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_polycystic_ovarian_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["E28.2"]
        icd9s = ["256.4"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_type_2_diabetes_mellitus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["E11"]
        icd9s = [
            "250.00",
            "250.02",
            "250.10",
            "250.12",
            "250.20",
            "250.22",
            "250.30",
            "250.32",
            "250.40",
            "250.42",
            "250.50",
            "250.52",
            "250.60",
            "250.62",
            "250.70",
            "250.72",
            "250.80",
            "250.82",
            "250.90",
            "250.92",
        ]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_endometrial_polyps(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["N84.0"]
        icd9s = ["621"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_colonic_polyps_adenomatous_and_serrated(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["Z83.710"]
        icd9s = ["V12.72"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_albinism(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["E70.30"]
        icd9s = ["270.2"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_chronic_skin_inflammation_eczema(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L20"]
        # No ICD-9 provided in the table
        icd9s = []
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_chronic_skin_inflammation_psoriasis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L40"]
        icd9s = ["696.1"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_chronic_skin_inflammation_alopecia_areata(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L63"]
        icd9s = ["704.01"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_chronic_skin_inflammation_rosacea(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L71"]
        icd9s = ["695.3"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_chronic_skin_inflammation_lichen_planus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L43"]
        icd9s = ["697.0"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_chronic_skin_inflammation_hidradenitis_suppurativa(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L73.2"]
        icd9s = ["705.83"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class comorbid_chronic_skin_inflammation_chronic_skin_ulcer_excluding_pressure_ulcer(
    PtFeatureBase
):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L98.4"]
        icd9s = ["707.1", "707.8", "707.9"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


# # No ICD codes provided for 'wounds'; left as pass
# class comorbid_chronic_skin_inflammation_wounds(PtFeatureBase):
#     @staticmethod
#     def compute(dfs: dict):
#         pass


class comorbid_chronic_skin_inflammation_burns(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["T20", "T21", "T22", "T23", "T24", "T25", "T30", "T31"]
        icd9s = ["940", "941", "942", "943", "944", "945", "946", "948"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class radiodermatitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L58"]
        icd9s = ["692.82"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class sunburn_first_degree(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L55.0, L55.9"]
        icd9s = ["692.71"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class sunburn_blistering_second_or_third_degree(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L55.1, L55.2"]
        icd9s = ["692.76, 692.77"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


class exposure_to_tanning_bed_and_other_manmade_visible_or_uv_light(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["W89"]
        icd9s = ["E926.2"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


# class tanning(PtFeatureBase):
#     @staticmethod
#     def compute(dfs: dict):
#         pass


class antibiotics(PtFeatureBase):
    keywords = [
        "tetracycline",
        "doxycycline",
        "minocycline",
        "adoxa",
        "adoxa pak",
        "brodspec",
        "cleeravue-m",
        "declomycin",
        "doryx",
        "dynacin",
        "minocin",
        "nuzyra",
        "sumycin",
        "vibramycin calcium",
        "tmp",
        "smx",
        "tmp/smx",
        "tmp-smx",
        "trimethoprim sulfamethoxazole",
        "bactrim",
        "septra",
        "smz-tmp",
        "sulfatrim",
        "co-trimoxazole",
        "sxt",
        "tmp-sulfa",
        "amoxicot",
        "amoxil",
        "dispermox",
        "moxatag",
        "moxilin",
        "trimox",
        "amoxicillin",
        "cephalexin",
        "bio-cef",
        "keflex",
        "panixine disperdose",
        "azithromycin",
        "zithromax",
        "zithromax tri-pak",
        "z-pak",
        "zmax",
    ]

    @classmethod
    def query(cls, **kwargs):
        """Return the query for antibiotics usage."""
        return f"Does this medical record indicate that the patient took any of the following antibiotics, ignoring those mentioned as allergic reactions: {', '.join(cls.keywords)}? Options are: A. Yes, B. No" # todo: distinguis amoxicillin vs amoxicillin clavulanate, doxy 

    options = ["A", "B"]
    val_var = True

    def pooling_fn(preds: list):
        if "A" in preds:
            return "A"
        return "B"


class antibiotic_duration(PtFeatureBase):
    @classmethod
    def query(cls, **kwargs):
        return """
# antibiotic_duration
## Query
Query function: 

How long did the patient take the {abx}? Round all values up to the nearest integer (i.e. 0.1 -> 1). A. 0 days (no indication of antibiotic use), B. 1-15 days, C. 16-45 days, D. 46-135 days, E. 136+ days, F. Taken but dates unknown 

### Step 1. Did the patient take the antibiotic?

Ask:
**"Did the patient take {keyword}? If the antibiotic is listed as an allergy, consider as 'no indication of antibiotic use'. Options are A. yes, B. no"**

* If the answer is **(yes)** → continue to Step 2.
* If the answer is **(no)** → set the final label to **A ("no indication of antibiotic use")** and stop.

---

### Step 2. How many days in total did the patient take the antibiotic for?

Ask:
**"How many days total did the patient take {keyword}? Answer according to the total duration explicitly written in the note (i.e. 5 days, 7 days, 1 week, 1 month). For example, "patient took {keyword} for 5 days" = 5 total days or "1 week course of {keyword}" = 7 total days or "1 month course of {keyword}" = 30 days. If it states that the patient took the pill as needed, PRN, or is prescribed to be taken under hypothetical future scenarios (i.e. IF the patient experiences traveler's diarrhea), then answer 'F' (patient took antibiotic but duration unknown). If there is not enough information, answer 'NA'. Assume 30 days per month. Answer with a number (of days).**

* If a valid **number of days** is given → use this number as the total duration (final answer)
* If the answer is **"NA"** or can’t be determined → move on to step 3. 

---

### Step 3. If the total duration is NOT explicitly written somewhere in the note, estimate duration from pill counts, number of refills, and the frequency of the dose. 

Ask:
**"How many days total did the patient take {keyword}? If the total duration is NOT explicitly written somewhere in the note, calculate the duration based on the number of pills prescribed, the number of refills, and the frequency of the dose. For example, if the patient was prescribed 10 pills, and the frequency is 2 pills per day, the total days of medication taken by the patient are 5 days. As another example, if the patient was prescribed 20 pills, had 2 refills, and took 3 pills per day, the total days of medication taken by the patient are 20 days. DO NOT assume the patient takes one pill per day. Note that 'refills' are in addition to the initial prescription. For example, 2 refills of 10 pills = 30 pills total. Note that BID is an abreviation meaning "twice a day". When calculating total days, DO NOT include days that the patient did not take the medication. For example, if the patient took the medication once a week for 4 weeks, the patient took the medication for 4 days. If there is not enough information, answer 'NA'. Assume 30 days per month. Answer with a number (of days).**

* If a valid **pill count**, **number of refills**, and **frequency of the dose** is given → calculate the total duration (final answer)
* If the answer is **"NA"** or can’t be determined → move on to step 4.

---

### Step 4. If the pill counts are available in the note, but there is NOT information on the number of refills and/or the frequency of the dose, then it is OK to estimate duration from the pill counts alone

Ask:
**"How many days total did the patient take {keyword}? If the total duration is NOT explicitly written somewhere in the note, AND the pill counts are available in the note, but there is not information in the note on frequency of dose and/or number of refills for {keyword}, then we should calculate the total days by assuming one pill was taken each day. For example, if patient received a prescription for 28 pills of {keyword}, then we should assume they took a 28 day course of {keyword}. If it states that the patient took the pill as needed, PRN, or is prescribed to be taken under hypothetical future scenarios (i.e. IF the patient experiences traveler's diarrhea), then answer 'F' (patient took antibiotic but duration unknown). If there is not enough information, answer 'NA'. Assume 30 days per month. Answer with a number (of days).**


* If a valid **number of days** is given → use this number as the total duration (final answer)
* If the answer is **"NA"**, then the total duration remains unknown.

---

### Step 5. If the total duration is not explicitly written somewhere in the note AND the pill counts, number of refills and the frequency of the dose are not provided, then answer 'NA'. Answer with a number (of days). Do NOT use start and/or end dates to estimate total duration"** 

---

### Step 6. Interpret the duration

* If the calculated duration (**days on antibiotic**) is missing, negative, or unrealistically large (>1,000,000 days):
  set label to **F ("dates unknown")**.

* Otherwise, assign labels based on the duration:

  * **≤15 days** → label **B**
  * **16–45 days** → label **C**
  * **46–135 days** → label **D**
  * **>135 days** → label **E**

---

### Final Output

For each patient, record:

* Start date (if known)
* End date (if known)
* Days on antibiotic (if calculable)
* Whether they took the antibiotic (**yes/no**)
* The final label (**A–G**, as determined above)

---


Example query: None
## Keywords
['tetracycline', 'doxycycline', 'minocycline', 'adoxa', 'adoxa pak', 'brodspec', 'cleeravue-m', 'declomycin', 'doryx', 'dynacin', 'minocin', 'nuzyra', 'sumycin', 'vibramycin calcium', 'tmp', 'smx', 'tmp/smx', 'tmp-smx', 'trimethoprim sulfamethoxazole', 'bactrim', 'septra', 'smz-tmp', 'sulfatrim', 'co-trimoxazole', 'sxt', 'tmp-sulfa', 'amoxicot', 'amoxil', 'dispermox', 'moxatag', 'moxilin', 'trimox', 'amoxicillin', 'cephalexin', 'bio-cef', 'keflex', 'panixine disperdose', 'azithromycin', 'zithromax', 'zithromax tri-pak', 'z-pak', 'zmax']
## Number of records with keyword: 100
## Number of records: 100
## Number of chunks: 339


    """

    keywords = antibiotics.keywords
    val_var = True

    @classmethod
    def forward(
        cls, model: MrModel, chunk: str, keyword: str, inference_type="logit", **kwargs
    ):
        keyword = KEYWORD_ADDITIONAL_INFO[keyword] or keyword
        ANSWER_OPTIONS_STR = "A. 0 days (no indication of antibiotic use), B. 1-15 days, C. 16-45 days, D. 46-135 days, E. 136+ days, F. Taken but dates unknown"

        # Define modular question functions
        def _ask_took():
            """Ask if the patient took the antibiotic."""
            took_q = f"Did the patient take {keyword}?\n\n{ENTRY_FORMAT_STR}\n\nEdge cases: If the patient discontinued, stopped taking, or previously took the antibiotic, consider it `yes`. If the antibiotic is listed as an allergy, consider as `no`. Options are A. yes, B. no"
            options = ["A", "B"]
            took_history = model.format_chunk_qs(q=took_q, chunk=chunk, options=options)
            if inference_type == "cot":
                pred = model.predict_with_cot(
                    took_history, options=options, max_answer_tokens=1, sample=True
                )
                return pred
            else:
                return model.predict_single_with_logit_trick(
                    took_history, output_choices=["A", "B"]
                )

        def _ask_days_on_explicit_duration():
            """Ask how many days the patient was on the antibiotic based on the total duration explicitly written in the note."""
            count_doses_q = f"""How many days total did the patient take {keyword}?\n\n{ENTRY_FORMAT_STR}\n\nAnswer according to the total duration explicitly written in the note (i.e. 5 days, 7 days, 1 week, 1 month). For example, "patient took {keyword} for 5 days" = 5 total days or "1 week course of {keyword}" = 7 total days or "1 month course of {keyword}" = 30 days.If it states that the patient took the pill as needed, PRN, or is prescribed to be taken under hypothetical future scenarios (i.e. IF the patient experiences traveler's diarrhea), then answer 'F' (patient took antibiotic but duration unknown). Assume 30 days per month. Answer one of: {ANSWER_OPTIONS_STR}, stating only the letter of the answer. """
            count_doses_history = model.format_chunk_qs(
                q=count_doses_q, chunk=chunk, options=["A", "B", "C", "D", "E", "F"]
            )
            if inference_type == "cot":
                return model.predict_with_cot(
                    count_doses_history,
                    options=["A", "B", "C", "D", "E", "F"],
                    max_answer_tokens=1,
                    sample=True,
                )
            else:
                return model.predict_single_with_logit_trick(
                    count_doses_history, output_choices=["A", "B", "C", "D", "E", "F"]
                )
            # return retry_with_validation(model, count_doses_history, _count_validation)

        def _ask_based_on_pill_count():
            """Ask how many days the patient was on the antibiotic based on the number of pills prescribed, number of refills, and the frequency of the dose."""
            count_doses_q = f"""How many days total did the patient take {keyword}?\n\n{ENTRY_FORMAT_STR}\n\nCalculate the duration based on the number of pills prescribed, the number of refills, and the frequency of the dose. For example, if the patient was prescribed 10 pills, and the frequency is 2 pills per day (aka BID), the total days of medication taken by the patient are 5 days. As another example, if the patient was prescribed 20 pills, had 2 refills, and took 3 pills per day, the total days of medication taken by the patient are 20 days. If no pill count is given, assume one pill was taken each day. Note that 'refills' are in addition to the initial prescription. For example, 2 refills of 10 pills = 30 pills total. When calculating total days, DO NOT include days that the patient did not take the medication. For example, if the patient took the medication once a week for 4 weeks, the patient took the medication for 4 days. If there is not enough information, answer `F`. Assume 30 days per month. Answer one of: {ANSWER_OPTIONS_STR}, stating only the letter of the answer. """
            count_doses_history = model.format_chunk_qs(
                q=count_doses_q, chunk=chunk, options=["A", "B", "C", "D", "E", "F"]
            )
            if inference_type == "cot":
                return model.predict_with_cot(
                    count_doses_history,
                    options=["A", "B", "C", "D", "E", "F"],
                    max_answer_tokens=1,
                    sample=True,
                )
            else:
                return model.predict_single_with_logit_trick(
                    count_doses_history, output_choices=["A", "B", "C", "D", "E", "F"]
                )

        took_pred = _ask_took()
        if took_pred == "B":
            return {
                "took": took_pred,
                "pred": "A",
                "days_on_explicit_duration": None,
                "days_on_pill_count": None,
            }
        days_on_explicit_duration_pred = _ask_days_on_explicit_duration()
        if days_on_explicit_duration_pred != "F":
            return {
                "took": took_pred,
                "pred": days_on_explicit_duration_pred,
                "days_on_explicit_duration": days_on_explicit_duration_pred,
                "days_on_pill_count": None,
            }
        days_on_pill_count_pred = _ask_based_on_pill_count()
        if days_on_pill_count_pred != "F":
            return {
                "took": took_pred,
                "pred": days_on_pill_count_pred,
                "days_on_explicit_duration": days_on_explicit_duration_pred,
                "days_on_pill_count": days_on_pill_count_pred,
            }
        return {
            "took": took_pred,
            "pred": "F",
            "days_on_explicit_duration": days_on_explicit_duration_pred,
            "days_on_pill_count": days_on_pill_count_pred,
        }

    def pooling_fn(preds: list):
        # Get the highest duration listed
        if "E" in preds:
            return "E"
        if "D" in preds:
            return "D"
        if "C" in preds:
            return "C"
        if "B" in preds:
            return "B"
        if "A" in preds:
            return "A"
        if "F" in preds:
            return "F"
        return "G"


class antibiotic_tetracyclines(PtFeatureBase):
    pass


class antibiotic_tmp_smx(PtFeatureBase):
    pass


class antibiotic_amoxicillin(PtFeatureBase):
    pass


class antibiotic_cephalexin(PtFeatureBase):
    pass


class antibiotic_azithromycin(PtFeatureBase):
    pass
