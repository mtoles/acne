import pandas as pd

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
# fmt: on

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
        if name not in ["PtFeatureBase"]:
            PtFeaturesMeta.registry[name] = cls
        return cls


class PtFeatureBase(metaclass=PtFeaturesMeta):
    @staticmethod
    def compute(dfs: dict):
        pass


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


class smoking_amount(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class alcohol_status(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class alcohol_amount_drinking_index(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class alcohol_amount(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class immunosuppressed_transplant_organ_name(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class immunosuppressed_transplant_organ_date(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class immunosuppressed_medication_medication_name(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class immunosuppressed_disease_disease_name(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
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
    @staticmethod
    def compute(dfs: dict):
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


class military_years(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class military_retirement_date(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class military_agent_orange(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class military_oef_oif(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class military_camp_lejeune(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class pregnancy(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class pregnancy_age(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class pregnancy_count(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class early_menstruation(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class menopause(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class menopause_late(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class birth_control_copper_iud(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class birth_control_hormonal_iud(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class birth_control_implant(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class birth_control_shot(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class birth_control_patch(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class birth_control_ring(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class birth_control_pill(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_arsenic(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_asbestos(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_benzene(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_beryllium(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_cadmium(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_chromium(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_ethylene_oxide(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_nickel(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_radon(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_vinyl_chloride(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_smoke(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_gasoline(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_formaldehyde(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_hair_dyes(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class carcinogen_soot(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
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


class cancer_cancer_date_of_diagnosis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        cancer_df = get_rows_by_icd(df, cancer_icd9s, cancer_icd10s)
        if cancer_df.empty:
            return None
        dates = pd.to_datetime(cancer_df["Date"], format="%m/%d/%Y")
        return dates.min()


class cancer_cancer_stage_at_diagnosis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_cancer_maximum_stage(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_cancer_status_at_last_follow_up(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_treatment_radiation(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_treatment_chemo(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_treatment_other(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_family_mother(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_family_father(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_family_brother(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_family_sister(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_family_paternal_grandparent(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_family_maternal_grandparent(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class cancer_family_cancer(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class medical_radiation_exposure_type(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class medical_radiation_exposure_date(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
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
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_paraganglioma_pheochromocytoma_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_gi_cancer_any(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_gi_cancer_lynch_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_gi_cancer_polyposis_syndromes(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_gi_cancer_peutz_jeghers_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_gi_cancer_myh_associated_polyposis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_gi_cancer_pancreatic_cancer(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_brca1_mutation(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_brca2_mutation(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_von_hippel_lindau_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hereditary_cowden_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
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
        pass


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


class infection_history_streptococcus_bovis_gallolyticus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


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
    @staticmethod
    def compute(dfs: dict):
        pass


class hrt_type(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hrt_duration(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class hrt_pre_post_menopause(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_rheumatoid_arthritis_seropositive(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["M05"]
        icd9s = ["714.0"," 714.1","714.2"]
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
        icd9s = ["250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43", "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93"]
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
            "250.00", "250.02", "250.10", "250.12", "250.20", "250.22", "250.30", "250.32",
            "250.40", "250.42", "250.50", "250.52", "250.60", "250.62", "250.70", "250.72",
            "250.80", "250.82", "250.90", "250.92"
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


class comorbid_chronic_skin_inflammation_chronic_skin_ulcer_excluding_pressure_ulcer(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        df = dfs["Dia"]
        icd10s = ["L98.4"]
        icd9s = ["707.1", "707.8", "707.9"]
        df = get_rows_by_icd(df, icd9s, icd10s)
        return not df.empty


# No ICD codes provided for 'wounds'; left as pass
class comorbid_chronic_skin_inflammation_wounds(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


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


class tanning(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class antibiotics(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class antibiotic_tetracyclines(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class antibiotic_tmp_smx(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class antibiotic_amoxicillin(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class antibiotic_cephalexin(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class antibiotic_azithromycin(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass
