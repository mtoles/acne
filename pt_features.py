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
        smoking_statuses = df[df["Concept_Name"]=="Smoking status"]["Result"]
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
        #"140-149  Malignant Neoplasm Of Lip, Oral Cavity, And Pharynx
        # 150-159  Malignant Neoplasm Of Digestive Organs And Peritoneum
        # 160-165  Malignant Neoplasm Of Respiratory And Intrathoracic Organs
        # 170-176  Malignant Neoplasm Of Bone, Connective Tissue, Skin, And Breast
        # 179-189  Malignant Neoplasm Of Genitourinary Organs
        # 190-199  Malignant Neoplasm Of Other And Unspecified Sites
        # 200-209  Malignant Neoplasm Of Lymphatic And Hematopoietic Tissue http://www.icd9data.com/2015/Volume1/140-239/default.htm "
        cancer_ic9s = [i for sublist in [
            list(range(140, 150)),
            list(range(150, 160)),
            list(range(160, 166)),
            list(range(170, 177)),
            list(range(179, 190)),
            list(range(190, 200)),
            list(range(200, 210))
        ] for i in sublist]
        # C00-C14  Malignant neoplasms of lip, oral cavity and pharynx
        # C15-C26  Malignant neoplasms of digestive organs
        # C30-C39  Malignant neoplasms of respiratory and intrathoracic organs
        # C40-C41  Malignant neoplasms of bone and articular cartilage
        # C43-C44  Melanoma and other malignant neoplasms of skin
        # C45-C49  Malignant neoplasms of mesothelial and soft tissue
        # C50-C50  Malignant neoplasms of breast
        # C51-C58  Malignant neoplasms of female genital organs
        # C60-C63  Malignant neoplasms of male genital organs
        # C64-C68  Malignant neoplasms of urinary tract
        # C69-C72  Malignant neoplasms of eye, brain and other parts of central nervous system
        # C73-C75  Malignant neoplasms of thyroid and other endocrine glands
        # C76-C80  Malignant neoplasms of ill-defined, other secondary and unspecified sites
        # C7A-C7A  Malignant neuroendocrine tumors
        # C7B-C7B  Secondary neuroendocrine tumors
        # C81-C96  Malignant neoplasms of lymphoid, hematopoietic and related tissue https://www.icd10data.com/ICD10CM/Codes/C00-D49
        # fmt: off
        cancer_ic10s = [
            "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14",
            "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
            "C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39",
            "C40", "C41",
            "C43", "C44",
            "C45", "C46", "C47", "C48", "C49",
            "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58",
            "C60", "C61", "C62", "C63",
            "C64", "C65", "C66", "C67", "C68",
            "C69", "C70", "C71", "C72",
            "C73", "C74", "C75",
            "C76", "C77", "C78", "C79", "C80",
            "C7A", "C7B",
            "C81", "C82", "C83", "C84", "C85", "C86", "C87", "C88", "C89", "C90", "C91", "C92", "C93", "C94", "C95", "C96"
        ]
        had_cancer_icd9 = df["Code"].isin(cancer_ic9s)
        had_cancer_icd10 = df["Code"].apply(lambda x: x[:3] in cancer_ic10s)
        cancer_df = df[had_cancer_icd9 | had_cancer_icd10]
        if cancer_df.empty:
            return None
        return True
        # fmt: on
        pass


class cancer_cancer_date_of_diagnosis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


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
    @staticmethod
    def compute(dfs: dict):
        pass


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
    def compute(dfs: dict):
        pass


class infection_history_ebv_epstein_barr_virus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_hbv_hepatitis_b_virus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_hcv_hepatitis_c_virus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_hiv_human_immunodeficiency_virus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_hhv_8_kaposis_sarcoma_associated_herpesvirus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_htlv_1_human_t_lymphotropic_virus_1(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_mcv_merkel_cell_polyomavirus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_covid_19(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_helicobacter_pylori(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_chlamydia_trachomatis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_salmonella_typhi(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_streptococcus_bovis_gallolyticus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_mycobacterium_tuberculosis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_nontuberculous_mycobacteria(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_opisthorchis_viverrini(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_schistosoma_hematobium(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_aspergillus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_uti_any(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_uti_cystitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_uti_pyelonephritis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_uti_urethritis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class infection_history_uti_count(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


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
        pass


class autoimmune_psoriasis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_type_1_diabetes(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_hashimotos_thyroiditis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_graves_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_multiple_sclerosis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_systemic_lupus_erythematosus_sle(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_celiac_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_inflammatory_bowel_disease_ulcerative_colitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_inflammatory_bowel_disease_crohns_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_sjogrens_syndrome_sicca_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_vitiligo(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_polymyalgia_rheumatica(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_ankylosing_spondylitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_pernicious_anemia(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_autoimmune_hepatitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_autoimmune_adrenalitis_addison_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_alopecia_areata(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_guillain_barre_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_goodpasture_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_primary_biliary_cirrhosis_primary_biliary_cholangitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_behcets_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_systemic_sclerosis_scleroderma(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class autoimmune_myasthenia_gravis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_crohns_disease(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_ulcerative_colitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_kidney_stones(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_kidney_disease_acute_chronic_unspecified(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_pancreatitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_dense_breast_tissue(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_polycystic_ovarian_syndrome(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_type_2_diabetes_mellitus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_endometrial_polyps(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_colonic_polyps_adenomatous_and_serrated(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_albinism(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_skin_inflammation_eczema(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_skin_inflammation_psoriasis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_skin_inflammation_alopecia_areata(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_skin_inflammation_rosacea(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_skin_inflammation_lichen_planus(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_skin_inflammation_hidradenitis_suppurativa(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_skin_inflammation_chronic_skin_ulcer_excluding_pressure_ulcer(
    PtFeatureBase
):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_skin_inflammation_wounds(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class comorbid_chronic_skin_inflammation_burns(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class radiodermatitis(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class sunburn_first_degree(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class sunburn_blistering_second_or_third_degree(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


class exposure_to_tanning_bed_and_other_manmade_visible_or_uv_light(PtFeatureBase):
    @staticmethod
    def compute(dfs: dict):
        pass


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
