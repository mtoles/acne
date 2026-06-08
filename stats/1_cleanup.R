# title: "Antibiotic Exposure and Cancer Risk - Data Clean Up"
# author: Daniel Kim

################################################################################
## 1. clear all
################################################################################

rm(list = ls(all.names = TRUE))

################################################################################
## 2. set locations
################################################################################

library(tidyverse)
library(gtsummary)
library(survival)
library(survminer)
library(lubridate)
library(smd)

input  <- '/Users/danielkim/Partners HealthCare Dropbox/Daniel Kim/Research/Barbieri'
output <- '/Users/danielkim/Partners HealthCare Dropbox/Daniel Kim/Research/Barbieri/eda_output'

dir.create(output, showWarnings = FALSE)

################################################################################
## 3. data read-in
################################################################################

raw <- read_csv(file.path(input, "pooled_records.csv"),
                col_types   = cols(.default = "c"),
                name_repair = "minimal")

################################################################################
## 4. column inventory
################################################################################

all.cols <- names(raw)

dur.cols            <- grep("^antibiotic_duration_numeric__treatment__", all.cols, value = TRUE)
cancer.outcome.cols <- grep("^cancer_cancer__outcome__",                 all.cols, value = TRUE)
cancer.type.cols    <- grep("^cancer_date_of_diagnosis__outcome__",      all.cols, value = TRUE)
fam.cols            <- grep("^cancer_family_any__pre_index__",           all.cols, value = TRUE)
dis.cols            <- grep("^disease__pre_index__",                     all.cols, value = TRUE)

REAL.ABX      <- c("AMOXICILLIN", "AZITHROMYCIN", "CEPHALEXIN", "TETRACYCLINE", "TMP-SMX")
real.abx.cols <- paste0("antibiotics__treatment__", REAL.ABX)

################################################################################
## 5. clean up
################################################################################

raw.clean <- raw %>%
  mutate(

    ## --- cancer outcome ---
    Cancer.Dx = case_when(
      rowSums(across(all_of(cancer.outcome.cols), ~ as.integer(.x == "Yes")),
              na.rm = TRUE) > 0 ~ "Cancer",
      TRUE ~ "No Cancer"),

    ## --- antibiotic exposure ---
    Any.Abx = case_when(
      rowSums(across(all_of(real.abx.cols), ~ as.integer(.x == "Yes")),
              na.rm = TRUE) > 0 ~ "Yes",
      TRUE ~ "No"),

    Abx.Penicillin = case_when(
      `antibiotics__treatment__AMOXICILLIN`  == "Yes" ~ "Yes",
      TRUE ~ "No"),

    Abx.Macrolide = case_when(
      `antibiotics__treatment__AZITHROMYCIN` == "Yes" ~ "Yes",
      TRUE ~ "No"),

    Abx.Cephalosporin = case_when(
      `antibiotics__treatment__CEPHALEXIN`   == "Yes" ~ "Yes",
      TRUE ~ "No"),

    Abx.Tetracycline = case_when(
      `antibiotics__treatment__TETRACYCLINE` == "Yes" ~ "Yes",
      TRUE ~ "No"),

    Abx.TmpSmx = case_when(
      `antibiotics__treatment__TMP-SMX`      == "Yes" ~ "Yes",
      TRUE ~ "No"),

    ## --- demographics ---
    Age = as.numeric(`demographics__age_at_index_date`),

    Age.group = case_when(
      as.numeric(`demographics__age_at_index_date`) <  18 ~ "<18",
      as.numeric(`demographics__age_at_index_date`) <  30 ~ "18-29",
      as.numeric(`demographics__age_at_index_date`) >= 30 ~ "30+",
      TRUE ~ NA_character_),

    BMI = as.numeric(`demographics__bmi`),

    BMI.Category = case_when(
      str_detect(`demographics__bmi_category`, regex("underweight", ignore_case = TRUE)) ~ "Underweight",
      str_detect(`demographics__bmi_category`, regex("normal",      ignore_case = TRUE)) ~ "Normal weight",
      str_detect(`demographics__bmi_category`, regex("overweight",  ignore_case = TRUE)) ~ "Overweight",
      str_detect(`demographics__bmi_category`, regex("obese",       ignore_case = TRUE)) ~ "Obese",
      TRUE ~ "Unknown"),

    Sex = case_when(
      str_detect(`demographics__sex`, regex("female", ignore_case = TRUE)) ~ "Female",
      str_detect(`demographics__sex`, regex("male",   ignore_case = TRUE)) ~ "Male",
      TRUE ~ "Unknown"),

    Race = case_when(
      str_detect(`demographics__race`, regex("white",    ignore_case = TRUE)) ~ "White",
      str_detect(`demographics__race`, regex("black",    ignore_case = TRUE)) ~ "Black",
      str_detect(`demographics__race`, regex("hispanic", ignore_case = TRUE)) ~ "Hispanic",
      str_detect(`demographics__race`, regex("asian",    ignore_case = TRUE)) ~ "Asian",
      str_detect(`demographics__race`, regex("other",    ignore_case = TRUE)) ~ "Other",
      TRUE ~ "Unknown"),

    ## --- other covariates ---
    Contraceptives = case_when(
      `contraceptives__treatment__pooled` == "Yes" ~ "Yes",
      TRUE ~ "No"),

    Smoking = case_when(
      str_detect(`smoking_status__pre_index__pooled`, "Never")   ~ "Never",
      str_detect(`smoking_status__pre_index__pooled`, "Former")  ~ "Former",
      str_detect(`smoking_status__pre_index__pooled`, "Current") ~ "Current",
      TRUE ~ "Unknown"),

    Alcohol = case_when(
      str_detect(`alcohol_status__pre_index__pooled`,
                 regex("currently drinks",   ignore_case = TRUE)) ~ "Drinks",
      str_detect(`alcohol_status__pre_index__pooled`,
                 regex("does not currently", ignore_case = TRUE)) ~ "Non-drinker",
      TRUE ~ "Unknown"),

    Transplant = case_when(
      `transplant__pre_index__STRUCTURED_DATA` == "Yes" ~ "Yes",
      TRUE ~ "No"),

    Fam.Cancer = case_when(
      rowSums(across(all_of(fam.cols), ~ as.integer(.x == "Yes")),
              na.rm = TRUE) > 0 ~ "Yes",
      TRUE ~ "No"),

    n.comorbidities = rowSums(across(all_of(dis.cols), ~ as.integer(.x == "Yes")),
                               na.rm = TRUE),

    Comorbidities = case_when(
      n.comorbidities == 0 ~ "None",
      n.comorbidities == 1 ~ "1",
      n.comorbidities == 2 ~ "2",
      n.comorbidities >= 3 ~ "3+",
      TRUE ~ "None"),

    ## --- cancer type indicators ---
    Ca.Skin.NM    = case_when(`cancer_cancer__outcome__Other and unspecified malignant neoplasm of skin` == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Breast     = case_when(`cancer_cancer__outcome__Malignant neoplasms of breast`                   == "Yes" |
                              `cancer_cancer__outcome__female breast`                                   == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Melanoma   = case_when(`cancer_cancer__outcome__melanoma of skin`                                == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Thyroid    = case_when(`cancer_cancer__outcome__thyroid gland`                                   == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Lung       = case_when(`cancer_cancer__outcome__bronchus and lung`                               == "Yes" |
                              `cancer_cancer__outcome__trachea bronchus and lung`                       == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Colorectal = case_when(`cancer_cancer__outcome__colon`                                           == "Yes" |
                              `cancer_cancer__outcome__rectum`                                          == "Yes" |
                              `cancer_cancer__outcome__rectosigmoid junction`                           == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Prostate   = case_when(`cancer_cancer__outcome__prostate`                                        == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Lymphoma   = case_when(`cancer_cancer__outcome__Other specified and unspecified types of non-Hodgkin lymphoma` == "Yes" |
                              `cancer_cancer__outcome__Non-follicular lymphoma`                         == "Yes" |
                              `cancer_cancer__outcome__Hodgkin lymphoma`                                == "Yes" |
                              `cancer_cancer__outcome__Follicular lymphoma`                             == "Yes" |
                              `cancer_cancer__outcome__Malignant immunoproliferative diseases and certain other B-cell lymphomas` == "Yes" |
                              `cancer_cancer__outcome__Mature T/NK-cell lymphomas`                      == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Uterine    = case_when(`cancer_cancer__outcome__corpus uteri`                                    == "Yes" |
                              `cancer_cancer__outcome__body of uterus`                                  == "Yes" |
                              `cancer_cancer__outcome__uterus, part unspecified`                        == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Kidney     = case_when(`cancer_cancer__outcome__kidney, except renal pelvis`                     == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Bladder    = case_when(`cancer_cancer__outcome__bladder`                                         == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Ovary      = case_when(`cancer_cancer__outcome__ovary`                                           == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Leukemia   = case_when(`cancer_cancer__outcome__Lymphoid leukemia`                               == "Yes" |
                              `cancer_cancer__outcome__Myeloid leukemia`                                == "Yes" |
                              `cancer_cancer__outcome__Leukemia of unspecified cell type`               == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Brain      = case_when(`cancer_cancer__outcome__brain`                                           == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Pancreas   = case_when(`cancer_cancer__outcome__pancreas`                                        == "Yes" ~ "Yes", TRUE ~ "No"),
    Ca.Cervix     = case_when(`cancer_cancer__outcome__cervix uteri`                                    == "Yes" ~ "Yes", TRUE ~ "No"),

    ## --- survival dates ---
    index.dt        = as.Date(substr(`index_date`,                      1, 10)),
    last.record.dt  = as.Date(substr(`demographics__last_record_date`,  1, 10)),
    death.dt        = as.Date(substr(`demographics__date_of_death`,     1, 10)),
    deceased        = (`demographics__deceased` == "Yes")

  ) %>%
  mutate(

    ## --- antibiotic class count (depends on individual flags above) ---
    n.abx.classes = rowSums(
      across(c(Abx.Penicillin, Abx.Macrolide, Abx.Cephalosporin,
               Abx.Tetracycline, Abx.TmpSmx),
             ~ as.integer(.x == "Yes")), na.rm = TRUE),

    Abx.Classes = case_when(
      n.abx.classes == 0 ~ "None",
      n.abx.classes == 1 ~ "1 class",
      n.abx.classes == 2 ~ "2 classes",
      n.abx.classes >= 3 ~ "3+ classes",
      TRUE ~ "None"),

    ## --- antibiotic duration ---
    ## Convert each duration column to numeric, treating "F", "", and literal
    ## "0" as NA. Literal "0" appears only among Received==Yes patients and
    ## almost certainly represents "received but duration not captured" rather
    ## than a true 0-day course. Then sum across columns. If ALL columns are
    ## missing for a patient, rowSums returns 0 with na.rm=TRUE — we set those
    ## to NA explicitly. Non-ABX patients are also set to NA.
    Abx.Duration = {
      dur.num  <- select(., all_of(dur.cols)) %>%
        mutate(across(everything(),
                      ~ suppressWarnings(as.numeric(ifelse(.x == "F" | .x == "" | .x == "0",
                                                            NA, .x)))))
      n.valid  <- rowSums(!is.na(dur.num))
      row.sums <- rowSums(dur.num, na.rm = TRUE)
      ifelse(n.valid == 0, NA_real_, row.sums)
    },
    Abx.Duration = case_when(
      Any.Abx == "No" ~ NA_real_,
      TRUE            ~ Abx.Duration),

    ## Categorical duration: Low / Medium / High (NA for non-ABX or unknown duration)
    Abx.Duration.Cat = case_when(
      is.na(Abx.Duration)  ~ NA_character_,
      Abx.Duration <= 90   ~ "Low (≤90 days)",
      Abx.Duration <= 180  ~ "Medium (91–180 days)",
      Abx.Duration >  180  ~ "High (>180 days)",
      TRUE                 ~ NA_character_),

    ## --- cancer diagnosis date (earliest across all cancer type cols) ---
    cancer.diag.dt = {
      raw.dates <- select(., all_of(cancer.type.cols))
      as.Date(sapply(seq_len(nrow(raw.dates)), function(i) {
        vals  <- unlist(raw.dates[i, ])
        vals  <- vals[!is.na(vals) & nchar(trimws(vals)) > 0]
        if (length(vals) == 0) return(NA_character_)
        dates <- tryCatch(as.Date(substr(vals, 1, 10), format = "%Y-%m-%d"),
                          error = function(e) as.Date(NA))
        dates <- dates[!is.na(dates)]
        if (length(dates) == 0) return(NA_character_)
        as.character(min(dates))
      }))
    },

    ## --- survival time ---
    censor.dt   = case_when(
      deceased & !is.na(death.dt) ~ pmin(death.dt, last.record.dt, na.rm = TRUE),
      TRUE ~ last.record.dt),

    surv.end.dt = case_when(
      Cancer.Dx == "Cancer" & !is.na(cancer.diag.dt) ~ cancer.diag.dt,
      TRUE ~ censor.dt),

    follow.time = as.numeric(surv.end.dt - (index.dt + years(1))) / 365.25,
    surv.event  = as.integer(Cancer.Dx == "Cancer"),

  ) %>%
  select(Cancer.Dx,
         Any.Abx,
         Abx.Classes,
         Abx.Duration,
         Abx.Duration.Cat,
         Abx.Penicillin,
         Abx.Macrolide,
         Abx.Cephalosporin,
         Abx.Tetracycline,
         Abx.TmpSmx,
         Age,
         Age.group,
         BMI,
         BMI.Category,
         Sex,
         Race,
         Contraceptives,
         Smoking,
         Alcohol,
         Transplant,
         Fam.Cancer,
         n.comorbidities,
         Comorbidities,
         Ca.Skin.NM,
         Ca.Breast,
         Ca.Melanoma,
         Ca.Thyroid,
         Ca.Lung,
         Ca.Colorectal,
         Ca.Prostate,
         Ca.Lymphoma,
         Ca.Uterine,
         Ca.Kidney,
         Ca.Bladder,
         Ca.Ovary,
         Ca.Leukemia,
         Ca.Brain,
         Ca.Pancreas,
         Ca.Cervix,
         follow.time,
         surv.event)

################################################################################
## 6. factor levels
################################################################################

final <- raw.clean %>%
  filter(!is.na(Cancer.Dx)) %>%
  mutate(
    Cancer.Dx = factor(Cancer.Dx,
                       levels = c("No Cancer", "Cancer")),

    Any.Abx = factor(Any.Abx,
                     levels = c("No", "Yes")),

    Abx.Classes = factor(Abx.Classes,
                         levels = c("None", "1 class", "2 classes", "3+ classes")),

    Abx.Duration.Cat = factor(Abx.Duration.Cat,
                              levels = c("Low (≤90 days)", "Medium (91–180 days)", "High (>180 days)")),

    Abx.Penicillin = factor(Abx.Penicillin,
                            levels = c("No", "Yes")),

    Abx.Macrolide = factor(Abx.Macrolide,
                           levels = c("No", "Yes")),

    Abx.Cephalosporin = factor(Abx.Cephalosporin,
                               levels = c("No", "Yes")),

    Abx.Tetracycline = factor(Abx.Tetracycline,
                              levels = c("No", "Yes")),

    Abx.TmpSmx = factor(Abx.TmpSmx,
                        levels = c("No", "Yes")),

    Age.group = factor(Age.group,
                       levels = c("<18", "18-29", "30+")),

    BMI.Category = factor(BMI.Category,
                          levels = c("Underweight", "Normal weight", "Overweight", "Obese", "Unknown")),

    Sex = factor(Sex,
                 levels = c("Female", "Male")),

    Race = factor(Race,
                  levels = c("White", "Black", "Hispanic", "Asian", "Other", "Unknown")),

    Contraceptives = factor(Contraceptives,
                            levels = c("No", "Yes")),

    Smoking = factor(Smoking,
                     levels = c("Never", "Former", "Current", "Unknown")),

    Alcohol = factor(Alcohol,
                     levels = c("Non-drinker", "Drinks", "Unknown")),

    Transplant = factor(Transplant,
                        levels = c("No", "Yes")),

    Fam.Cancer = factor(Fam.Cancer,
                        levels = c("No", "Yes")),

    Comorbidities = factor(Comorbidities,
                           levels = c("None", "1", "2", "3+")),

    Ca.Skin.NM    = factor(Ca.Skin.NM,    levels = c("No", "Yes")),
    Ca.Breast     = factor(Ca.Breast,     levels = c("No", "Yes")),
    Ca.Melanoma   = factor(Ca.Melanoma,   levels = c("No", "Yes")),
    Ca.Thyroid    = factor(Ca.Thyroid,    levels = c("No", "Yes")),
    Ca.Lung       = factor(Ca.Lung,       levels = c("No", "Yes")),
    Ca.Colorectal = factor(Ca.Colorectal, levels = c("No", "Yes")),
    Ca.Prostate   = factor(Ca.Prostate,   levels = c("No", "Yes")),
    Ca.Lymphoma   = factor(Ca.Lymphoma,   levels = c("No", "Yes")),
    Ca.Uterine    = factor(Ca.Uterine,    levels = c("No", "Yes")),
    Ca.Kidney     = factor(Ca.Kidney,     levels = c("No", "Yes")),
    Ca.Bladder    = factor(Ca.Bladder,    levels = c("No", "Yes")),
    Ca.Ovary      = factor(Ca.Ovary,      levels = c("No", "Yes")),
    Ca.Leukemia   = factor(Ca.Leukemia,   levels = c("No", "Yes")),
    Ca.Brain      = factor(Ca.Brain,      levels = c("No", "Yes")),
    Ca.Pancreas   = factor(Ca.Pancreas,   levels = c("No", "Yes")),
    Ca.Cervix     = factor(Ca.Cervix,     levels = c("No", "Yes"))
  ) %>%
  filter(!is.na(Cancer.Dx),
         !is.na(follow.time),
         follow.time > 0)
