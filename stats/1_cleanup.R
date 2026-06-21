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

input  <- '/home/mtoles/acne/full_inference_out'
output <- '/home/mtoles/acne/stats/eda_output'

dir.create(output, showWarnings = FALSE)

## --- analysis config -------------------------------------------------------- #
## use_smoking_amount: when TRUE, current smokers are subdivided by packs/week
## (Current <2 / <6 / 6+ / amt unknown). When FALSE, all current smokers share a
## single "Current" category. Toggling this here propagates to every stats script
## (2-9), since they all source this file.
use_smoking_amount <- FALSE

## include_preexisting_cancer: when TRUE, a patient's preexisting (pre-index) cancer
## is INCLUDED AS A COVARIATE (Prior.Cancer) in the adjusted models. When FALSE, it is
## not adjusted for. EITHER WAY all patients stay in the cohort — this never changes the
## study population. Propagates to every stats script (2-9).
## TESTING: THIS SHOULD BE TRUE
include_preexisting_cancer <- TRUE

################################################################################
## 3. data read-in
################################################################################

raw <- read_csv(file.path(input, "pooled_records.csv"),
                col_types   = cols(.default = "c"),
                name_repair = "minimal")
# Column schema: full_inference_out/pooled_records_columns.txt

################################################################################
## 4. column inventory
################################################################################

all.cols <- names(raw)

dur.cols            <- grep("^antibiotic_duration_numeric__treatment__", all.cols, value = TRUE)
cancer.outcome.cols <- grep("^cancer_outcome__",                         all.cols, value = TRUE)
cancer.type.cols    <- grep("^cancer_date_of_diagnosis__outcome__",      all.cols, value = TRUE)
fam.cols            <- grep("^cancer_family_any__(pre_index|treatment)__", all.cols, value = TRUE)
dis.cols            <- grep("^disease__pre_index__",                     all.cols, value = TRUE)
precancer.cols      <- grep("^cancer_preexisting__",                     all.cols, value = TRUE)

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

    ## Smoking status & packs/week amount, merged across periods with priority
    ## pre_index > treatment > outcome (baseline preferred; fall back to later
    ## periods for coverage, since most amount data sits in the follow-up window).
    smk.status = coalesce(na_if(`smoking_status__pre_index__pooled`, ""),
                          na_if(`smoking_status__treatment__pooled`, ""),
                          na_if(`smoking_status__outcome__pooled`,   "")),
    smk.amount = coalesce(na_if(`smoking_amount__pre_index__pooled`, ""),
                          na_if(`smoking_amount__treatment__pooled`, ""),
                          na_if(`smoking_amount__outcome__pooled`,   "")),

    ## Packs/week bins: <2 = {0, 1-2}, <6 = {3-5}, 6+ = {6+}; anything else
    ## (incl. "unknown quantity" and missing) -> "amt unknown". Used only to
    ## subdivide current smokers below.
    Smoke.Amt = case_when(
      str_detect(smk.amount, "^0") | smk.amount == "1-2" ~ "<2",
      smk.amount == "3-5"                                ~ "<6",
      smk.amount == "6+"                                 ~ "6+",
      TRUE                                               ~ "amt unknown"),

    ## Subdivide current smokers by amount only when use_smoking_amount is TRUE;
    ## otherwise all current smokers collapse to a single "Current" category.
    Smoking = case_when(
      str_detect(smk.status, "Never")   ~ "Never",
      str_detect(smk.status, "Former")  ~ "Former",
      str_detect(smk.status, "Current") ~ if (use_smoking_amount) paste0("Current ", Smoke.Amt) else "Current",
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

    ## --- preexisting cancers: treated as a confounder, parallel to comorbidities ---
    n.prior.cancer = rowSums(across(all_of(precancer.cols), ~ as.integer(.x == "Yes")),
                             na.rm = TRUE),

    Prior.Cancer = case_when(
      n.prior.cancer == 0 ~ "None",
      n.prior.cancer == 1 ~ "1",
      n.prior.cancer == 2 ~ "2",
      n.prior.cancer >= 3 ~ "3+",
      TRUE ~ "None"),

    ## --- cancer type indicators ---
    ## Generated automatically (one Yes/No column per raw cancer_outcome__<label>
    ## column) further below — no manual grouping. See section 7.

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

    ## Start of follow-up: 1-year washout after index. Kept as a column so the
    ## type-specific clocks (8_cox_types.R / 7_km_types.R) share the same origin.
    start.dt    = index.dt + years(1),
    follow.time = as.numeric(surv.end.dt - start.dt) / 365.25,
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
         n.prior.cancer,
         Prior.Cancer,
         all_of(cancer.outcome.cols),   # raw per-type indicators, recoded in section 7
         all_of(cancer.type.cols),      # raw per-type dx-date strings, parsed in section 7
         start.dt,
         censor.dt,
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
                     levels = if (use_smoking_amount)
                       c("Never", "Former",
                         "Current <2", "Current <6", "Current 6+",
                         "Current amt unknown", "Unknown")
                     else
                       c("Never", "Former", "Current", "Unknown")),

    Alcohol = factor(Alcohol,
                     levels = c("Non-drinker", "Drinks", "Unknown")),

    Transplant = factor(Transplant,
                        levels = c("No", "Yes")),

    Fam.Cancer = factor(Fam.Cancer,
                        levels = c("No", "Yes")),

    Comorbidities = factor(Comorbidities,
                           levels = c("None", "1", "2", "3+")),

    Prior.Cancer = factor(Prior.Cancer,
                          levels = c("None", "1", "2", "3+"))
  ) %>%
  filter(!is.na(Cancer.Dx),
         !is.na(follow.time),
         follow.time > 0)

################################################################################
## 7. cancer type indicators (automatic, one per raw cancer_outcome__ column)
##
## Every `cancer_outcome__<label>` column becomes its own Yes/No factor — no
## manual grouping. Downstream scripts iterate over `ca.type.cols` (the column
## names, e.g. "Non-melanoma_skin_cancer") and `ca.type.labels` (display labels
## with underscores turned into spaces). The two vectors are positionally
## aligned. Edit nothing here when the label set changes — it adapts.
################################################################################

ca.type.cols   <- sub("^cancer_outcome__", "", cancer.outcome.cols)
ca.type.labels <- str_trim(gsub("_", " ", ca.type.cols))

final <- final %>%
  ## indicators: recode Yes / (blank|NA) -> Yes / No (missing = "No" so non-cases
  ## become "No" rather than NA), then strip the cancer_outcome__ prefix.
  mutate(across(all_of(cancer.outcome.cols),
                ~ factor(if_else(.x == "Yes", "Yes", "No", missing = "No"),
                         levels = c("No", "Yes")))) %>%
  rename_with(~ sub("^cancer_outcome__", "", .x), all_of(cancer.outcome.cols)) %>%
  ## per-type diagnosis dates: parse the raw date strings to Date and rename to
  ## dxdt__<type>, positionally matching the ca.type.cols indicators above. These
  ## drive the type-specific survival clock in 8_cox_types.R / 7_km_types.R.
  mutate(across(all_of(cancer.type.cols),
                ~ as.Date(substr(.x, 1, 10), format = "%Y-%m-%d"))) %>%
  rename_with(~ paste0("dxdt__", sub("^cancer_date_of_diagnosis__outcome__", "", .x)),
              all_of(cancer.type.cols))

## Guard: every type flagged "Yes" must carry a parseable diagnosis date, because
## its survival clock is dated from it. Crash rather than silently mis-time events.
for (.t in ca.type.cols) {
  .bad <- sum(final[[.t]] == "Yes" & is.na(final[[paste0("dxdt__", .t)]]))
  if (.bad > 0) stop(sprintf("%d patient(s) have %s == 'Yes' but no diagnosis date", .bad, .t))
}
rm(.t, .bad)

################################################################################
## 8. shared Cox covariate set
##
## Fully-adjusted covariate set used by every Cox model (6_cox.R, 8_cox_types.R).
## Defined once here so the adjustment set cannot silently diverge between scripts.
################################################################################

## Prior.Cancer is included in the adjustment set only when include_preexisting_cancer
## is TRUE. The cohort itself is unchanged either way — this flag toggles the COVARIATE
## only, never which patients are in the study.
adj.vars <- c("Age", "Sex", "Race", "BMI.Category", "Smoking",
              "Alcohol", "Contraceptives", "Fam.Cancer", "Transplant",
              "Comorbidities",
              if (include_preexisting_cancer) "Prior.Cancer")

## Keep only covariates that actually vary in a given data frame: factors need >=2
## observed levels, numerics need >=2 distinct values. A constant covariate would
## otherwise crash coxph with a "contrasts ... 2 or more levels" error.
usable.covars <- function(df, vars) {
  vars[vapply(vars, function(v) {
    x <- df[[v]]
    if (is.factor(x)) nlevels(droplevels(x[!is.na(x)])) >= 2
    else length(unique(x[!is.na(x)])) >= 2
  }, logical(1))]
}
