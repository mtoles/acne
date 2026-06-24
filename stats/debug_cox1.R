# COX1 (all-cancer, fully adjusted) ablation harness.
#
# Single parameterized code path: each change that can differ from Daniel's original
# stats code is a CLI toggle. No before/after code duplication.
#
# Smoking is always Daniel's pre-index pooled status (smoking_status__pre_index__pooled,
# from the pre-change .bak snapshot); status only, no amount subdivision.
#
# Usage:
#   Rscript debug_cox1.R <label> <family_merge> <comorbidities> <prior_cancer> <full_csv>
# the toggles are 0/1 (or off/on). full_csv is the path the FULL coefficient table is
# appended to (header written by debug.sh). Prints one summary CSV line to stdout:
#   label,family_merge,comorbidities,prior_cancer,N,HR,se,z,p,vif
# and appends one row per model term to full_csv:
#   config,term,HR,coef,se,z,p,ci_low,ci_high

suppressMessages({
  library(tidyverse)
  library(survival)
  library(lubridate)
})

args <- commandArgs(trailingOnly = TRUE)
lbl  <- args[1]
on   <- function(i) tolower(args[i]) %in% c("1", "on", "true", "t", "yes")
FAMILY_MERGE   <- on(2)
COMORBIDITIES  <- on(3)
PRIOR_CANCER   <- on(4)
FULL_CSV       <- args[5]   # path to append this fit's full coefficient table to
stopifnot(!is.na(FULL_CSV))

input <- "/home/mtoles/acne/full_inference_out"
raw <- read_csv(file.path(input, "pooled_records.csv"),
                col_types = cols(.default = "c"), name_repair = "minimal")

## Daniel's pre_index pooled smoking status exists only in the pre-change snapshot;
## join it in by EMPI.
bak <- read_csv(file.path(input, "pooled_records.csv.bak"),
                col_types = cols(.default = "c"), name_repair = "minimal") %>%
  select(EMPI, `smoking_status__pre_index__pooled`)
raw <- raw %>% left_join(bak, by = "EMPI")
all.cols <- names(raw)

cancer.outcome.cols <- grep("^cancer_outcome__",                    all.cols, value = TRUE)
cancer.type.cols    <- grep("^cancer_date_of_diagnosis__outcome__", all.cols, value = TRUE)
dis.cols            <- grep("^disease__pre_index__",                all.cols, value = TRUE)
precancer.cols      <- grep("^cancer_preexisting__",                all.cols, value = TRUE)
REAL.ABX      <- c("AMOXICILLIN", "AZITHROMYCIN", "CEPHALEXIN", "TETRACYCLINE", "TMP-SMX")
real.abx.cols <- paste0("antibiotics__treatment__", REAL.ABX)

## --- TOGGLE: family-history period(s) ---
fam.cols <- if (FAMILY_MERGE)
  grep("^cancer_family_any__(pre_index|treatment)__", all.cols, value = TRUE) else
  grep("^cancer_family_any__pre_index__",             all.cols, value = TRUE)

## Smoking status: Daniel's pre-index pooled value (joined from .bak above).
smk.status <- na_if(raw[["smoking_status__pre_index__pooled"]], "")

raw.clean <- raw %>%
  mutate(
    Cancer.Dx = if_else(rowSums(across(all_of(cancer.outcome.cols), ~ as.integer(.x == "Yes")),
                                na.rm = TRUE) > 0, "Cancer", "No Cancer"),
    Any.Abx   = if_else(rowSums(across(all_of(real.abx.cols), ~ as.integer(.x == "Yes")),
                                na.rm = TRUE) > 0, "Yes", "No"),

    Age = as.numeric(`demographics__age_at_index_date`),
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
    Contraceptives = if_else(`contraceptives__treatment__pooled` == "Yes", "Yes", "No", missing = "No"),
    Alcohol = case_when(
      str_detect(`alcohol_status__pre_index__pooled`, regex("currently drinks",   ignore_case = TRUE)) ~ "Drinks",
      str_detect(`alcohol_status__pre_index__pooled`, regex("does not currently", ignore_case = TRUE)) ~ "Non-drinker",
      TRUE ~ "Unknown"),
    Transplant = if_else(`transplant__pre_index__STRUCTURED_DATA` == "Yes", "Yes", "No", missing = "No"),
    Fam.Cancer = if_else(rowSums(across(all_of(fam.cols), ~ as.integer(.x == "Yes")),
                                 na.rm = TRUE) > 0, "Yes", "No"),

    n.comorbidities = rowSums(across(all_of(dis.cols), ~ as.integer(.x == "Yes")), na.rm = TRUE),
    Comorbidities = case_when(
      n.comorbidities == 0 ~ "None", n.comorbidities == 1 ~ "1",
      n.comorbidities == 2 ~ "2",    n.comorbidities >= 3 ~ "3+", TRUE ~ "None"),

    n.prior.cancer = rowSums(across(all_of(precancer.cols), ~ as.integer(.x == "Yes")), na.rm = TRUE),
    Prior.Cancer = case_when(
      n.prior.cancer == 0 ~ "None", n.prior.cancer == 1 ~ "1",
      n.prior.cancer == 2 ~ "2",    n.prior.cancer >= 3 ~ "3+", TRUE ~ "None"),

    ## --- smoking status (pre-index pooled) ---
    smk.status = smk.status,
    Smoking = case_when(
      str_detect(smk.status, "Never")   ~ "Never",
      str_detect(smk.status, "Former")  ~ "Former",
      str_detect(smk.status, "Current") ~ "Current",
      TRUE ~ "Unknown"),

    ## --- survival (identical in Daniel and HEAD) ---
    index.dt       = as.Date(substr(`index_date`,                     1, 10)),
    last.record.dt = as.Date(substr(`demographics__last_record_date`, 1, 10)),
    death.dt       = as.Date(substr(`demographics__date_of_death`,    1, 10)),
    deceased       = (`demographics__deceased` == "Yes"),
    cancer.diag.dt = {
      raw.dates <- select(., all_of(cancer.type.cols))
      as.Date(sapply(seq_len(nrow(raw.dates)), function(i) {
        vals <- unlist(raw.dates[i, ]); vals <- vals[!is.na(vals) & nchar(trimws(vals)) > 0]
        if (length(vals) == 0) return(NA_character_)
        dts <- as.Date(substr(vals, 1, 10), format = "%Y-%m-%d"); dts <- dts[!is.na(dts)]
        if (length(dts) == 0) return(NA_character_)
        as.character(min(dts))
      }))
    },
    censor.dt   = case_when(deceased & !is.na(death.dt) ~ pmin(death.dt, last.record.dt, na.rm = TRUE),
                            TRUE ~ last.record.dt),
    surv.end.dt = case_when(Cancer.Dx == "Cancer" & !is.na(cancer.diag.dt) ~ cancer.diag.dt,
                            TRUE ~ censor.dt),
    follow.time = as.numeric(surv.end.dt - (index.dt + years(1))) / 365.25,
    surv.event  = as.integer(Cancer.Dx == "Cancer")
  )

final <- raw.clean %>%
  mutate(
    Any.Abx        = factor(Any.Abx,        c("No", "Yes")),
    BMI.Category   = factor(BMI.Category,   c("Underweight", "Normal weight", "Overweight", "Obese", "Unknown")),
    Sex            = factor(Sex,            c("Female", "Male")),
    Race           = factor(Race,           c("White", "Black", "Hispanic", "Asian", "Other", "Unknown")),
    Contraceptives = factor(Contraceptives, c("No", "Yes")),
    Smoking        = factor(Smoking,        c("Never", "Former", "Current", "Unknown")),
    Alcohol        = factor(Alcohol,        c("Non-drinker", "Drinks", "Unknown")),
    Transplant     = factor(Transplant,     c("No", "Yes")),
    Fam.Cancer     = factor(Fam.Cancer,     c("No", "Yes")),
    Comorbidities  = factor(Comorbidities,  c("None", "1", "2", "3+")),
    Prior.Cancer   = factor(Prior.Cancer,   c("None", "1", "2", "3+"))
  ) %>%
  filter(!is.na(Cancer.Dx), !is.na(follow.time), follow.time > 0)

## --- adjustment set: toggles add covariates ---
adj <- c("Age", "Sex", "Race", "BMI.Category", "Smoking",
         "Alcohol", "Contraceptives", "Fam.Cancer", "Transplant")
if (COMORBIDITIES) adj <- c(adj, "Comorbidities")
if (PRIOR_CANCER)  adj <- c(adj, "Prior.Cancer")
## guard against constant covariates
adj <- adj[vapply(adj, function(v) {
  x <- final[[v]]
  if (is.factor(x)) nlevels(droplevels(x[!is.na(x)])) >= 2 else length(unique(x[!is.na(x)])) >= 2
}, logical(1))]

f   <- as.formula(paste("Surv(follow.time, surv.event) ~ Any.Abx +", paste(adj, collapse = " + ")))
fit <- coxph(f, data = final)
co  <- summary(fit)$coefficients["Any.AbxYes", ]
hr  <- unname(co["exp(coef)"]); se <- unname(co["se(coef)"])
z   <- unname(co["z"]);         p  <- unname(co["Pr(>|z|)"])

## Variance inflation of the Any.Abx coefficient from the adjustment set:
## regress the exposure on all covariates; VIF = 1/(1 - R^2). VIF>1 means the
## covariates overlap the exposure and inflate SE(Any.Abx) (-> larger p) even
## when the HR barely moves. na.omit aligns it to the same usable rows.
vif.df <- final[, c("Any.Abx", adj)]
lp     <- lm(as.integer(Any.Abx == "Yes") ~ ., data = vif.df, na.action = na.omit)
r2     <- summary(lp)$r.squared
vif    <- 1 / (1 - r2)

cat(sprintf("%s,%d,%d,%d,%d,%.3f,%.4f,%.3f,%.4f,%.3f\n",
            lbl, FAMILY_MERGE, COMORBIDITIES, PRIOR_CANCER,
            nrow(final), hr, se, z, p, vif))

## --- full coefficient table (EVERY term, not just Any.Abx) appended to FULL_CSV ---
## The header is written once by debug.sh; here we only append data rows so the file
## ends up as one long-format table across all configs (config column distinguishes them).
co.all <- summary(fit)$coefficients   # coef, exp(coef), se(coef), z, Pr(>|z|)
ci.all <- summary(fit)$conf.int       # exp(coef), exp(-coef), lower .95, upper .95
full <- data.frame(
  config  = lbl,
  term    = rownames(co.all),
  HR      = co.all[, "exp(coef)"],
  coef    = co.all[, "coef"],
  se      = co.all[, "se(coef)"],
  z       = co.all[, "z"],
  p       = co.all[, "Pr(>|z|)"],
  ci_low  = ci.all[, "lower .95"],
  ci_high = ci.all[, "upper .95"],
  row.names = NULL, check.names = FALSE
)
write.table(full, FULL_CSV, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)
