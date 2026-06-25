# COX1 (all-cancer) ablation harness — Daniel's 6_cox <-> our current 6_cox.
#
# Reproduces the COX1 ("Any Antibiotic" -> any-cancer) models from 6_cox.R using the SAME
# run.cox / tbl_regression path, with independently-toggled axes: the data snapshot plus the
# code changes that differ between Daniel's 6_cox and ours.
#
# Axes (0 = Daniel, 1 = ours):
#   data          "bak"=pre-change pooled_records.csv.bak | "current"=current pooled_records.csv
#   followup      0 -> index+1yr (Daniel); 1 -> index+2yr (outcome window)
#   comorbidities 0 -> not adjusted;        1 -> Comorbidities in adj.vars
#   preexisting   0 -> no prior-cancer covariate; 1 -> a prior-cancer covariate is adjusted for
#   leuk_lymph    0 -> leukemia/lymphoma folded INTO the prior-cancer covariate (Prior.Cancer.All);
#                 1 -> leukemia/lymphoma is its OWN covariate and EXCLUDED from prior-cancer
#                      (which becomes Prior.Cancer.Other)
#   hiv           0 -> HIV stays in the comorbidity count, no HIV covariate;
#                 1 -> Prior.HIV is its own covariate AND HIV is removed from the comorbidity count
#   transplant    0 -> no Transplant covariate; 1 -> Transplant in adj.vars
# Transplant is in adj.vars for BOTH Daniel's and our real 6_cox, so both real-equivalent
# configs use transplant=1; this axis exists to isolate dropping it.
# "ours" (current 6_cox) = followup1 comorb1 preexist1 leuk1 hiv1 transplant1;
# "daniel" = followup0 comorb0 preexist0 leuk0 hiv0 transplant1.
#
# The preexisting/leuk/HIV flags are DERIVED here from the per-type cancer_preexisting__
# columns (leukemia/lymphoma = column name containing "leukemia"/"lymphoma", exactly the
# rule postprocess uses) and the pre-index HIV disease columns, so the harness works on
# BOTH snapshots even though the .bak predates the postprocess category columns.
#
# Usage:
#   Rscript debug_cox1.R <configs_file> <out_dir>
# configs_file: one config per line "<label> <data> <followup> <comorbidities> <preexisting> <leuk_lymph> <hiv> <transplant>".
# For each config writes <out_dir>/cox_debug_<label>.csv in the SAME format as 6_cox's
# COX1_any_cancer.csv (tbl_regression as_tibble + exposure + model) plus a `config` column.

suppressMessages({
  library(tidyverse)
  library(survival)
  library(lubridate)
  library(gtsummary)
})

args <- commandArgs(trailingOnly = TRUE)
configs_file <- args[1]
out_dir      <- args[2]
stopifnot(!is.na(configs_file), !is.na(out_dir))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

input    <- "/home/mtoles/acne/full_inference_out"
bak_path <- file.path(input, "pooled_records.csv.bak")
REAL.ABX <- c("AMOXICILLIN", "AZITHROMYCIN", "CEPHALEXIN", "TETRACYCLINE", "TMP-SMX")

## --- helpers copied verbatim from 1_cleanup.R / 6_cox.R for fidelity ---
usable.covars <- function(df, vars) {
  vars[vapply(vars, function(v) {
    x <- df[[v]]
    if (is.factor(x)) nlevels(droplevels(x[!is.na(x)])) >= 2 else length(unique(x[!is.na(x)])) >= 2
  }, logical(1))]
}

run.cox <- function(df, exposure, label, adj.vars) {
  f.unadj <- as.formula(paste0("Surv(follow.time, surv.event) ~ ", exposure))
  f.age   <- as.formula(paste0("Surv(follow.time, surv.event) ~ ", exposure, " + Age"))
  f.full  <- as.formula(paste0("Surv(follow.time, surv.event) ~ ", exposure, " + ",
                                paste(usable.covars(df, adj.vars), collapse = " + ")))
  fmt <- function(fit, model.label) {
    tbl_regression(fit, exponentiate = TRUE) %>%
      as_tibble() %>%
      mutate(exposure = label, model = model.label)
  }
  bind_rows(
    fmt(coxph(f.unadj, data = df), "Unadjusted"),
    fmt(coxph(f.age,   data = df), "Age-adjusted"),
    fmt(coxph(f.full,  data = df), "Fully adjusted")
  )
}

base.adj <- c("Age", "Sex", "Race", "BMI.Category", "Smoking",
              "Alcohol", "Contraceptives", "Fam.Cancer")

## --- toggle-independent feature frame for one data snapshot (built ONCE per snapshot) ---
## Mirrors 1_cleanup.R. Comorbidities (depends on the preexisting/HIV toggle) and
## follow.time (depends on the followup toggle) are computed per-config below.
build_base <- function(csv) {
  raw <- read_csv(csv, col_types = cols(.default = "c"), name_repair = "minimal")
  ## Daniel's pre-index pooled smoking status lives only in the .bak snapshot; the current
  ## file emits closest-structured instead, so join the pre-index column in when absent.
  if (!("smoking_status__pre_index__pooled" %in% names(raw))) {
    bk <- read_csv(bak_path, col_types = cols(.default = "c"), name_repair = "minimal") %>%
      select(EMPI, `smoking_status__pre_index__pooled`)
    raw <- raw %>% left_join(bk, by = "EMPI")
  }
  all.cols <- names(raw)

  cancer.outcome.cols <- grep("^cancer_outcome__",                    all.cols, value = TRUE)
  cancer.type.cols    <- grep("^cancer_date_of_diagnosis__outcome__", all.cols, value = TRUE)
  fam.cols            <- grep("^cancer_family_any__pre_index__",      all.cols, value = TRUE)
  dis.cols.all        <- grep("^disease__pre_index__",                all.cols, value = TRUE)
  hiv.cols            <- dis.cols.all[grepl("human_immunodeficiency", dis.cols.all, ignore.case = TRUE)]
  dis.cols.nohiv      <- setdiff(dis.cols.all, hiv.cols)
  precancer.cols      <- grep("^cancer_preexisting__",               all.cols, value = TRUE)
  leuk.lymph.cols     <- precancer.cols[grepl("leukemia|lymphoma", precancer.cols, ignore.case = TRUE)]
  other.cancer.cols   <- setdiff(precancer.cols, leuk.lymph.cols)
  real.abx.cols       <- paste0("antibiotics__treatment__", REAL.ABX)

  ## count "Yes" across a set of columns (0 if the set is empty)
  count_yes <- function(cols) {
    if (length(cols) == 0) return(rep(0L, nrow(raw)))
    rowSums(as.data.frame(lapply(raw[cols], function(x) as.integer(x == "Yes"))), na.rm = TRUE)
  }
  n.cancer.outcome <- count_yes(cancer.outcome.cols)
  n.abx            <- count_yes(real.abx.cols)
  n.fam            <- count_yes(fam.cols)
  n.other          <- count_yes(other.cancer.cols)
  n.leuk           <- count_yes(leuk.lymph.cols)
  n.hiv            <- count_yes(hiv.cols)

  smk.status <- na_if(raw[["smoking_status__pre_index__pooled"]], "")

  raw %>%
    mutate(
      Cancer.Dx = if_else(n.cancer.outcome > 0, "Cancer", "No Cancer"),
      Any.Abx   = if_else(n.abx > 0, "Yes", "No"),
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
      Contraceptives = case_when(`contraceptives__treatment__pooled` == "Yes" ~ "Yes", TRUE ~ "No"),
      Alcohol = case_when(
        str_detect(`alcohol_status__pre_index__pooled`, regex("currently drinks",   ignore_case = TRUE)) ~ "Drinks",
        str_detect(`alcohol_status__pre_index__pooled`, regex("does not currently", ignore_case = TRUE)) ~ "Non-drinker",
        TRUE ~ "Unknown"),
      Transplant = case_when(`transplant__pre_index__STRUCTURED_DATA` == "Yes" ~ "Yes", TRUE ~ "No"),
      Fam.Cancer = if_else(n.fam > 0, "Yes", "No"),
      Smoking = case_when(
        str_detect(smk.status, "Never")   ~ "Never",
        str_detect(smk.status, "Former")  ~ "Former",
        str_detect(smk.status, "Current") ~ "Current",
        TRUE ~ "Unknown"),

      n.comorb.all   = count_yes(dis.cols.all),
      n.comorb.nohiv = count_yes(dis.cols.nohiv),

      Prior.Cancer.Other = if_else(n.other > 0, "Yes", "No"),              # preexisting cancers excl leuk/lymph
      Prior.Cancer.All   = if_else(n.other > 0 | n.leuk > 0, "Yes", "No"), # all preexisting cancers (incl leuk/lymph)
      Prior.Leuk.Lymph   = if_else(n.leuk  > 0, "Yes", "No"),
      Prior.HIV          = if_else(n.hiv   > 0, "Yes", "No"),

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
      surv.event  = as.integer(Cancer.Dx == "Cancer")
    ) %>%
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
      Prior.Cancer.Other = factor(Prior.Cancer.Other, c("No", "Yes")),
      Prior.Cancer.All   = factor(Prior.Cancer.All,   c("No", "Yes")),
      Prior.Leuk.Lymph   = factor(Prior.Leuk.Lymph,   c("No", "Yes")),
      Prior.HIV          = factor(Prior.HIV,          c("No", "Yes"))
    )
}

## Build each data snapshot at most once (lazy cache keyed by data value).
data_path <- function(d) if (d == "bak") bak_path else file.path(input, "pooled_records.csv")
base_cache <- list()
get_base <- function(d) {
  if (is.null(base_cache[[d]])) base_cache[[d]] <<- build_base(data_path(d))
  base_cache[[d]]
}

## --- per-config fits ---
configs <- readLines(configs_file)
for (line in configs) {
  line <- trimws(line)
  if (line == "" || startsWith(line, "#")) next
  parts    <- strsplit(line, "\\s+")[[1]]
  lbl      <- parts[1]
  dat      <- parts[2]                 # "bak" | "current"
  fu       <- as.integer(parts[3])     # 0 = Daniel (index+1yr), 1 = ours (index+2yr)
  comorb   <- as.integer(parts[4])
  preexist <- as.integer(parts[5])
  leuk     <- as.integer(parts[6])     # split leuk/lymph into its own covariate (excl from prior cancer)
  hiv      <- as.integer(parts[7])     # split HIV into its own covariate (remove from comorbidity count)
  trans    <- as.integer(parts[8])     # include Transplant in adj.vars
  stopifnot(dat %in% c("bak", "current"), !is.na(trans))
  fu.years <- if (fu == 1) 2 else 1

  base <- get_base(dat)
  ## HIV is removed from the comorbidity count only when the HIV split is on.
  ncomorb <- if (hiv == 1) base$n.comorb.nohiv else base$n.comorb.all

  d <- base %>%
    mutate(
      follow.time = as.numeric(surv.end.dt - (index.dt + years(fu.years))) / 365.25,
      Comorbidities = factor(case_when(
        ncomorb == 0 ~ "None", ncomorb == 1 ~ "1",
        ncomorb == 2 ~ "2",    ncomorb >= 3 ~ "3+", TRUE ~ "None"),
        levels = c("None", "1", "2", "3+"))
    ) %>%
    filter(!is.na(Cancer.Dx), !is.na(follow.time), follow.time > 0) %>%
    mutate(across(where(is.factor), droplevels))   # mirrors 6_cox.R final.cox

  ## prior-cancer covariate: "other" (leuk/lymph split out) when leuk==1, else "all"; only
  ## when preexisting==1. Leuk/lymph and HIV are their own covariates when their toggle is on.
  prior.cancer.var <- if (preexist == 1) (if (leuk == 1) "Prior.Cancer.Other" else "Prior.Cancer.All") else NULL
  adj <- c(base.adj,
           if (comorb == 1) "Comorbidities",
           prior.cancer.var,
           if (leuk == 1) "Prior.Leuk.Lymph",
           if (hiv  == 1) "Prior.HIV",
           if (trans == 1) "Transplant")

  res <- run.cox(d, "Any.Abx", "Any Antibiotic", adj) %>% mutate(config = lbl)
  write_csv(res, file.path(out_dir, paste0("cox_debug_", lbl, ".csv")))
  cat(sprintf("wrote %-18s data=%-7s N=%d  fu=%dyr comorb=%d preexist=%d leuk=%d hiv=%d trans=%d\n",
              lbl, dat, nrow(d), fu.years, comorb, preexist, leuk, hiv, trans))
}
