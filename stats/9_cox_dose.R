# title: "Antibiotic Exposure and Cancer Risk - Cox Dose-Response (total abx duration)"
# author: Daniel Kim

################################################################################
## 1. clear all
################################################################################

rm(list = ls(all.names = TRUE))

################################################################################
## 2. set locations
################################################################################

source('/home/mtoles/acne/stats/1_cleanup.R')

################################################################################
## 3. data prep — bin total per-patient abx duration (all classes pooled)
##
## Reference = "None" (no antibiotic exposure). Exposed patients are binned by
## Abx.Duration (total days across ALL antibiotic classes, defined in
## 1_cleanup.R) using the thresholds 1-29, 30-89, 90-364, 365+ days.
##
## Exposed patients whose duration was never captured (Any.Abx == "Yes" but
## Abx.Duration is NA) are kept as their own "Unknown" category rather than dropped.
################################################################################

final.cox <- final %>%
  mutate(
    Abx.Dose = case_when(
      Any.Abx == "No"      ~ "None",
      is.na(Abx.Duration)  ~ "Unknown",
      Abx.Duration <= 29   ~ "1-29",
      Abx.Duration <= 89   ~ "30-89",
      Abx.Duration <= 364  ~ "90-364",
      TRUE                 ~ "365+"),
    Abx.Dose = factor(Abx.Dose,
                      levels = c("None", "1-29", "30-89", "90-364", "365+", "Unknown"))
  ) %>%
  ## Drop empty factor levels so all-zero dummy columns don't enter the models
  mutate(across(where(is.factor), droplevels))

## Distribution of the dose bins (sanity check)
cat("Abx.Dose distribution:\n")
print(table(final.cox$Abx.Dose, useNA = "ifany"))

## Fully adjusted covariate set (adj.vars) is defined in 1_cleanup.R

## Helper: run unadj / fully-adj Cox and return tidy tibble. (No age-adjusted model: age is a
## matching variable, balanced by design and excluded from every Cox model -- see adj.vars.)
run.cox <- function(exposure, label) {

  f.unadj <- as.formula(paste0("Surv(follow.time, surv.event) ~ ", exposure))
  f.full  <- as.formula(paste0("Surv(follow.time, surv.event) ~ ", exposure, " + ",
                                paste(usable.covars(final.cox, adj.vars), collapse = " + ")))

  fmt <- function(fit, model.label) {
    tbl_regression(fit, exponentiate = TRUE) %>%
      as_tibble() %>%
      mutate(exposure = label, model = model.label)
  }

  bind_rows(
    fmt(coxph(f.unadj, data = final.cox), "Unadjusted"),
    fmt(coxph(f.full,  data = final.cox), "Fully adjusted")
  )
}

################################################################################
## 4. Cox — dose-response on total antibiotic duration
################################################################################

cox.dose <- run.cox("Abx.Dose", "Total ABX Duration")
write_csv(cox.dose, file.path(output, "COX5_abx_dose.csv"))
