# title: "Antibiotic Exposure and Cancer Risk - Cox Models (longest-single-course-only sensitivity)"
# author: Daniel Kim
#
# Sensitivity analysis mirroring 6_cox.R but restricting the exposed group to patients whose
# LONGEST single antibiotic-class course was >=30 days. The unexposed (no antibiotic) comparison
# group is kept as-is, so the exposure contrast is "no antibiotics" vs "long (>=30 day) single
# antibiotic course". Writes its own COX_LONGEST* output files so it never overwrites 6_cox.R
# or the cumulative-dose variant (11_cox_long_only.R, which writes COX_LONG*).

################################################################################
## 1. clear all
################################################################################

rm(list = ls(all.names = TRUE))

################################################################################
## 2. set locations
################################################################################

source('/home/mtoles/acne/stats/1_cleanup.R')

################################################################################
## 3. data prep
################################################################################

## Longest-single-course-only restriction: keep every unexposed (no antibiotic) patient as the
## comparison group and, among exposed patients, keep only those whose LONGEST single antibiotic-
## class course was >=30 days (Abx.Duration.Max, the MAX across classes in 1_cleanup.R). Exposed
## patients whose longest course was <30 days -- or with unknown/uncaptured duration
## (Abx.Duration.Max NA) -- are dropped so the contrast is "no antibiotics" vs "long (>=30 day)
## single antibiotic course". Applied BEFORE droplevels so any factor levels emptied by the
## filter are removed.
final.cox <- final %>%
  filter(Any.Abx == "No" | (!is.na(Abx.Duration.Max) & Abx.Duration.Max >= 30)) %>%
  mutate(across(where(is.factor), droplevels))

## Fully adjusted covariate set (adj.vars) is defined in 1_cleanup.R

## Helper: run unadj / fully-adj Cox and return tidy tibble. (No age-adjusted model: the cohort
## is matched on age, so age is balanced by design and is not a covariate -- see adj.vars in
## 1_cleanup.R, which also excludes the other matching variables sex/race/bmi/smoking/alcohol.)
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
## 4. Cox — any antibiotic exposure
################################################################################

cox.any <- run.cox("Any.Abx", "Any Antibiotic (longest >=30d)")
write_csv(cox.any, file.path(output, "COX_LONGEST1_any_cancer.csv"))

################################################################################
## 5. Cox — number of antibiotic classes
################################################################################

cox.cls <- run.cox("Abx.Classes", "ABX Classes (longest >=30d)")
write_csv(cox.cls, file.path(output, "COX_LONGEST2_abx_classes.csv"))

################################################################################
## 6. Cox — individual antibiotic classes (all five simultaneously)
################################################################################

ind.vars <- paste(c("Abx.Penicillin", "Abx.Macrolide", "Abx.Cephalosporin",
                     "Abx.Tetracycline", "Abx.TmpSmx"), collapse = " + ")

cox.ind <- run.cox(ind.vars, "Individual ABX Classes (longest >=30d)")
write_csv(cox.ind, file.path(output, "COX_LONGEST3_individual_abx.csv"))
