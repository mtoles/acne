# title: "Antibiotic Exposure and Cancer Risk - Cox Proportional Hazards Models"
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
## 3. data prep
################################################################################

## Drop empty factor levels so all-zero dummy columns don't enter the models
final.cox <- final %>% mutate(across(where(is.factor), droplevels))

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

cox.any <- run.cox("Any.Abx", "Any Antibiotic")
write_csv(cox.any, file.path(output, "COX1_any_cancer.csv"))

################################################################################
## 5. Cox — number of antibiotic classes
################################################################################

cox.cls <- run.cox("Abx.Classes", "ABX Classes")
write_csv(cox.cls, file.path(output, "COX2_abx_classes.csv"))

################################################################################
## 6. Cox — individual antibiotic classes (all five simultaneously)
################################################################################

ind.vars <- paste(c("Abx.Penicillin", "Abx.Macrolide", "Abx.Cephalosporin",
                     "Abx.Tetracycline", "Abx.TmpSmx"), collapse = " + ")

cox.ind <- run.cox(ind.vars, "Individual ABX Classes")
write_csv(cox.ind, file.path(output, "COX3_individual_abx.csv"))
