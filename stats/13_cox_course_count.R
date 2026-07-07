# title: "Antibiotic Exposure and Cancer Risk - Cox Dose-Response (number of antibiotic courses)"
# author: Daniel Kim
#
# Experimental variable = the NUMBER OF ANTIBIOTIC COURSES a patient received, binned 1 / 2 / 3+,
# with "None" (no antibiotic) as the reference. A "course" is one unique structured prescription
# date in the treatment window (same-day prescriptions = one course); see count_abx_courses_by_date
# in postprocess_records.py and n.abx.courses in 1_cleanup.R. Every exposed patient has >=1
# structured Rx, so there is no "unknown" bin. Mirrors the dose-response scripts (9/10_cox).

################################################################################
## 1. clear all
################################################################################

rm(list = ls(all.names = TRUE))

################################################################################
## 2. set locations
################################################################################

source('/home/mtoles/acne/stats/1_cleanup.R')

################################################################################
## 3. data prep — bin number of antibiotic courses
##
## Reference = "None" (no antibiotic exposure, Any.Abx == "No"). Exposed patients are binned
## by n.abx.courses (# of unique structured Rx dates, built in 1_cleanup.R) into 1 / 2 / 3+.
## Every Any.Abx == "Yes" patient has at least one structured prescription, so there is no
## "unknown" bin -- guarded by the stopifnot below.
################################################################################

## Guard: no exposed patient should have 0 counted courses (would signal a missing structured Rx).
stopifnot(all(final$n.abx.courses[final$Any.Abx == "Yes"] >= 1))

final.cox <- final %>%
  mutate(
    Abx.Courses = case_when(
      Any.Abx == "No"     ~ "None",
      n.abx.courses == 1  ~ "1",
      n.abx.courses == 2  ~ "2",
      n.abx.courses >= 3  ~ "3+"),
    Abx.Courses = factor(Abx.Courses,
                         levels = c("None", "1", "2", "3+"))
  ) %>%
  ## Drop empty factor levels so all-zero dummy columns don't enter the models
  mutate(across(where(is.factor), droplevels))

## Distribution of the course bins (sanity check)
cat("Abx.Courses distribution:\n")
print(table(final.cox$Abx.Courses, useNA = "ifany"))

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
## 4. Cox — dose-response on number of antibiotic courses
################################################################################

cox.courses <- run.cox("Abx.Courses", "Number of ABX Courses")
write_csv(cox.courses, file.path(output, "COX7_abx_course_count.csv"))
