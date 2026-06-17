# title: "Antibiotic Exposure and Cancer Risk - Cancer Type-Specific Cox Models"
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

## Restrict to patients with at most one cancer type so that follow.time
## (time to first cancer) is unambiguous for type-specific event coding.
final.cox <- final %>%
  mutate(n.ca.types = rowSums(across(all_of(ca.type.cols),
                                     ~ as.integer(.x == "Yes")),
                              na.rm = TRUE)) %>%
  filter(n.ca.types <= 1)

adj.vars <- c("Age", "Sex", "Race", "BMI.Category", "Smoking",
              "Alcohol", "Contraceptives", "Fam.Cancer", "Transplant")

## Cancer type columns and labels are taken automatically from 1_cleanup.R
## (ca.type.cols / ca.type.labels) — one entry per raw cancer_outcome__ column.
## Types with too few events are skipped (see MIN.TYPE.EVENTS in section 5).
MIN.TYPE.EVENTS <- 10

################################################################################
## 4. helper function
##
##    Runs unadj / age-adj / fully-adj Cox for one cancer type (Any.Abx as
##    exposure) and returns a tidy tibble.
##
##    NOTE: fully adjusted models for small cancer types (bladder n=114,
##    ovary n=112, brain n=91, pancreas n=75, cervix n=60) have <10 events
##    per predictor. Interpret with caution — CIs will be wide.
################################################################################

run.type.cox <- function(data, ca.col, cancer.label) {

  d <- data %>%
    mutate(type.event = as.integer(.data[[ca.col]] == "Yes"))

  f.unadj <- as.formula("Surv(follow.time, type.event) ~ Any.Abx")
  f.age   <- as.formula("Surv(follow.time, type.event) ~ Any.Abx + Age")
  f.full  <- as.formula(paste0("Surv(follow.time, type.event) ~ Any.Abx + ",
                                paste(adj.vars, collapse = " + ")))

  fmt <- function(fit, model.label) {
    tbl_regression(fit, exponentiate = TRUE) %>%
      as_tibble() %>%
      mutate(cancer.type = cancer.label, model = model.label)
  }

  bind_rows(
    fmt(coxph(f.unadj, data = d), "Unadjusted"),
    fmt(coxph(f.age,   data = d), "Age-adjusted"),
    fmt(coxph(f.full,  data = d), "Fully adjusted")
  )
}

################################################################################
## 5. run models for every cancer type with enough events
##
## A type with too few events cannot support a (let alone fully adjusted) Cox
## model, so we skip it rather than fit a degenerate / non-converging model.
################################################################################

keep <- vapply(seq_along(ca.type.cols), function(i) {
  n.events <- sum(final.cox[[ca.type.cols[i]]] == "Yes", na.rm = TRUE)
  if (n.events < MIN.TYPE.EVENTS) {
    message(sprintf("Cox skipped: %-55s %d events (< %d)",
                    ca.type.labels[i], n.events, MIN.TYPE.EVENTS))
    FALSE
  } else TRUE
}, logical(1))

cox.types <- bind_rows(lapply(which(keep), function(i) {
  run.type.cox(final.cox, ca.type.cols[i], ca.type.labels[i])
}))

write_csv(cox.types, file.path(output, "COX4_cancer_types.csv"))
