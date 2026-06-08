# title: "Antibiotic Exposure and Cancer Risk - Cancer Type-Specific Cox Models"
# author: Daniel Kim

################################################################################
## 1. clear all
################################################################################

rm(list = ls(all.names = TRUE))

################################################################################
## 2. set locations
################################################################################

source('/Users/danielkim/Partners HealthCare Dropbox/Daniel Kim/Research/Barbieri/1_cleanup.R')

################################################################################
## 3. data prep
################################################################################

## Restrict to patients with at most one cancer type so that follow.time
## (time to first cancer) is unambiguous for type-specific event coding.
final.cox <- final %>%
  mutate(n.ca.types = rowSums(across(starts_with("Ca."),
                                     ~ as.integer(.x == "Yes")),
                              na.rm = TRUE)) %>%
  filter(n.ca.types <= 1)

adj.vars <- c("Age", "Sex", "Race", "BMI.Category", "Smoking",
              "Alcohol", "Contraceptives", "Fam.Cancer", "Transplant")

## Cancer type columns and labels
ca.types <- list(
  list(col = "Ca.Skin.NM",    label = "Non-Melanoma Skin Cancer"),
  list(col = "Ca.Breast",     label = "Breast Cancer"),
  list(col = "Ca.Melanoma",   label = "Melanoma"),
  list(col = "Ca.Thyroid",    label = "Thyroid Cancer"),
  list(col = "Ca.Lung",       label = "Lung Cancer"),
  list(col = "Ca.Colorectal", label = "Colorectal Cancer"),
  list(col = "Ca.Prostate",   label = "Prostate Cancer"),
  list(col = "Ca.Lymphoma",   label = "Lymphoma"),
  list(col = "Ca.Uterine",    label = "Uterine Cancer"),
  list(col = "Ca.Kidney",     label = "Kidney Cancer"),
  list(col = "Ca.Bladder",    label = "Bladder Cancer"),
  list(col = "Ca.Ovary",      label = "Ovarian Cancer"),
  list(col = "Ca.Leukemia",   label = "Leukemia"),
  list(col = "Ca.Brain",      label = "Brain Cancer"),
  list(col = "Ca.Pancreas",   label = "Pancreatic Cancer"),
  list(col = "Ca.Cervix",     label = "Cervical Cancer")
)

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
## 5. run models for all 16 cancer types
################################################################################

cox.types <- bind_rows(lapply(ca.types, function(x) {
  run.type.cox(final.cox, x$col, x$label)
}))

write_csv(cox.types, file.path(output, "COX4_cancer_types.csv"))
