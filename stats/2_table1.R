# title: "Antibiotic Exposure and Cancer Risk - Table 1 & Table 1A"
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
## 3. shared SMD helper
################################################################################

smd.stat <- function(data, variable, by, ...) {
  tryCatch(round(abs(smd::smd(x = data[[variable]], g = data[[by]])$estimate), 3),
           error = function(e) NA_real_)
}

################################################################################
## 4. Table 1 — baseline characteristics by cancer outcome
################################################################################

df1 <- final %>%
  select(Cancer.Dx,
         Age,
         Age.group,
         Sex,
         Race,
         BMI,
         BMI.Category,
         Smoking,
         Alcohol,
         Contraceptives,
         Fam.Cancer,
         Comorbidities,
         Prior.Cancer,
         Transplant,
         Any.Abx,
         Abx.Classes)

table1 <- tbl_summary(df1,
                      by        = Cancer.Dx,
                      label     = list(Age            ~ "Age (years)",
                                       Age.group      ~ "Age group",
                                       Sex            ~ "Sex",
                                       Race           ~ "Race / ethnicity",
                                       BMI            ~ "BMI (kg/m²)",
                                       BMI.Category   ~ "BMI category",
                                       Smoking        ~ "Smoking status",
                                       Alcohol        ~ "Alcohol use",
                                       Contraceptives ~ "Contraceptive use",
                                       Fam.Cancer     ~ "Family history of cancer",
                                       Comorbidities  ~ "No. of comorbid conditions",
                                       Prior.Cancer   ~ "No. of preexisting cancers",
                                       Transplant     ~ "Prior organ transplant",
                                       Any.Abx        ~ "Any antibiotic exposure",
                                       Abx.Classes    ~ "No. of antibiotic classes"),
                      statistic = list(all_continuous()  ~ "{median} ({IQR})",
                                       all_categorical() ~ "{n} ({p}%)"),
                      digits    = list(all_continuous() ~ 1),
                      missing   = "no") %>%
  add_overall(last = FALSE, col_label = "**Overall**  \nN = {N}") %>%
  add_p(test      = list(all_continuous()  ~ "kruskal.test",
                         all_categorical() ~ "chisq.test"),
        pvalue_fun = label_style_pvalue(digits = 3)) %>%
  add_stat(fns = everything() ~ smd.stat) %>%
  bold_labels() %>%
  modify_header(label      ~ "**Characteristic**",
                stat_1     ~ "**No Cancer**  \nN = {n}",
                stat_2     ~ "**Cancer**  \nN = {n}",
                add_stat_1 ~ "**SMD**") %>%
  modify_spanning_header(c(stat_1, stat_2) ~ "**Cancer Outcome**") %>%
  modify_caption("**Table 1.** Baseline Characteristics by Cancer Outcome Among Acne Patients") %>%
  modify_footnote(all_stat_cols() ~ "Median (interquartile range) for continuous; n (%) for categorical variables.",
                  p.value         ~ "Kruskal-Wallis test (continuous); Pearson χ² test (categorical).",
                  add_stat_1      ~ "SMD: Standardized Mean Difference (|SMD| >0.10 suggests meaningful imbalance).")

write.csv(gtsummary::as_tibble(table1),
          file.path(output, "table1_cancer_outcome.csv"),
          row.names = FALSE)

################################################################################
## 5. Table 1A — baseline characteristics by antibiotic exposure (cohort view)
################################################################################

df1a <- final %>%
  select(Any.Abx,
         Age,
         Age.group,
         Sex,
         Race,
         BMI,
         BMI.Category,
         Smoking,
         Alcohol,
         Contraceptives,
         Fam.Cancer,
         Comorbidities,
         Prior.Cancer,
         Transplant,
         Cancer.Dx)

table1a <- tbl_summary(df1a,
                       by        = Any.Abx,
                       label     = list(Age            ~ "Age (years)",
                                        Age.group      ~ "Age group",
                                        Sex            ~ "Sex",
                                        Race           ~ "Race / ethnicity",
                                        BMI            ~ "BMI (kg/m²)",
                                        BMI.Category   ~ "BMI category",
                                        Smoking        ~ "Smoking status",
                                        Alcohol        ~ "Alcohol use",
                                        Contraceptives ~ "Contraceptive use",
                                        Fam.Cancer     ~ "Family history of cancer",
                                        Comorbidities  ~ "No. of comorbid conditions",
                                        Prior.Cancer   ~ "No. of preexisting cancers",
                                        Transplant     ~ "Prior organ transplant",
                                        Cancer.Dx      ~ "Cancer diagnosis"),
                       statistic = list(all_continuous()  ~ "{median} ({IQR})",
                                        all_categorical() ~ "{n} ({p}%)"),
                       digits    = list(all_continuous() ~ 1),
                       missing      = "ifany",
                       missing_text = "Unknown") %>%
  add_overall(last = FALSE, col_label = "**Overall**  \nN = {N}") %>%
  add_p(test      = list(all_continuous()  ~ "kruskal.test",
                         all_categorical() ~ "chisq.test"),
        pvalue_fun = label_style_pvalue(digits = 3)) %>%
  add_stat(fns = everything() ~ smd.stat) %>%
  bold_labels() %>%
  modify_header(label      ~ "**Characteristic**",
                stat_1     ~ "**No Antibiotic**  \nN = {n}",
                stat_2     ~ "**Antibiotic**  \nN = {n}",
                add_stat_1 ~ "**SMD**") %>%
  modify_spanning_header(c(stat_1, stat_2) ~ "**Antibiotic Exposure**") %>%
  modify_caption("**Table 1A.** Baseline Characteristics by Antibiotic Exposure Among Acne Patients") %>%
  modify_footnote(all_stat_cols() ~ "Median (interquartile range) for continuous; n (%) for categorical variables.",
                  p.value         ~ "Kruskal-Wallis test (continuous); Pearson χ² test (categorical).",
                  add_stat_1      ~ "SMD: Standardized Mean Difference (|SMD| >0.10 suggests meaningful imbalance).")

write.csv(gtsummary::as_tibble(table1a),
          file.path(output, "table1a_abx_exposure.csv"),
          row.names = FALSE)
