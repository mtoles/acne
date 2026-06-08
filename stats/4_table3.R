# title: "Antibiotic Exposure and Cancer Risk - Table 3: Cancer Type Breakdown Among Cancer Patients"
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
## 3. table 3
################################################################################

df <- final %>%
  filter(Cancer.Dx == "Cancer") %>%
  select(Any.Abx,
         Ca.Skin.NM,
         Ca.Breast,
         Ca.Melanoma,
         Ca.Thyroid,
         Ca.Lung,
         Ca.Colorectal,
         Ca.Prostate,
         Ca.Lymphoma,
         Ca.Uterine,
         Ca.Kidney,
         Ca.Bladder,
         Ca.Ovary,
         Ca.Leukemia,
         Ca.Brain,
         Ca.Pancreas,
         Ca.Cervix)

table3 <- tbl_summary(df,
                      by       = Any.Abx,
                      label    = list(Ca.Skin.NM    ~ "Non-melanoma skin cancer",
                                      Ca.Breast     ~ "Breast cancer",
                                      Ca.Melanoma   ~ "Melanoma",
                                      Ca.Thyroid    ~ "Thyroid cancer",
                                      Ca.Lung       ~ "Lung cancer",
                                      Ca.Colorectal ~ "Colorectal cancer",
                                      Ca.Prostate   ~ "Prostate cancer",
                                      Ca.Lymphoma   ~ "Lymphoma",
                                      Ca.Uterine    ~ "Uterine / endometrial cancer",
                                      Ca.Kidney     ~ "Kidney cancer",
                                      Ca.Bladder    ~ "Bladder cancer",
                                      Ca.Ovary      ~ "Ovarian cancer",
                                      Ca.Leukemia   ~ "Leukemia",
                                      Ca.Brain      ~ "Brain cancer",
                                      Ca.Pancreas   ~ "Pancreatic cancer",
                                      Ca.Cervix     ~ "Cervical cancer"),
                      statistic = list(all_categorical() ~ "{n} ({p}%)"),
                      missing      = "ifany",
                      missing_text = "Unknown") %>%
  add_overall(last = FALSE, col_label = "**Overall**  \nN = {N}") %>%
  add_p(test = list(all_categorical() ~ "chisq.test"),
        pvalue_fun = label_style_pvalue(digits = 3)) %>%
  bold_labels() %>%
  modify_header(label  ~ "**Cancer Type**",
                stat_1 ~ "**No Antibiotics**  \nN = {n}",
                stat_2 ~ "**Antibiotics**  \nN = {n}") %>%
  modify_spanning_header(c(stat_1, stat_2) ~ "**Any Antibiotic Exposure**") %>%
  modify_caption("**Table 3.** Cancer Type Distribution Among Cancer Patients by Antibiotic Exposure Status") %>%
  modify_footnote(all_stat_cols() ~ "n (%) for each cancer type; patients may have more than one cancer type.",
                  p.value         ~ "Pearson χ² test.")

write.csv(gtsummary::as_tibble(table3), file.path(output, "table3_cancer_types.csv"),
          row.names = FALSE)
