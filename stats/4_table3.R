# title: "Antibiotic Exposure and Cancer Risk - Table 3: Cancer Type Breakdown Among Cancer Patients"
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
## 3. table 3
################################################################################

df <- final %>%
  filter(Cancer.Dx == "Cancer") %>%
  select(Any.Abx, all_of(ca.type.cols))

## Display labels for each raw cancer type column (auto-generated in 1_cleanup.R)
type.labels <- setNames(as.list(ca.type.labels), ca.type.cols)

table3 <- tbl_summary(df,
                      by       = Any.Abx,
                      label    = type.labels,
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
