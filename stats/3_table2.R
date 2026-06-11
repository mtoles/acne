# title: "Antibiotic Exposure and Cancer Risk - Table 2: Antibiotic Breakdown Among ABX Users"
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
## 3. table 2
################################################################################

df <- final %>%
  filter(Any.Abx == "Yes") %>%
  select(Cancer.Dx,
         Abx.Classes,
         Abx.Duration,
         Abx.Duration.Cat,
         Abx.Penicillin,
         Abx.Macrolide,
         Abx.Cephalosporin,
         Abx.Tetracycline,
         Abx.TmpSmx)

table2 <- tbl_summary(df,
                      by       = Cancer.Dx,
                      label    = list(Abx.Classes       ~ "No. of antibiotic classes",
                                      Abx.Duration      ~ "Total ABX duration (days)",
                                      Abx.Duration.Cat  ~ "Total ABX duration category",
                                      Abx.Penicillin    ~ "Penicillin (amoxicillin)",
                                      Abx.Macrolide     ~ "Macrolide (azithromycin)",
                                      Abx.Cephalosporin ~ "Cephalosporin (cephalexin)",
                                      Abx.Tetracycline  ~ "Tetracycline",
                                      Abx.TmpSmx        ~ "TMP-SMX"),
                      statistic = list(all_continuous()  ~ "{median} ({IQR})",
                                       all_categorical() ~ "{n} ({p}%)"),
                      digits       = list(all_continuous() ~ 1),
                      missing      = "ifany",
                      missing_text = "Unknown") %>%
  add_overall(last = FALSE, col_label = "**Overall**  \nN = {N}") %>%
  add_p(test = list(all_continuous()  ~ "kruskal.test",
                    all_categorical() ~ "chisq.test"),
        pvalue_fun = label_style_pvalue(digits = 3)) %>%
  bold_labels() %>%
  modify_header(label  ~ "**Antibiotic Class**",
                stat_1 ~ "**No Cancer**  \nN = {n}",
                stat_2 ~ "**Cancer**  \nN = {n}") %>%
  modify_spanning_header(c(stat_1, stat_2) ~ "**Cancer Outcome**") %>%
  modify_caption("**Table 2.** Antibiotic Class Breakdown Among Antibiotic-Exposed Acne Patients by Cancer Outcome") %>%
  modify_footnote(all_stat_cols() ~ "Median (interquartile range) for continuous; n (%) for categorical variables.",
                  p.value         ~ "Kruskal-Wallis test (continuous); Pearson χ² test (categorical).")

write.csv(gtsummary::as_tibble(table2), file.path(output, "table2_abx_breakdown.csv"),
          row.names = FALSE)
