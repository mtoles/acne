# title: "Antibiotic Exposure and Cancer Risk - Cancer Type-Specific Kaplan-Meier Curves"
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
final.km <- final %>%
  mutate(n.ca.types = rowSums(across(starts_with("Ca."),
                                     ~ as.integer(.x == "Yes")),
                              na.rm = TRUE)) %>%
  filter(n.ca.types <= 1)

xlim.max <- ceiling(max(final.km$follow.time, na.rm = TRUE) / 5) * 5
pal.abx  <- c("#4A90D9", "#E05252")

################################################################################
## 4. helper function
################################################################################

plot.type.km <- function(data, ca.col, cancer.label, filename) {

  d <- data %>%
    mutate(type.event = as.integer(.data[[ca.col]] == "Yes"))

  km.fit <- survfit(Surv(follow.time, type.event) ~ Any.Abx, data = d)

  p <- ggsurvplot(
    km.fit,
    data                = d,
    conf.int            = TRUE,
    risk.table          = TRUE,
    risk.table.fontsize = 5.25,
    palette             = pal.abx,
    legend.labs         = c("No Antibiotic", "Antibiotic"),
    legend.title        = "",
    xlab                = "Time (years)",
    ylab                = paste0("Cancer-Free Survival: ", cancer.label),
    pval                = TRUE,
    pval.size           = 6,
    xlim                = c(0, xlim.max),
    break.time.by       = 5,
    ggtheme             = theme_classic(base_size = 22),
    tables.theme        = theme_cleantable()
  )

  pdf(file.path(output, filename), width = 14, height = 11)
  print(p, newpage = FALSE)
  dev.off()
}

################################################################################
## 5. cancer type-specific KM figures
################################################################################

plot.type.km(final.km, "Ca.Skin.NM",    "Non-Melanoma Skin Cancer", "KM4_nmsc.pdf")
plot.type.km(final.km, "Ca.Breast",     "Breast Cancer",            "KM4_breast.pdf")
plot.type.km(final.km, "Ca.Melanoma",   "Melanoma",                 "KM4_melanoma.pdf")
plot.type.km(final.km, "Ca.Thyroid",    "Thyroid Cancer",           "KM4_thyroid.pdf")
plot.type.km(final.km, "Ca.Lung",       "Lung Cancer",              "KM4_lung.pdf")
plot.type.km(final.km, "Ca.Colorectal", "Colorectal Cancer",        "KM4_colorectal.pdf")
plot.type.km(final.km, "Ca.Prostate",   "Prostate Cancer",          "KM4_prostate.pdf")
plot.type.km(final.km, "Ca.Lymphoma",   "Lymphoma",                 "KM4_lymphoma.pdf")
plot.type.km(final.km, "Ca.Uterine",    "Uterine Cancer",           "KM4_uterine.pdf")
plot.type.km(final.km, "Ca.Kidney",     "Kidney Cancer",            "KM4_kidney.pdf")
plot.type.km(final.km, "Ca.Bladder",    "Bladder Cancer",           "KM4_bladder.pdf")
plot.type.km(final.km, "Ca.Ovary",      "Ovarian Cancer",           "KM4_ovary.pdf")
plot.type.km(final.km, "Ca.Leukemia",   "Leukemia",                 "KM4_leukemia.pdf")
plot.type.km(final.km, "Ca.Brain",      "Brain Cancer",             "KM4_brain.pdf")
plot.type.km(final.km, "Ca.Pancreas",   "Pancreatic Cancer",        "KM4_pancreas.pdf")
plot.type.km(final.km, "Ca.Cervix",     "Cervical Cancer",          "KM4_cervix.pdf")
