# title: "Antibiotic Exposure and Cancer Risk - Kaplan-Meier Curves"
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

final.km <- final 

## Round up to nearest 5 so risk table columns fall exactly on x-axis breaks
xlim.max  <- ceiling(max(final.km$follow.time, na.rm = TRUE) / 5) * 5
pal.abx   <- c("#4A90D9", "#E05252")
pal.cls   <- c("#4A90D9", "#5BAD6F", "#E8A838", "#E05252")
pal.ind   <- c("#4A90D9", "#5BAD6F", "#E8A838", "#E05252", "#9B59B6")

################################################################################
## 4. KM curve — any antibiotic exposure
################################################################################

km.fit.abx <- survfit(Surv(follow.time, surv.event) ~ Any.Abx, data = final.km)

p.abx <- ggsurvplot(
  km.fit.abx,
  data                = final.km,
  conf.int            = TRUE,
  risk.table          = TRUE,
  risk.table.fontsize = 5.25,
  palette             = pal.abx,
  legend.labs         = c("No Antibiotic", "Antibiotic"),
  legend.title        = "",
  xlab                = "Time (years)",
  ylab                = "Cancer-Free Survival Probability",
  pval                = TRUE,
  pval.size           = 6,
  xlim                = c(0, xlim.max),
  break.time.by       = 5,
  ggtheme             = theme_classic(base_size = 22),
  tables.theme        = theme_cleantable()
)

pdf(file.path(output, "KM1_any_abx.pdf"), width = 14, height = 11)
print(p.abx, newpage = FALSE)
dev.off()

################################################################################
## 5. KM curve — number of antibiotic classes
################################################################################

km.fit.cls <- survfit(Surv(follow.time, surv.event) ~ Abx.Classes, data = final.km)

p.cls <- ggsurvplot(
  km.fit.cls,
  data                = final.km,
  conf.int            = TRUE,
  risk.table          = TRUE,
  risk.table.fontsize = 5.25,
  palette             = pal.cls,
  legend.labs         = c("None", "1 Class", "2 Classes", "3+ Classes"),
  legend.title        = "",
  xlab                = "Time (years)",
  ylab                = "Cancer-Free Survival Probability",
  pval                = TRUE,
  pval.size           = 6,
  xlim                = c(0, xlim.max),
  break.time.by       = 5,
  ggtheme             = theme_classic(base_size = 22),
  tables.theme        = theme_cleantable()
)

pdf(file.path(output, "KM2_abx_classes.pdf"), width = 14, height = 11)
print(p.cls, newpage = FALSE)
dev.off()

################################################################################
## 6. KM curve — individual antibiotic classes
##
##    Restricted to patients who received exactly one antibiotic class so that
##    groups are mutually exclusive.
################################################################################

final.long <- final.km %>%
  pivot_longer(cols      = c(Abx.Penicillin, Abx.Macrolide,
                              Abx.Cephalosporin, Abx.Tetracycline, Abx.TmpSmx),
               names_to  = "Abx.Type",
               values_to = "Received") %>%
  mutate(Abx.Type = case_when(
    Abx.Type == "Abx.Penicillin"    ~ "Penicillin",
    Abx.Type == "Abx.Macrolide"     ~ "Macrolide",
    Abx.Type == "Abx.Cephalosporin" ~ "Cephalosporin",
    Abx.Type == "Abx.Tetracycline"  ~ "Tetracycline",
    Abx.Type == "Abx.TmpSmx"        ~ "TMP-SMX"),
    Abx.Type = factor(Abx.Type,
                      levels = c("Penicillin", "Macrolide",
                                 "Cephalosporin", "Tetracycline", "TMP-SMX")))

final.exp <- final.long %>%
  filter(Received == "Yes", Abx.Classes == "1 class")

km.fit.ind <- survfit(Surv(follow.time, surv.event) ~ Abx.Type, data = final.exp)

p.ind <- ggsurvplot(
  km.fit.ind,
  data                = final.exp,
  conf.int            = TRUE,
  risk.table          = TRUE,
  risk.table.fontsize = 5.25,
  palette             = pal.ind,
  legend.labs         = c("Penicillin", "Macrolide", "Cephalosporin",
                           "Tetracycline", "TMP-SMX"),
  legend.title        = "",
  xlab                = "Time (years)",
  ylab                = "Cancer-Free Survival Probability",
  pval                = TRUE,
  pval.size           = 6,
  xlim                = c(0, xlim.max),
  break.time.by       = 5,
  ggtheme             = theme_classic(base_size = 22),
  tables.theme        = theme_cleantable()
)

pdf(file.path(output, "KM3_individual_abx.pdf"), width = 14, height = 12)
print(p.ind, newpage = FALSE)
dev.off()
