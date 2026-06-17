# title: "Antibiotic Exposure and Cancer Risk - Cancer Type-Specific Kaplan-Meier Curves"
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
final.km <- final %>%
  mutate(n.ca.types = rowSums(across(all_of(ca.type.cols),
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
## 5. cancer type-specific KM figures (one per raw cancer type, automatic)
##
## Types with fewer than MIN.TYPE.EVENTS events in the single-cancer subset are
## skipped — a KM curve built on a handful of events is not interpretable and
## can produce a degenerate fit.
################################################################################

MIN.TYPE.EVENTS <- 10

slugify <- function(x) tolower(gsub("_+", "_", gsub("[^A-Za-z0-9]+", "_", x)))

for (i in seq_along(ca.type.cols)) {
  ca.col   <- ca.type.cols[i]
  lbl      <- ca.type.labels[i]
  n.events <- sum(final.km[[ca.col]] == "Yes", na.rm = TRUE)

  if (n.events < MIN.TYPE.EVENTS) {
    message(sprintf("KM skipped: %-55s %d events (< %d)", lbl, n.events, MIN.TYPE.EVENTS))
    next
  }

  plot.type.km(final.km, ca.col, lbl, paste0("KM4_", slugify(lbl), ".pdf"))
}
