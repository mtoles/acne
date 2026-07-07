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

## Keep ALL patients, including those with multiple cancer types. Each cancer
## type gets its own survival clock (built in run.type.cox), so multi-cancer
## patients no longer have to be dropped.
final.cox <- final %>%
  ## Drop empty factor levels so all-zero dummy columns don't enter the models
  mutate(across(where(is.factor), droplevels))

## Fully adjusted covariate set (adj.vars) is defined in 1_cleanup.R

## Cancer type columns and labels are taken automatically from 1_cleanup.R
## (ca.type.cols / ca.type.labels) — one entry per raw cancer_outcome__ column.
## Types with too few events are skipped (see MIN.TYPE.EVENTS in section 5).
MIN.TYPE.EVENTS <- 10

## Benjamini-Hochberg FDR level for the across-cancer-type multiple-testing correction
## (section 6). The BH family is the fully-adjusted Any.Abx="Yes" exposure p-value, one per
## cancer type modelled below.
BH.Q <- 0.05

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

  dt.col <- paste0("dxdt__", ca.col)

  ## Type-specific clock (marginal cause-specific; competing cancers ignored):
  ##   event = this cancer type
  ##   time  = start.dt -> (this type's dx date if case, else censor.dt)
  ## Non-cases — including patients with a DIFFERENT cancer — stay at risk until
  ## their normal censor, so no one is dropped for having multiple cancers.
  d <- data %>%
    mutate(
      type.event = as.integer(.data[[ca.col]] == "Yes"),
      type.time  = as.numeric(if_else(type.event == 1L,
                                      .data[[dt.col]], censor.dt) - start.dt) / 365.25
    ) %>%
    filter(!is.na(type.time), type.time > 0)

  ## Cancer-specific comorbidity/exposure covariates for THIS cancer: the page-2 binaries
  ## (cancerspecific__pre_index__<ca.col>__<category>). No age-adjusted model and no universal
  ## comorbidity categories — see type.adj.vars in 1_cleanup.R. usable.covars drops any covariate
  ## that is constant in this cancer's risk set (e.g. an exposure no case/control happens to have).
  cs.prefix    <- paste0("cancerspecific__pre_index__", ca.col, "__")
  cs.for.type  <- names(d)[startsWith(names(d), cs.prefix)]
  full.covars  <- c(usable.covars(d, type.adj.vars), usable.covars(d, cs.for.type))

  ## Backtick-quote covariates: cancerspecific__ names contain ';', '-', and "'" (from the cancer
  ## label) which would otherwise break R's formula parser.
  f.unadj <- as.formula("Surv(type.time, type.event) ~ Any.Abx")
  f.full  <- as.formula(paste0("Surv(type.time, type.event) ~ Any.Abx + ",
                                paste(sprintf("`%s`", full.covars), collapse = " + ")))

  fit.unadj <- coxph(f.unadj, data = d)
  fit.full  <- coxph(f.full,  data = d)

  fmt <- function(fit, model.label) {
    tbl_regression(fit, exponentiate = TRUE) %>%
      as_tibble() %>%
      mutate(cancer.type = cancer.label, model = model.label)
  }

  tidy <- bind_rows(
    fmt(fit.unadj, "Unadjusted"),
    fmt(fit.full,  "Fully adjusted")
  )

  ## Exact fully-adjusted Any.Abx="Yes" exposure result for the BH correction (section 6).
  ## gtsummary's p-value strings are rounded/thresholded (">0.9", "<0.001") and cannot be
  ## ranked, so pull the unrounded coefficient p-value straight from the coxph fit.
  sc      <- summary(fit.full)$coefficients
  abx.row <- which(rownames(sc) == "Any.AbxYes")
  stopifnot(length(abx.row) == 1)
  bh <- tibble(cancer.type = cancer.label,
               n.events    = fit.full$nevent,
               HR          = sc[abx.row, "exp(coef)"],
               p.value     = sc[abx.row, "Pr(>|z|)"])

  list(tidy = tidy, bh = bh)
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

results <- lapply(which(keep), function(i) {
  run.type.cox(final.cox, ca.type.cols[i], ca.type.labels[i])
})

cox.types <- bind_rows(lapply(results, `[[`, "tidy"))
write_csv(cox.types, file.path(output, "COX4_cancer_types.csv"))

################################################################################
## 6. Benjamini-Hochberg correction across cancer types
##
## Multiple-testing family: the fully-adjusted Any.Abx="Yes" exposure p-value, one per cancer
## type modelled above. p.adjust(method = "BH") applies the Benjamini-Hochberg step-up and
## returns adjusted p-values (q-values); a type is significant when its q-value <= Q.
################################################################################

cox.types.bh <- bind_rows(lapply(results, `[[`, "bh")) %>%
  arrange(p.value) %>%
  mutate(
    rank           = row_number(),
    p.adjusted     = p.adjust(p.value, method = "BH"),   # BH q-value
    bh.significant = p.adjusted <= BH.Q
  )

cat(sprintf("\nBenjamini-Hochberg (fully-adjusted Any.Abx, Q = %.2f, m = %d types):\n",
            BH.Q, nrow(cox.types.bh)))
print(as.data.frame(cox.types.bh), digits = 3)

write_csv(cox.types.bh, file.path(output, "COX4_cancer_types_bh.csv"))
