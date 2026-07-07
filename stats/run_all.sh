#!/usr/bin/env bash
# Run the full R stats pipeline.
#
# 1_cleanup.R is sourced by each of scripts 2-13, so it is not run directly.
# Each script rebuilds the `final` data frame and writes its output into
# stats/eda_output/ (created by 1_cleanup.R).
#
# Usage:
#   ./run_all.sh           # install missing packages (if any), then run 2-13
#   ./run_all.sh --no-install   # skip the package-install step

set -euo pipefail

cd "$(dirname "$0")"   # always operate inside the stats dir

# Local package library (the system lib /usr/local/lib/R/site-library is not
# writable). Exporting R_LIBS_USER makes it the install target AND puts it on
# .libPaths() for every Rscript call below, so the scripts find the packages.
export R_LIBS_USER="$(pwd)/R_libs"
mkdir -p "$R_LIBS_USER"
echo "using R library: $R_LIBS_USER"

# The `fs` package (a dependency of tidyverse/gt/gtsummary) needs system libuv
# to compile, which isn't installed and needs sudo. This tells `fs` to build its
# own bundled static libuv instead, so no apt/sudo is required.
export USE_BUNDLED_LIBUV=1

REPO_PKGS='c("tidyverse","gtsummary","survminer","lubridate","smd","survival","broom","broom.helpers")'

if [[ "${1:-}" != "--no-install" ]]; then
  echo "===== checking / installing R packages ====="
  Rscript -e "
    pkgs <- $REPO_PKGS
    missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
    if (length(missing)) {
      cat('installing:', paste(missing, collapse = ', '), '\n')
      install.packages(missing, lib = Sys.getenv('R_LIBS_USER'),
                       repos = 'https://cloud.r-project.org')
    } else {
      cat('all packages already installed\n')
    }
  "
fi

echo "===== sanity check (1_cleanup.R) ====="
Rscript -e "
  source('1_cleanup.R')
  cat(nrow(final), 'rows x', ncol(final), 'cols\n')
  print(table(final\$Cancer.Dx, final\$Any.Abx))
"

echo "===== running pipeline (scripts 2-13) ====="
for f in 2_table1.R 3_table2.R 4_table3.R 5_km.R 6_cox.R 7_km_types.R 8_cox_types.R \
         9_cox_dose.R 10_cox_dose_max.R 11_cox_long_only.R 12_cox_longest_only.R \
         13_cox_course_count.R; do
  echo "----- $f -----"
  Rscript "$f"
done

echo "===== sanity checks (sanity_check.py) ====="
# Validates records / pooled_records.csv / COX outputs. Exits non-zero (fails this script
# under set -e) if any hard CHECK fails. Uses the project venv for pandas/tqdm.
/home/mtoles/acne/.venv/bin/python sanity_check.py

echo "===== done. outputs in: $(pwd)/eda_output ====="
ls -1 eda_output
