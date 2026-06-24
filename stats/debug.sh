#!/usr/bin/env bash
# CUMULATIVE ablation of the changes between Daniel's original stats code and current
# HEAD, measured ONLY on COX1 (all-cancer, fully adjusted) Any.Abx HR + p.
#
# Starting from Daniel's exact config, each row turns ON one more change, in rough
# order of expected magnitude of p-change. The Δp column is each variable's marginal
# effect (vs the row above). Same N=28950 cohort throughout.
#
# Variables (in sweep order):
#   1 prior_cancer    Prior.Cancer added to adj.vars
#   2 comorbidities   Comorbidities added to adj.vars
#   3 family_merge    family history pre_index -> pre_index + treatment
#
# Smoking is always Daniel's pre-index pooled status (status only, no amount).
# Row 1 (daniel) should reproduce orig_daniel; the last row is current HEAD.

set -euo pipefail
cd "$(dirname "$0")"
export R_LIBS_USER="$(pwd)/R_libs"

OUT=/home/mtoles/acne/tmp/stats_debug.md
OUT_CSV=/home/mtoles/acne/tmp/stats_debug.csv
FULL_CSV=/home/mtoles/acne/tmp/stats_debug_full.csv   # full Cox coefficient table, all terms, all configs
mkdir -p "$(dirname "$OUT")"
RESULTS=""

# Header for the full coefficient table; debug_cox1.R appends one row per term per config.
echo "config,term,HR,coef,se,z,p,ci_low,ci_high" > "$FULL_CSV"

#        label              fam comorb prior
COMBOS=(
  "daniel__________      0 0 0"
#   "+prior_cancer___      0 0 1"
#   "+comorbidities__      0 1 0"
#   "+family_merge___      1 0 0"
#   "current(=HEAD)__      0 1 1"
)

echo "Running ${#COMBOS[@]} COX1 cumulative ablation fits..."
for c in "${COMBOS[@]}"; do
  read -r label fam comorb prior <<< "$c"
  echo "  $label"
  RESULTS+=$(Rscript debug_cox1.R "$label" "$fam" "$comorb" "$prior" "$FULL_CSV")$'\n'
done
echo "Wrote $FULL_CSV"

# ---- write raw CSV (one row per fit) ----
{
  echo "label,family_merge,comorbidities,prior_cancer,N,HR,se,z,p,vif"
  printf '%s' "$RESULTS"
} > "$OUT_CSV"
echo "Wrote $OUT_CSV"

# ---- write markdown ----
{
  echo "# COX1 (all-cancer, fully adjusted) — cumulative change ablation"
  echo
  echo "Daniel -> current HEAD, turning on one change per row in order of expected"
  echo "magnitude. Effect = Any.Abx hazard ratio + p from the fully adjusted Cox model."
  echo "Δp is the change in p for that row relative to the daniel baseline (row 1)."
  echo "Same N=28,950 cohort throughout."
  echo
  echo "Columns: fam_merge=family pre_index+treatment, comorb=Comorbidities covariate,"
  echo "prior=Prior.Cancer covariate. (Smoking is always pre-index pooled status.)"
  echo "HR/se/z/p are for Any.Abx (z=log(HR)/se, p from z). When HR is flat but p moves,"
  echo "the change lives in se. vif = exposure variance inflation from the adjustment set"
  echo "(1/(1-R^2) of Any.Abx regressed on covariates); vif>1 inflates se -> larger p."
  echo
  chk() { [ "$1" = "1" ] && echo "✓" || echo "·"; }
  # char-count (UTF-8 safe) padders; %*s only emits spaces so no multibyte issue
  lpad() { local n=$(( $2 - ${#1} )); ((n<0))&&n=0; printf "%s%*s" "$1" "$n" ""; }
  rpad() { local n=$(( $2 - ${#1} )); ((n<0))&&n=0; printf "%*s%s" "$n" "" "$1"; }
  cpad() { local t=$(( $2 - ${#1} )); ((t<0))&&t=0; local l=$((t/2)); printf "%*s%s%*s" "$l" "" "$1" "$((t-l))" ""; }
  dash() { printf "%*s" "$1" "" | tr ' ' '-'; }
  # column widths: config fam comorb prior N HR se z p Δp vif
  Wc=22 Wf=9 Wo=6 Wp=5 Wn=5 Wh=5 Wse=6 Wz=7 Wpv=6 Wd=6 Wv=5
  row() {
    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
      "$(lpad "$1" $Wc)" "$(cpad "$2" $Wf)" "$(cpad "$3" $Wo)" "$(cpad "$4" $Wp)" \
      "$(rpad "$5" $Wn)" "$(rpad "$6" $Wh)" "$(rpad "$7" $Wse)" "$(rpad "$8" $Wz)" \
      "$(rpad "$9" $Wpv)" "$(rpad "${10}" $Wd)" "$(rpad "${11}" $Wv)"
  }
  row "config" "fam_merge" "comorb" "prior" "N" "HR" "se" "z" "p" "Δp" "vif"
  printf "|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|\n" \
    ":$(dash $((Wc+1)))" ":$(dash $Wf):" ":$(dash $Wo):" ":$(dash $Wp):" \
    "$(dash $((Wn+1))):" "$(dash $((Wh+1))):" "$(dash $((Wse+1))):" "$(dash $((Wz+1))):" \
    "$(dash $((Wpv+1))):" "$(dash $((Wd+1))):" "$(dash $((Wv+1))):"
  base=""
  while IFS=, read -r label fam comorb prior n hr se z p vif; do
    [ -z "$label" ] && continue
    if [ -z "$base" ]; then base=$p; dp="—"; else dp=$(awk "BEGIN{printf \"%+.3f\", $p-$base}"); fi
    row "$label" "$(chk $fam)" "$(chk $comorb)" "$(chk $prior)" "$n" "$hr" "$se" "$z" "$p" "$dp" "$vif"
  done <<< "$(printf '%s' "$RESULTS")"
} > "$OUT"

echo "Wrote $OUT"
cat "$OUT"
