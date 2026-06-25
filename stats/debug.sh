#!/usr/bin/env bash
# Ablation of the COX1 (all-cancer, fully adjusted) "Any Antibiotic" hazard ratio across
# the changes between Daniel's 6_cox and our current 6_cox. It runs:
#   * the REAL orig_daniel/6_cox.R  (Daniel's code on the .bak snapshot)
#   * the REAL 6_cox.R              (our code on the current pooled_records.csv)
#   * debug_cox1.R over whatever configs you list in the grid below (edit at will).
#
# Each debug config picks a data snapshot and the three code toggles independently, so you
# can isolate any single change. The report lists the ACTUAL params used per row (read back
# from configs.txt), not a fixed assumed layout -- edit the grid freely.
#
# Axes (0 = Daniel, 1 = ours):
#   data           bak (Daniel's pooled_records.csv.bak) | current (ours)
#   followup       follow-up origin index+1yr -> index+2yr (outcome-window aligned)
#   comorbidities  Comorbidities covariate added to adj.vars
#   preexisting    prior-cancer covariate adjusted for
#   leuk_lymph     leukemia/lymphoma split into its own covariate (excluded from prior cancer)
#   hiv            HIV split into its own covariate (and removed from the comorbidity count)
#   transplant     Transplant covariate in adj.vars (on for both real runs; 0 isolates dropping it)
#
# All CSVs land in tmp/cox_ablation/ in 6_cox's COX1 format. The report is
# tmp/cox_ablation/cox_ablation_report.md.

set -euo pipefail
cd "$(dirname "$0")"
export R_LIBS_USER="$(pwd)/R_libs"

OUTDIR=/home/mtoles/acne/tmp/cox_ablation
mkdir -p "$OUTDIR"

# 1. Real Daniel 6_cox (orig_daniel/1_cleanup.R reads the .bak snapshot).
echo "Running real orig_daniel/6_cox.R ..."
Rscript orig_daniel/6_cox.R >/dev/null
cp eda_output_daniel_original/COX1_any_cancer.csv "$OUTDIR/cox_real_daniel.csv"

# 2. Real ours 6_cox (1_cleanup.R reads the current pooled_records.csv).
echo "Running real 6_cox.R ..."
Rscript 6_cox.R >/dev/null
cp eda_output/COX1_any_cancer.csv "$OUTDIR/cox_real_ours.csv"

# 3. Debug ablation grid (label data followup comorbidities preexisting).
#    data: bak (Daniel's snapshot) | current (ours). toggles: 0=Daniel, 1=ours.
#    Daniel-side configs run on .bak so "daniel" reproduces the real Daniel run; ours-side
#    configs run on current so "ours" reproduces the real ours run. daniel@current isolates
#    the pure data/postprocess effect (Daniel's code on our data).
CONFIGS="$OUTDIR/configs.txt"
cat > "$CONFIGS" <<'EOF'
# label              data     followup comorbidities preexisting leuk_lymph hiv transplant
daniel               bak      0 0 0 0 0 1
ours                 current  1 1 1 1 1 1
ours-leuk_off        current  1 1 1 0 1 1
ours-hiv_off         current  1 1 1 1 0 1
ours-trans_off       current  1 1 1 1 1 0
ours-all_off         current  1 1 1 0 0 0
EOF

echo "Running debug_cox1.R over the config grid ..."
Rscript debug_cox1.R "$CONFIGS" "$OUTDIR"

# 4. Build the markdown report.
echo "Building report ..."
python3 - "$OUTDIR" <<'PY'
import sys
from pathlib import Path
import pandas as pd

outdir = Path(sys.argv[1])

def fully_adj_any_abx(csv):
    """Return (HR, 95% CI, p) for the fully-adjusted Any.Abx 'Yes' term. Within the
    Fully-adjusted block the exposure (Any.Abx) is the first term, so its estimate is the
    first row with a non-null HR."""
    df = pd.read_csv(csv)
    fa = df[df["model"] == "Fully adjusted"]
    hit = fa[fa["**HR**"].notna()].iloc[0]
    return hit["**HR**"], hit["**95% CI**"], hit["**p-value**"]

def hrp(fname):
    csv = outdir / fname
    if not csv.exists():
        return ("—", "—", "—")
    return fully_adj_any_abx(csv)

# Parse the ACTUAL config grid (configs.txt is the source of truth for params).
grid = []
for line in (outdir / "configs.txt").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    label, data, fu, comorb, preexist, leuk, hiv, trans = line.split()[:8]
    grid.append((label, data, fu, comorb, preexist, leuk, hiv, trans))

rows = []
# Real reference runs (Daniel's / our actual 6_cox). Params are fixed by those scripts.
hr, ci, p = hrp("cox_real_daniel.csv")
rows.append(("daniels (real 6_cox)", "bak", "0", "0", "0", "0", "0", "1", hr, ci, p))
hr, ci, p = hrp("cox_real_ours.csv")
rows.append(("ours (real 6_cox)", "current", "1", "1", "1", "1", "1", "1", hr, ci, p))
# Debug configs — report each with the params actually used.
for label, data, fu, comorb, preexist, leuk, hiv, trans in grid:
    hr, ci, p = hrp(f"cox_debug_{label}.csv")
    rows.append((label, data, fu, comorb, preexist, leuk, hiv, trans, hr, ci, p))

# Drop duplicate rows: identical (data, toggles, HR, CI, p) -> same fit. Keep the first
# occurrence (real reference runs come first, so they win over equivalent debug configs).
seen = set()
deduped = []
for r in rows:
    key = r[1:]   # everything except the label
    if key in seen:
        continue
    seen.add(key)
    deduped.append(r)
rows = deduped

lines = []
lines.append("# COX1 ablation — Any.Abx → any cancer (fully adjusted HR)")
lines.append("")
lines.append("Each row is one Cox fit; columns show the **actual** parameters used "
             "(0 = Daniel, 1 = ours). `data`: bak = Daniel's pre-change "
             "`pooled_records.csv.bak`, current = ours.")
lines.append("")
lines.append("- **followup** — follow-up origin index+1yr → index+2yr")
lines.append("- **comorbidities** — Comorbidities covariate in adj.vars")
lines.append("- **preexisting** — a prior-cancer covariate is adjusted for")
lines.append("- **leuk** — leukemia/lymphoma split into its own covariate (excluded from prior cancer)")
lines.append("- **hiv** — HIV split into its own covariate (and removed from the comorbidity count)")
lines.append("- **trans** — Transplant covariate in adj.vars (on for both real runs; 0 isolates dropping it)")
lines.append("")
lines.append("The first two rows are the real `orig_daniel/6_cox.R` and `6_cox.R`; the rest "
             "are `debug_cox1.R` over `configs.txt`.")
lines.append("")
headers = ["config", "data", "followup", "comorb", "preexist", "leuk", "hiv", "trans", "HR", "95% CI", "p"]
aligns  = ["left", "center", "center", "center", "center", "center", "center", "center", "right", "center", "right"]
table   = [headers] + [[str(c) for c in r] for r in rows]
widths  = [max(len(row[i]) for row in table) for i in range(len(headers))]

def fmt_cell(text, width, align):
    if align == "right":
        return text.rjust(width)
    if align == "center":
        return text.center(width)
    return text.ljust(width)

def fmt_row(cells):
    return "| " + " | ".join(fmt_cell(c, w, a) for c, w, a in zip(cells, widths, aligns)) + " |"

def sep_cell(width, align):
    if align == "right":
        return "-" * (width + 1) + ":"
    if align == "center":
        return ":" + "-" * width + ":"
    return ":" + "-" * (width + 1)

lines.append(fmt_row(headers))
lines.append("|" + "|".join(sep_cell(w, a) for w, a in zip(widths, aligns)) + "|")
for r in rows:
    lines.append(fmt_row([str(c) for c in r]))
lines.append("")
report = outdir / "cox_ablation_report.md"
report.write_text("\n".join(lines))
print(f"Wrote {report}")
print("\n".join(lines))
PY

echo
echo "All outputs in $OUTDIR"
ls -1 "$OUTDIR"
