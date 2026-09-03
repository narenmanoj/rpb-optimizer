#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-$HOME/project_pi_das227/kp875/LLM_optimizer}"
REPO="${REPO:-$BASE/rpb-optimizer}"
OUT="${OUT:-$BASE/mlopt_paper_assets}"
module load miniconda
conda activate torch_env
python3 "$REPO/generate_mlopt_paper_assets.py" --base "$BASE" --output "$OUT"
python3 "$REPO/verify_mlopt_paper_assets.py" "$OUT"
(cd "$BASE" && tar -czf mlopt_paper_assets.tar.gz "$(basename "$OUT")")
echo "Created $BASE/mlopt_paper_assets.tar.gz"
