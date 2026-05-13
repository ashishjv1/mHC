#!/bin/bash
# Run all three comparison experiments + optional ablation.
# Uses identical seed and data ordering across runs.
#
# Prerequisites:
#   - WANDB_API_KEY set in environment
#   - Data prepared (auto-downloads on first run)
#   - GPU available
#
# Usage:
#   bash scripts/run_all.sh              # 3 main runs
#   bash scripts/run_all.sh --ablation   # + unconstrained HC ablation

set -euo pipefail

SEED=42
DATA_DIR="data"
COMMON_ARGS="--seed $SEED --data_dir $DATA_DIR --compile"

echo "=== Run 1/3: Baseline AdamW ==="
python -m src.train --preset baseline_adamw $COMMON_ARGS

echo "=== Run 2/3: Baseline Muon ==="
python -m src.train --preset baseline_muon $COMMON_ARGS

echo "=== Run 3/3: mHC + Muon ==="
python -m src.train --preset mhc_muon $COMMON_ARGS

if [[ "${1:-}" == "--ablation" ]]; then
    echo "=== Run 4/4: HC Unconstrained + Muon (ablation) ==="
    python -m src.train --preset hc_muon_unconstrained $COMMON_ARGS
fi

echo "=== Generating plots ==="
python scripts/plot_results.py

echo "=== All runs complete ==="
