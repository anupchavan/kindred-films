#!/bin/zsh
cd "$(dirname "$0")/.."
for wd in 1e-5 1e-4 1e-3; do
  echo "######## weight_decay=$wd"
  python experiments/36_rmse_hybrid.py $wd 14 mf,hybrid 2>&1 | grep -E "FINAL|^(mf|hybrid)\s*:"
done
echo SWEEP-DONE
