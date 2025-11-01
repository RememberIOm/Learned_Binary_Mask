#!/usr/bin/env bash
# run_sparsity_sweep.sh
# Purpose: Run sparsity sweeps for wanda and mi, prune with original data only,
#          then decompose and evaluate performance on the original test set.

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

# --- Experiment Configuration ---
# Define the experiments to run. Format: "model_type:model_size:dataset_name"
EXPERIMENTS=("bert:tiny:dbpedia_14") # You can add more, e.g., ("bert:small:ag_news" "vit:small:cifar10")
# Define the number of classes for each dataset to loop through targets.
# This must correspond to the experiments above.
NUM_CLASSES=(14) # e.g., (4 10) for ag_news and cifar10
# Define methods to compare.
METHODS="wanda mi"
# Define sparsity ratios to sweep.
SPARSITIES="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# --- Run Sweep ---
for i in "${!EXPERIMENTS[@]}"; do
  EXP="${EXPERIMENTS[$i]}"
  N_CLASSES="${NUM_CLASSES[$i]}"
  
  echo "================================================================="
  echo "Starting sweep for experiment: $EXP"
  echo "================================================================="

  # Example replacement: direct CLI loops via cli.run (wanda/mi, original-only prune, then decompose+eval)
  IFS=':' read -r MTYPE MSIZE DSET <<< "$EXP"
  for TC in $(seq 0 $((N_CLASSES-1))); do
    for SP in $SPARSITIES; do
      for M in $METHODS; do
        echo "--- $EXP  target=$TC  method=$M  sparsity=$SP ---"
        # prune-then-decompose on original data only (ratio=0.0), evaluate original set
        python -m cli.run prune-then-decompose \
          --exp "$MTYPE" --model_size "$MSIZE" --dataset "$DSET" \
          --methods "$M" --ratios 0.0 \
          --sparsity "$SP" --targets "$TC" \
          --eval original \
          --outdir "./results_sparsity_sweep/${EXP//:/_}"
      done
    done
  done
done

echo "✅ All sweeps finished."