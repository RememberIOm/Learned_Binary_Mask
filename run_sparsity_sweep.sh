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

  # The python script handles the inner loops over methods and sparsities.
  # We loop over each target class for decomposition evaluation.
  for TC in $(seq 0 $((N_CLASSES-1))); do
    echo "--- Running for Target Class: $TC ---"
    python pruning_sweep.py \
      --experiments "$EXP" \
      --methods $METHODS \
      --sparsities $SPARSITIES \
      --outdir "./results_sparsity_sweep/${EXP//:/_}" \
      --decompose \
      --target_class "$TC"
  done
done

echo "✅ All sweeps finished."