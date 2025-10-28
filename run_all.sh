#!/usr/bin/env bash
# run_all.sh
# Purpose: Run decomposition-aware pruning and evaluation over adversarial ratios.
# Notes:
#   - Uses adv_prune_decompose.py which correctly handles the prune-then-decompose workflow.
#   - Set CUDA_VISIBLE_DEVICES outside when you need a specific GPU.

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

# Define shared parameters for the experiments.
RATIOS="0.0 0.25 0.5 0.75 1.0"
# METHODS=("wanda" "lbmask" "mi")
METHODS=("wanda" "mi")
SPARSITY="0.5"

run_job() {
  local EXP="$1"        # bert | vit
  local SIZE="$2"       # tiny | small
  local DATASET="$3"    # ag_news | dbpedia_14 | cifar10 | fashion_mnist
  local NUM_CLASSES="$4"

  # Iterate through each target class to run the full prune-then-decompose pipeline.
  # The python script handles the inner loops over methods and ratios.
  for TC in $(seq 0 $((NUM_CLASSES-1))); do
    echo "Running job for ${EXP}/${SIZE}/${DATASET} with target class ${TC}"
    python adv_prune_decompose.py \
      --exp "$EXP" \
      --model_size "$SIZE" \
      --dataset "$DATASET" \
      --methods "${METHODS[@]}" \
      --ratios $RATIOS \
      --sparsity "$SPARSITY" \
      --target_class "$TC" \
      # --outdir "./results_adv_decompose/${EXP}_${SIZE}_${DATASET}"
      --outdir "./results_adv_decompose_no_robust/${EXP}_${SIZE}_${DATASET}"
  done
}

# --- Experiment Definitions ---
# The following lines define the experiments to be run.
# Each call to 'run_job' will execute the full evaluation
# for a specific model, dataset, and for all its classes.

# Example: BERT on DBPedia-14
# run_job bert small dbpedia_14 14

# Example: ViT on CIFAR-10
# run_job vit small cifar10 10

# Example: BERT on AG News
run_job bert tiny ag_news 4
# run_job bert small ag_news 4

# Example: ViT on Fashion MNIST
# run_job vit small fashion_mnist 10