#!/usr/bin/env bash
# run_all.sh
# Purpose: Run decomposition-aware pruning and evaluation over adversarial ratios.
# Notes:
#   - Uses adv_prune_decompose.py (no fine-tuning of base weights).
#   - Set CUDA_VISIBLE_DEVICES outside when you need a specific GPU.

set -euo pipefail

RATIOS="0.0 0.25 0.5 0.75 1.0"
METHODS=("wanda" "lbmask")
SPARSITY="0.5"

run_job() {
  local EXP="$1"        # bert | vit
  local SIZE="$2"       # tiny | small
  local DATASET="$3"    # ag_news | dbpedia_14 | cifar10 | fashion_mnist
  local NUM_CLASSES="$4"

  for TC in $(seq 0 $((NUM_CLASSES-1))); do
    echo "[RUN] exp=$EXP size=$SIZE dataset=$DATASET target_class=$TC"
    local OUTDIR="./results_adv_decompose/${EXP}_${SIZE}_${DATASET}"
    python adv_prune_decompose.py \
      --exp "$EXP" \
      --model_size "$SIZE" \
      --dataset "$DATASET" \
      --methods "${METHODS[@]}" \
      --ratios $RATIOS \
      --sparsity "$SPARSITY" \
      --eps-start-nlp 0.0 --eps-max-nlp 0.25 --eps-step-nlp 0.01 \
      --eps-start-vision 0.0 --eps-max-vision $(python - <<<'print(8/255)') --eps-step-vision $(python - <<<'print(2/255)') \
      --target_class "$TC" \
      --outdir "$OUTDIR"
  done
}

# Examples:
# run_job bert tiny ag_news 4
run_job bert tiny dbpedia_14 14
# run_job bert small ag_news 4
# run_job bert small dbpedia_14 14
# run_job vit tiny cifar10 10
# run_job vit tiny fashion_mnist 10
# run_job vit small cifar10 10
# run_job vit small fashion_mnist 10
