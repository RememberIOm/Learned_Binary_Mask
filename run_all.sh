# run_all.sh
set -euo pipefail

SPARSITIES="0.1 0.2 0.3 0.4 0.5 0.6"
METHODS="wanda lbmask"   # 필요 시 하나만 남겨도 됩니다.

run_one_gpu() {
  local GPU="$1"
  local EXP="$2"
  local NCLASS="$3"  # 4 for ag_news, 10 for yahoo_answers_topics

  for TC in $(seq 0 $((NCLASS-1))); do
    # outdir을 타겟별로 분리해 동시실행 충돌을 방지
    OUTDIR="./results_sweep/${EXP//:/_}/tc${TC}"
    python pruning_sweep.py \
      --experiments "$EXP" \
      --methods $METHODS \
      --sparsities $SPARSITIES \
      --gpu "$GPU" \
      --decompose --target_class "$TC" \
      --outdir "$OUTDIR" \
      --lbmask_progressive \
      --lbmask_schedules "constant"
  done
}

# GPU별로 한 실험씩 배정 (병렬)
(run_one_gpu 0 "bert:tiny:ag_news"              4  ) &
(run_one_gpu 1 "bert:small:ag_news"             4  ) &
(run_one_gpu 2 "bert:tiny:yahoo_answers_topics" 10 ) &
(run_one_gpu 3 "bert:small:yahoo_answers_topics" 10 ) &
wait
