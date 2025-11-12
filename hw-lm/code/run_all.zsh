# #!/usr/bin/env zsh
# set -euo pipefail

# # tune this to your machine (CPU/GPU). 1–2 if on a single GPU.
# typeset -i MAX_PROCS=4

# LR=1e-5
# LEX=../lexicons/words-gs-10.txt
# V=vocab-genspam.txt
# SMOOTHER=log_linear

# # Where to save models & logs
# OUTDIR=parallel_run_models
# mkdir -p "$OUTDIR"

# # Prevent CPU thread oversubscription
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export PYTORCH_NUM_THREADS=1

# run_job() {
#   local split=$1   # gen | spam
#   local C=$2

#   # filenames under OUTDIR
#   local out="${OUTDIR}/${split}~l2=${C}.model"
#   local log="${OUTDIR}/${split}~l2=${C}.log"

#   ./train_lm.py "$V" "$SMOOTHER" "../data/gen_spam/train/${split}" \
#     --lexicon "$LEX" --lr "$LR" --l2_regularization "$C" \
#     --output "$out" > "$log" 2>&1 &
# }

# # launch jobs with a simple slot limiter
# for C in 1.0 0.0 0.1 0.5 5; do
#   for split in gen spam; do
#     run_job "$split" "$C"
#     # throttle: wait until number of background jobs < MAX_PROCS
#     while [ "$(jobs -p | wc -l | tr -d ' ')" -ge "$MAX_PROCS" ]; do
#       sleep 1
#     done
#   done
# done

# # wait for all to finish
# wait
# echo "All runs done ✅  (saved in: $OUTDIR)"
