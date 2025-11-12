#!/usr/bin/env zsh
set -euo pipefail
setopt NO_BEEP
setopt monitor        # <-- ensure jobs -p works in this non-interactive shell

# ---- tune ----
typeset -i MAX_PROCS=4
LR=1e-5
LEX=../lexicons/words-gs-10.txt
V=vocab-genspam.txt
SMOOTHER=log_linear
EPOCHS=10
OUTDIR=parallel_run_models
typeset -a SPLITS=(gen spam)
typeset -a CLS=(1.0 0.0 0.1 0.5 5)
# --------------

mkdir -p "$OUTDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 PYTHONUNBUFFERED=1

# Expect to run from code/
PY=$(command -v python || true)
TRAIN=./train_lm.py
DATA_BASE=../data/gen_spam/train

TRAIN_ABS=${TRAIN:A}
V_ABS=${V:A}
LEX_ABS=${LEX:A}
DATA_BASE_ABS=${DATA_BASE:A}
OUTDIR_ABS=${OUTDIR:A}

# --- sanity ---
[[ -n "${PY}" ]] || { echo "ERROR: python not found"; exit 1; }
[[ -f "$TRAIN_ABS" ]] || { echo "ERROR: $TRAIN_ABS missing"; exit 1; }
[[ -f "$V_ABS" ]] || { echo "ERROR: $V_ABS missing"; exit 1; }
[[ -f "$LEX_ABS" ]] || { echo "ERROR: $LEX_ABS missing"; exit 1; }
[[ -d "$DATA_BASE_ABS" ]] || { echo "ERROR: $DATA_BASE_ABS missing"; exit 1; }

# Build log prefix mirroring get_model_filename()
build_log_prefix() {
  local smoother=$1 corpus=$2 vocab_base=$3 lex_base=$4 l2=$5 epochs=$6
  local prefix="corpus=${corpus}~vocab=${vocab_base}~smoother=${smoother}"
  case "$smoother" in
    uniform) echo "${prefix}" ;;
    add_lambda|backoff) echo "${prefix}~lambda=${l2}" ;;
    log_linear|improved|loglinear|LOG_LINEAR|IMPROVED)
      echo "${prefix}~lexicon=${lex_base}~l2=${l2}~epochs=${epochs}" ;;
    *) echo "${prefix}" ;;
  esac
}

launch_one() {
  local split=$1 C=$2
  local train_split="${DATA_BASE_ABS}/${split}"
  [[ -d "$train_split" || -f "$train_split" ]] || { echo "⚠️  WARN: missing $train_split — skipping"; return 0; }

  local vocab_base=${V_ABS:t}
  local lex_base=${LEX_ABS:t}
  local log_prefix=$(build_log_prefix "$SMOOTHER" "$split" "$vocab_base" "$lex_base" "$C" "$EPOCHS")
  local log="${OUTDIR_ABS}/${log_prefix}.log"

  echo "▶️  start  ${log_prefix}"
  (
    cd "$OUTDIR_ABS"
    "$PY" "$TRAIN_ABS" "$V_ABS" "$SMOOTHER" "$train_split" \
      --lexicon "$LEX_ABS" --lr "$LR" --l2_regularization "$C" --epochs "$EPOCHS"
  ) > "$log" 2>&1 &

  # tiny delay to let the job register so jobs -p sees it
  sleep 0.05
  echo "    jobs running: $(jobs -p | wc -l | tr -d ' ') / $MAX_PROCS"
}

# ---- launch with simple throttle ----
for C in "${CLS[@]}"; do
  for split in "${SPLITS[@]}"; do
    # throttle until fewer than MAX_PROCS running
    while [ "$(jobs -p | wc -l | tr -d ' ')" -ge "$MAX_PROCS" ]; do
      sleep 0.2
    done
    launch_one "$split" "$C"
  done
done

# wait for all background jobs
wait
echo "All runs done ✅ (saved in: $OUTDIR_ABS)"
