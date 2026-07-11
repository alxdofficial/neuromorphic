#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# POD BOOTSTRAP — runs ON a rented vast.ai pod. Trains ONE arm, pushes results to
# R2, signals done. It does NOT hold the vast.ai key and does NOT self-destruct by
# default: the LOCAL watchdog (scripts/pod/watchdog.py) destroys pods. This keeps
# the billing-capable key off the rented machine.
#
# Invoked by the pod's onstart (see launch.py), which sets these env vars:
#   ARM               required. one variant, e.g. slotgraph_baseline
#   R2_ACCESS_KEY_ID  required. R2 creds (data pull + result push)
#   R2_SECRET_ACCESS_KEY / R2_ENDPOINT / R2_BUCKET   required
#   R2_PREFIX         optional. object namespace (default neuromorphic)
#   REPO_URL          optional. git url (default the public repo)
#   REPO_REF          optional. branch or sha to train (default main)
#   OUT_TAG           optional. run tag (default podrun)
#   STEPS             optional. training steps (default 2000)
#   BATCH             optional. batch size (default 8; fits 24GB)
#   MAX_HOURS         optional. hard wall-clock cap on training (default 8)
#   RUN_ID            optional. label used in R2 result paths (default OUT_TAG)
#
# Result layout in R2 (under $R2_PREFIX/):
#   results/<RUN_ID>/<ARM>/run.jsonl         per-step metrics
#   results/<RUN_ID>/<ARM>/<ARM>.last.pt     final encoder+read-LoRA weights
#   results/<RUN_ID>/<ARM>/summary.json      per-variant summary
#   results/<RUN_ID>/<ARM>/bootstrap.log     full stdout/stderr of this script
#   results/<RUN_ID>/<ARM>/_STATUS           one line: RUNNING|DONE|FAILED (+code)
# The watchdog polls _STATUS and destroys the instance on DONE/FAILED/timeout.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

ARM="${ARM:?set ARM (one variant, e.g. slotgraph_baseline)}"
REPO_URL="${REPO_URL:-https://github.com/alxdofficial/neuromorphic.git}"
REPO_REF="${REPO_REF:-main}"
OUT_TAG="${OUT_TAG:-podrun}"
RUN_ID="${RUN_ID:-$OUT_TAG}"
STEPS="${STEPS:-2000}"
BATCH="${BATCH:-8}"
MAX_HOURS="${MAX_HOURS:-8}"
R2_PREFIX="${R2_PREFIX:-neuromorphic}"
WORK="${WORK_DIR:-/workspace}"
REPO_DIR="$WORK/neuromorphic"
LOG="$WORK/bootstrap.log"

mkdir -p "$WORK"
# Tee everything to a log we upload at the end (so failures are diagnosable off-pod).
exec > >(tee -a "$LOG") 2>&1
echo "[bootstrap] $(date -u +%FT%TZ) arm=$ARM ref=$REPO_REF steps=$STEPS batch=$BATCH max_hours=$MAX_HOURS"

# ── R2 helper (self-contained; no repo checkout needed yet) ──────────────────
export AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:?need R2_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:?need R2_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION=auto
R2_ENDPOINT="${R2_ENDPOINT:?need R2_ENDPOINT}"
R2_BUCKET="${R2_BUCKET:?need R2_BUCKET}"
RES="results/$RUN_ID/$ARM"
r2() { aws s3 --endpoint-url "$R2_ENDPOINT" "$@"; }
key() { echo "s3://$R2_BUCKET/$R2_PREFIX/$1"; }
put_status() {                      # write a one-line status marker to R2
  local msg="$1"; echo "$msg" | r2 cp - "$(key "$RES/_STATUS")" >/dev/null 2>&1 || true
}
upload_results() {                  # push whatever exists (best-effort; called on every exit)
  local d="$REPO_DIR/outputs/memory/${OUT_TAG}_${ARM}"
  r2 cp "$LOG" "$(key "$RES/bootstrap.log")" >/dev/null 2>&1 || true
  [ -f "$d/jsonl/${ARM}.jsonl" ]     && r2 cp "$d/jsonl/${ARM}.jsonl"   "$(key "$RES/run.jsonl")"     >/dev/null 2>&1 || true
  [ -f "$d/ckpts/${ARM}.last.pt" ]   && r2 cp "$d/ckpts/${ARM}.last.pt" "$(key "$RES/${ARM}.last.pt")" >/dev/null 2>&1 || true
  [ -f "$d/ckpts/${ARM}.best.pt" ]   && r2 cp "$d/ckpts/${ARM}.best.pt" "$(key "$RES/${ARM}.best.pt")" >/dev/null 2>&1 || true
  local summ="$REPO_DIR/outputs/memory/${OUT_TAG}_summary.json"
  [ -f "$summ" ]                     && r2 cp "$summ"                    "$(key "$RES/summary.json")"  >/dev/null 2>&1 || true
}

# Ensure a terminal status + log upload no matter how we exit (crash, OOM, timeout).
FINAL=FAILED
on_exit() {
  local code=$?
  upload_results
  put_status "$FINAL code=$code $(date -u +%FT%TZ)"
  echo "[bootstrap] exit code=$code final=$FINAL"
}
trap on_exit EXIT

# Install a lightweight aws CLI early (pip; the pytorch image has python) so status
# markers work even if later steps fail.
python3 -m pip install -q --no-input awscli 2>&1 | tail -1 || pip install -q awscli
put_status "RUNNING started=$(date -u +%FT%TZ)"

# ── code ─────────────────────────────────────────────────────────────────────
echo "[bootstrap] cloning $REPO_URL @ $REPO_REF"
rm -rf "$REPO_DIR"
git clone --filter=blob:none "$REPO_URL" "$REPO_DIR"
git -C "$REPO_DIR" checkout "$REPO_REF"
echo "[bootstrap] HEAD=$(git -C "$REPO_DIR" rev-parse --short HEAD)"

# ── python deps ──────────────────────────────────────────────────────────────
# vast.ai pytorch images ship torch+CUDA. Install only what the repo adds on top.
echo "[bootstrap] installing python deps"
if [ -f "$REPO_DIR/requirements.txt" ]; then
  python3 -m pip install -q --no-input -r "$REPO_DIR/requirements.txt" 2>&1 | tail -3
else
  python3 -m pip install -q --no-input transformers datasets safetensors pyarrow accelerate 2>&1 | tail -3
fi

# ── data ─────────────────────────────────────────────────────────────────────
echo "[bootstrap] pulling data.tar.gz from R2"
r2 cp "$(key data.tar.gz)" "$WORK/data.tar.gz"
mkdir -p "$REPO_DIR/data"
tar xzf "$WORK/data.tar.gz" -C "$REPO_DIR"     # tar stores paths as data/<source>/...
echo "[bootstrap] data sources: $(ls "$REPO_DIR/data" | tr '\n' ' ')"

# ── train ────────────────────────────────────────────────────────────────────
# titans keeps its per-window autograd memory live across windows → disable the
# streaming grad-checkpoint for that arm only (matches the local config).
EXTRA=()
[ "$ARM" = "titans_baseline" ] && EXTRA+=(--no-grad-ckpt-stream)

echo "[bootstrap] training $ARM for $STEPS steps (behavioral_kl, B=$BATCH)"
cd "$REPO_DIR"
set +e
timeout "${MAX_HOURS}h" python3 scripts/train/train.py \
  --task mixed --variants "$ARM" \
  --objective-mode behavioral_kl \
  --backbone HuggingFaceTB/SmolLM2-135M \
  --steps "$STEPS" --batch-size "$BATCH" \
  --out-tag "$OUT_TAG" \
  "${EXTRA[@]}"
TRAIN_CODE=$?
set -e
echo "[bootstrap] training exit=$TRAIN_CODE"

if [ "$TRAIN_CODE" -eq 0 ]; then
  FINAL=DONE
elif [ "$TRAIN_CODE" -eq 124 ]; then
  FINAL=TIMEOUT      # timeout(1) exit code; still upload partial results
else
  FINAL=FAILED
fi
# on_exit (trap) uploads results + writes the terminal _STATUS.
