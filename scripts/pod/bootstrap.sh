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
#   BATCH             optional. batch size (default 6; fits a 24GB 4090 for ALL arms incl. the KV
#                     ones. B=8 peaks ~22GB + fragmentation → OOM on a pod's ~23.5GB usable; B=6 = ~18GB.)
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
BATCH="${BATCH:-6}"
MAX_HOURS="${MAX_HOURS:-8}"
R2_PREFIX="${R2_PREFIX:-neuromorphic}"
WORK="${WORK_DIR:-/workspace}"
REPO_DIR="$WORK/neuromorphic"
LOG="$WORK/bootstrap.log"

mkdir -p "$WORK"
# Raise the open-file soft limit to the hard cap up front: multi-worker DataLoaders + pin_memory hold
# hundreds of shared-memory FDs, and the container default (soft 1024) triggers "Too many open files"
# / "received 0 items of ancdata". The trainer also raises it in-process, but doing it here covers every
# child. No privilege needed to raise soft→hard; ignore if the shell disallows it.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
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
put_status() {                      # write a one-line status marker to R2 (RETRIED)
  # The watchdog reaps on the TERMINAL _STATUS (DONE/FAILED/TIMEOUT). A single best-effort write that
  # dropped on a transient R2 hiccup would leave a finished pod billing to the 8.5h wall-cap (audit).
  # Retry a few times with backoff so a finished pod is reliably reaped promptly.
  local msg="$1" i
  for i in 1 2 3 4; do
    echo "$msg" | r2 cp - "$(key "$RES/_STATUS")" >/dev/null 2>&1 && return 0
    sleep 3
  done
  return 0                          # best-effort; never fatal to the run
}
upload_results() {                  # push whatever exists (best-effort; called periodically + on exit)
  local d="$REPO_DIR/outputs/memory/${OUT_TAG}_${ARM}"
  r2 cp "$LOG" "$(key "$RES/bootstrap.log")" >/dev/null 2>&1 || true
  # MIRROR the local layout on R2 (jsonl/ + ckpts/). `sync` (no --delete) uploads only new/changed objects:
  #   jsonl/<arm>.jsonl (metrics), ckpts/<arm>.{last,best,step<N>}.pt (last/best + retained milestones).
  # Ckpts are 55-175MB each and some hosts have a GLACIAL uplink → BOUND each sync with a timeout so the
  # final on_exit upload can't stall pod teardown/billing (observed 15+ min idle). The periodic sync already
  # ships the milestones, so a timed-out final sync loses at most the last <120s of .last.pt. Failures are
  # LOGGED (not silently swallowed) so a lagging uplink is visible in bootstrap.log.
  [ -d "$d/jsonl" ] && { timeout 120 r2 sync "$d/jsonl" "$(key "$RES/jsonl")" >/dev/null 2>&1 \
                         || echo "[upload] WARN jsonl sync timed out/failed"; }
  [ -d "$d/ckpts" ] && { timeout 300 r2 sync "$d/ckpts" "$(key "$RES/ckpts")" >/dev/null 2>&1 \
                         || echo "[upload] WARN ckpts sync timed out/failed (partial; periodic sync has milestones)"; }
  local summ="$REPO_DIR/outputs/memory/${OUT_TAG}_summary.json"
  [ -f "$summ" ]    && r2 cp "$summ" "$(key "$RES/summary.json")" >/dev/null 2>&1 || true
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

# FAIL LOUD on any setup error (clone/checkout/deps/R2-pull/untar). Without this (was only
# `set -uo pipefail`), a failed step fell through to TRAINING → a pod could silently train the wrong
# ref or a degraded data mix and report it as a result. The training call re-enables `set +e` (below)
# so its exit code is captured, not fatal.
set -e

# ── python env ───────────────────────────────────────────────────────────────
# Vast's own images (vastai/pytorch) ship torch in a venv at /venv/main. ACTIVATE it, or `python3`
# is the system interpreter with NO torch → pip would reinstall torch (gigabytes, minutes of billed
# idle). Also prefer `uv` (present on the vast image) → installs in seconds (~55s for the whole set).
if [ -f /venv/main/bin/activate ]; then
  source /venv/main/bin/activate
  echo "[bootstrap] activated venv /venv/main (python=$(command -v python))"
fi
pipi() { if command -v uv >/dev/null 2>&1; then uv pip install -q "$@"; else python3 -m pip install -q --no-input "$@"; fi; }

# Install a lightweight aws CLI early so status markers work even if later steps fail.
pipi awscli 2>&1 | tail -1 || python3 -m pip install -q --no-input awscli
# Tune S3 upload concurrency so checkpoint syncs saturate the uplink (default 10 → 32 parallel parts);
# helps the periodic + final ckpt sync finish faster on bandwidth-available hosts (audit: slow uploads
# blocked teardown). No effect on a truly starved uplink, but free upside where bandwidth exists.
aws configure set default.s3.max_concurrent_requests 32 2>/dev/null || true
aws configure set default.s3.max_queue_size 2000 2>/dev/null || true
put_status "RUNNING started=$(date -u +%FT%TZ)"

# ── code ─────────────────────────────────────────────────────────────────────
# cd to $WORK FIRST: drive.sh starts us with cwd=$REPO_DIR (it cd's there to find this script), and
# the next line rm -rf's $REPO_DIR — deleting our own cwd → `git clone` then dies with "Unable to read
# current working directory" (a RACE: only pods whose onstart pre-clone finished before drive.sh cd'd
# in hit it). Worse, the trap's FAILED status write also runs from the deleted cwd and fails, leaving a
# stale RUNNING in R2 so the watchdog never reaps. cd'ing out of $REPO_DIR before deleting it fixes both.
cd "$WORK"
echo "[bootstrap] cloning $REPO_URL @ $REPO_REF"
rm -rf "$REPO_DIR"
git clone --filter=blob:none "$REPO_URL" "$REPO_DIR"
git -C "$REPO_DIR" checkout "$REPO_REF"
echo "[bootstrap] HEAD=$(git -C "$REPO_DIR" rev-parse --short HEAD)"

# ── python deps ──────────────────────────────────────────────────────────────
# vast.ai pytorch images ship torch+CUDA. Install only what the repo adds on top.
echo "[bootstrap] installing python deps"
if [ -f "$REPO_DIR/requirements.txt" ]; then
  pipi -r "$REPO_DIR/requirements.txt" 2>&1 | tail -3
else
  pipi transformers datasets safetensors pyarrow accelerate 2>&1 | tail -3
fi

# ── data ─────────────────────────────────────────────────────────────────────
echo "[bootstrap] pulling data.tar.gz from R2"
r2 cp "$(key data.tar.gz)" "$WORK/data.tar.gz"
mkdir -p "$REPO_DIR/data"
# --no-same-owner: pods run as root but the tarball is packed as the local uid → without this, tar tries
# (and fails) to chown every file, spamming "Cannot change ownership" warnings + wasted syscalls (audit).
tar xzf "$WORK/data.tar.gz" -C "$REPO_DIR" --no-same-owner     # tar stores paths as data/<source>/...
echo "[bootstrap] data sources: $(ls "$REPO_DIR/data" | tr '\n' ' ')"
# Assert the untarred data matches the packer's MANIFEST by FILE COUNT + jsonl ROW COUNT per source (NOT
# byte totals) — a present-but-TRUNCATED source passes a bare `[ -d ]` check but would silently train a
# degraded/reweighted mix. Bytes are NOT compared: `du -sb` counts directory apparent-size, which differs
# between the ext4 pack host and the pod's overlayfs → a byte check false-FATALs the pod BEFORE any GPU
# work (audit). File+row counts are filesystem-invariant. babi/bio are NOT tarred (HF-runtime/procedural).
_manifest="$REPO_DIR/data/MANIFEST.txt"
[ -f "$_manifest" ] || { echo "[bootstrap] FATAL no data/MANIFEST.txt in tarball (repack with pack_data.sh)"; exit 1; }
_bad=""
while IFS="$(printf '\t')" read -r s fc rc by; do
  [ -n "$s" ] || continue
  d="$REPO_DIR/data/$s"
  if [ ! -d "$d" ]; then _bad="$_bad $s(missing)"; continue; fi
  afc=$(find "$d" -type f | wc -l)
  arc=$(find "$d" -type f -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
  if [ "$afc" != "$fc" ] || [ "$arc" != "$rc" ]; then
    _bad="$_bad $s(files:$afc/$fc rows:$arc/$rc)"
  fi
done < "$_manifest"
[ -z "$_bad" ] || { echo "[bootstrap] FATAL data manifest mismatch:$_bad"; exit 1; }
echo "[bootstrap] data manifest OK ($(grep -c . "$_manifest") sources verified)"

# ── pre-seed HF cache ─────────────────────────────────────────────────────────
# Pull the ~207MB models.tar.gz (SmolLM2-135M backbone + meta-llama/Llama-3.2-1B TOKENIZER, no Llama
# weights) and unpack into the HF hub cache, so the first training step doesn't (a) download SmolLM2 per
# pod (GPU-idle) or (b) depend on a live GATED-HF fetch of the Llama tokenizer (a network SPOF that can
# stall/fail the run). Best-effort: if absent/fails, training falls back to downloading from HF.
_HFHUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
mkdir -p "$_HFHUB"
if r2 cp "$(key models.tar.gz)" "$WORK/models.tar.gz" >/dev/null 2>&1; then
  tar xzf "$WORK/models.tar.gz" -C "$_HFHUB" --no-same-owner 2>/dev/null \
    && echo "[bootstrap] pre-seeded HF cache → $_HFHUB (SmolLM2 + Llama tokenizer)" \
    || echo "[bootstrap] WARN models.tar.gz untar failed — will download from HF at first step"
else
  echo "[bootstrap] no models.tar.gz on R2 — will download from HF at first step"
fi

# ── train ────────────────────────────────────────────────────────────────────
# titans keeps its per-window autograd memory live across windows → disable the
# streaming grad-checkpoint for that arm only (matches the local config).
EXTRA=()
[ "$ARM" = "titans_baseline" ] && EXTRA+=(--no-grad-ckpt-stream)

# Gated HF dep: the FineWeb src-tokenizer is meta-llama/Llama-3.2-1B (GATED — needs a Meta-approved
# HF token) for the reconstruct/continuation tasks. The vast image sets HF_HOME=/workspace/.hf_home,
# so a token file under ~/.cache is ignored → export HF_TOKEN (path-independent, highest priority).
# drive.sh ships the token to /root/.config/hf_token over the SSH channel (never vast metadata).
[ -f /root/.config/hf_token ] && export HF_TOKEN="$(cat /root/.config/hf_token)"
[ -n "${HF_TOKEN:-}" ] && echo "[bootstrap] HF_TOKEN present (gated FineWeb tokenizer enabled)" \
                       || echo "[bootstrap] WARNING no HF_TOKEN — meta-llama/Llama-3.2-1B will 401 (reconstruct/continuation fail)"

echo "[bootstrap] training $ARM for $STEPS steps (behavioral_kl, B=$BATCH)"
cd "$REPO_DIR"
# expandable_segments avoids the fragmentation OOM seen at the VRAM edge (a B=6 KV arm peaks ~18GB but
# fragments the last ~300MB over a ~23.5GB-usable 4090 → OOM without this). Free; recommended by torch.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Crash-safe artifacts: sync checkpoints + metrics to R2 every 120s DURING training, so a mid-run OOM/
# timeout still leaves every completed milestone (+ the jsonl) on R2, not just whatever the final
# on_exit upload catches. `|| true` keeps a transient R2 hiccup from killing the loop.
( while sleep 120; do upload_results || true; done ) &
SYNC_PID=$!
set +e
# In-training val is a SANITY GATE + best.pt selector here — the real cross-arm comparison is the separate
# local band-gate eval on the pulled ckpts. So run val LEANER on pods (default every 1000 steps × 16 batches
# vs the 500×32 default) → ~4× less validation overhead (~22min → ~5min/arm; audit). Override via env.
timeout "${MAX_HOURS}h" python3 scripts/train/train.py \
  --task mixed --variants "$ARM" \
  --objective-mode behavioral_kl \
  --backbone HuggingFaceTB/SmolLM2-135M \
  --steps "$STEPS" --batch-size "$BATCH" \
  --val-every "${VAL_EVERY:-1000}" --val-batches "${VAL_BATCHES:-16}" \
  --out-tag "$OUT_TAG" \
  "${EXTRA[@]}"
TRAIN_CODE=$?
set -e
kill "$SYNC_PID" 2>/dev/null || true      # stop the periodic syncer; on_exit does the final upload
echo "[bootstrap] training exit=$TRAIN_CODE"

if [ "$TRAIN_CODE" -eq 0 ]; then
  FINAL=DONE
elif [ "$TRAIN_CODE" -eq 124 ]; then
  FINAL=TIMEOUT      # timeout(1) exit code; still upload partial results
else
  FINAL=FAILED
fi
# on_exit (trap) uploads results + writes the terminal _STATUS.
