#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DRIVE — the "local tmux → ssh into pod" flow, scripted. For each pod in a run's
# state file: SSH in over the encrypted channel, copy R2 creds (NEVER via vast
# metadata), and start bootstrap.sh inside a REMOTE tmux session named after the arm.
#
# Preconditions:
#   * pods launched with:  python scripts/pod/launch.py --mode ssh --go
#   * your SSH public key is registered with vast.ai. Register once with:
#       scripts/pod/drive.sh --register-key            # uploads ~/.ssh/id_*.pub
#   * ~/.config/r2/credentials and ~/.config/vastai/api_key present locally.
#
# Usage:
#   scripts/pod/drive.sh <run_id>                 # start bootstrap on every pod
#   scripts/pod/drive.sh <run_id> --attach <arm>  # ssh in + attach the arm's tmux
#   scripts/pod/drive.sh --register-key           # register local ssh pubkey w/ vast
#
# After driving, watch remotely with:
#   scripts/pod/drive.sh <run_id> --attach slotgraph_baseline   (Ctrl-b d to detach)
# and keep the billing backstop running:
#   python scripts/pod/watchdog.py <run_id>
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
STATE_DIR="$HERE/state"
VAST="$REPO/.venv/bin/vastai"; [ -x "$VAST" ] || VAST=vastai
export VAST_API_KEY="$(cat "$HOME/.config/vastai/api_key")"
R2_CRED="$HOME/.config/r2/credentials"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)

register_key() {
  local pub
  pub="$(ls "$HOME"/.ssh/id_ed25519.pub "$HOME"/.ssh/id_rsa.pub 2>/dev/null | head -1)"
  [ -n "$pub" ] || { echo "no ~/.ssh/id_ed25519.pub or id_rsa.pub — run: ssh-keygen -t ed25519" >&2; exit 1; }
  echo "[drive] registering $pub with vast.ai"
  "$VAST" create ssh-key "$(cat "$pub")" 2>&1 | tail -2
}

# parse ssh://root@HOST:PORT from `vastai ssh-url`
ssh_host_port() {
  local url; url="$("$VAST" ssh-url "$1" 2>/dev/null | tr -d '[:space:]')"
  echo "$url" | sed -E 's#ssh://[^@]+@([^:]+):([0-9]+)#\1 \2#'
}

wait_ssh() {   # $1=host $2=port ; wait until sshd answers (pod may still be booting)
  local host="$1" port="$2" i
  for i in $(seq 1 60); do
    if ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" true 2>/dev/null; then return 0; fi
    sleep 10
  done
  return 1
}

start_pod() {   # $1=instance_id $2=arm  + run cfg via env RUN_ID/OUT_TAG/REPO_REF/STEPS/BATCH/MAX_HOURS
  local iid="$1" arm="$2"
  read -r host port < <(ssh_host_port "$iid")
  if [ -z "${host:-}" ]; then echo "[drive] $arm: no ssh-url yet for $iid (still booting?)"; return 1; fi
  echo "[drive] $arm ($iid) → root@$host:$port ; waiting for sshd…"
  wait_ssh "$host" "$port" || { echo "[drive] $arm: ssh never came up"; return 1; }
  # copy R2 creds over the encrypted channel (not vast metadata)
  ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" "mkdir -p /root/.config/r2"
  scp "${SSH_OPTS[@]}" -P "$port" "$R2_CRED" "root@$host:/root/.config/r2/credentials"
  # start bootstrap in a detached remote tmux; source creds → env that bootstrap reads
  ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" bash -s <<EOF
set -a; source /root/.config/r2/credentials; set +a
export ARM=$arm OUT_TAG=$RUN_ID RUN_ID=$RUN_ID REPO_REF=${REPO_REF:-main} \
       STEPS=${STEPS:-2000} BATCH=${BATCH:-8} MAX_HOURS=${MAX_HOURS:-8}
# repo was pre-cloned by onstart; fall back to cloning if not present
[ -d /workspace/neuromorphic ] || git clone --filter=blob:none \
    https://github.com/alxdofficial/neuromorphic.git /workspace/neuromorphic
cd /workspace/neuromorphic
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)
tmux kill-session -t $arm 2>/dev/null || true
tmux new-session -d -s $arm "bash scripts/pod/bootstrap.sh 2>&1 | tee /workspace/boot.log"
echo "[pod] bootstrap started in tmux session '$arm'"
EOF
  echo "[drive] $arm launched. attach: scripts/pod/drive.sh $RUN_ID --attach $arm"
}

attach_pod() {  # $1=run_id $2=arm : ssh in and attach the remote tmux
  local run_id="$1" arm="$2" iid host port
  iid="$(python3 -c "import json,sys; s=json.load(open('$STATE_DIR/$run_id.json'));
print(next((i['instance_id'] for i in s['instances'] if i['arm']=='$arm'),''))")"
  [ -n "$iid" ] || { echo "no instance for arm '$arm' in $run_id" >&2; exit 1; }
  read -r host port < <(ssh_host_port "$iid")
  echo "[drive] attaching $arm on root@$host:$port (Ctrl-b d to detach)"
  exec ssh "${SSH_OPTS[@]}" -t -p "$port" "root@$host" "tmux attach -t $arm"
}

# ── arg dispatch ─────────────────────────────────────────────────────────────
if [ "${1:-}" = "--register-key" ]; then register_key; exit 0; fi
RUN_ID="${1:?usage: drive.sh <run_id> [--attach <arm>] | --register-key}"
STATE="$STATE_DIR/$RUN_ID.json"
[ -f "$STATE" ] || { echo "no state at $STATE" >&2; exit 1; }

if [ "${2:-}" = "--attach" ]; then attach_pod "$RUN_ID" "${3:?arm}"; fi

# load run cfg into env for start_pod
eval "$(python3 - "$STATE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); c=s.get("cfg",{})
print(f'REPO_REF={c.get("ref","main")}; STEPS={c.get("steps",2000)}; '
      f'BATCH={c.get("batch",8)}; MAX_HOURS={c.get("max_hours",8)}')
PY
)"
export REPO_REF STEPS BATCH MAX_HOURS

# drive every pod
mapfile -t ROWS < <(python3 -c "import json;
s=json.load(open('$STATE'));
[print(i['instance_id'], i['arm']) for i in s['instances'] if i.get('instance_id')]")
[ ${#ROWS[@]} -gt 0 ] || { echo "no instances in $STATE" >&2; exit 1; }
for row in "${ROWS[@]}"; do
  start_pod $row || echo "[drive] (continuing despite error on: $row)"
done
echo "[drive] all pods driven. billing backstop: python scripts/pod/watchdog.py $RUN_ID"
