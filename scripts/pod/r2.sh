#!/usr/bin/env bash
# Thin R2 wrapper: sources ~/.config/r2/credentials and runs aws s3 against the R2 endpoint.
# Namespaces every object under $R2_PREFIX/ inside $R2_BUCKET so it doesn't collide with other files.
#
# Usage:
#   scripts/pod/r2.sh up   <local_path>  <remote_key>     # upload   (remote_key relative to the prefix)
#   scripts/pod/r2.sh down <remote_key>   <local_path>    # download
#   scripts/pod/r2.sh ls   [remote_prefix]                # list
#   scripts/pod/r2.sh rm   <remote_key>                   # delete
#   scripts/pod/r2.sh raw  <aws s3 args...>               # escape hatch: raw aws s3 with endpoint+profile
#
# On a pod, either copy ~/.config/r2/credentials over, or set the R2_* env vars directly
# (R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT / R2_BUCKET / R2_PREFIX) before calling.
set -euo pipefail

CRED="${R2_CRED_FILE:-$HOME/.config/r2/credentials}"
if [ -f "$CRED" ]; then
  # shellcheck disable=SC1090
  set -a; source "$CRED"; set +a
fi
: "${R2_ACCESS_KEY_ID:?set R2_ACCESS_KEY_ID (or provide $CRED)}"
: "${R2_SECRET_ACCESS_KEY:?set R2_SECRET_ACCESS_KEY}"
: "${R2_ENDPOINT:?set R2_ENDPOINT}"
: "${R2_BUCKET:?set R2_BUCKET}"
PREFIX="${R2_PREFIX:-neuromorphic}"

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION=auto
AWS=(aws s3 --endpoint-url "$R2_ENDPOINT")
KEY() { echo "s3://$R2_BUCKET/$PREFIX/$1"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  up)    "${AWS[@]}" cp "$1" "$(KEY "$2")" ;;
  down)  "${AWS[@]}" cp "$(KEY "$1")" "$2" ;;
  ls)    "${AWS[@]}" ls "s3://$R2_BUCKET/$PREFIX/${1:-}" ;;
  rm)    "${AWS[@]}" rm "$(KEY "$1")" ;;
  raw)   "${AWS[@]}" "$@" ;;
  *) echo "usage: r2.sh {up|down|ls|rm|raw} ..." >&2; exit 2 ;;
esac
