#!/usr/bin/env bash
# Create the isolated Python environments, one per requirement file in envs/.
# Usage: jobs/setup_envs.sh [--apptainer-fallback] [env ...]
# Environments conflict on pins, so each gets its own virtual environment under $ENV_ROOT.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REQUIRED_ENVS=(core train distill eval quantize serve-sglang serve-vllm)
OPTIONAL_ENVS=(distill-vllm eval-vllm)
PYTHON_VERSION="3.11"
USE_FALLBACK=0
TARGETS=()

for arg in "$@"; do
  case "$arg" in
    --apptainer-fallback) USE_FALLBACK=1 ;;
    -h | --help)
      sed -n '2,4p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) TARGETS+=("$arg") ;;
  esac
done
if [ "${#TARGETS[@]}" -eq 0 ]; then
  TARGETS=("${REQUIRED_ENVS[@]}" "${OPTIONAL_ENVS[@]}")
fi

is_optional() {
  local name
  for name in "${OPTIONAL_ENVS[@]}"; do
    [ "$name" = "$1" ] && return 0
  done
  return 1
}

create_env() {
  local name="$1" req="$REPO_DIR/envs/$1.txt" dest="$ENV_ROOT/$1"
  [ -f "$req" ] || die "missing requirement file $req"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYTHON_VERSION" --allow-existing "$dest"
    uv pip install --python "$dest/bin/python" -r "$req"
  else
    "python$PYTHON_VERSION" -m venv "$dest"
    "$dest/bin/python" -m pip install -r "$req"
  fi
}

# pull the official container for a serving environment when pip install fails
pull_image() {
  local name="$1" image
  image="$(awk -v n="$name" '$1 == n {print $2}' "$REPO_DIR/envs/images.txt")"
  [ -n "$image" ] || return 1
  command -v apptainer >/dev/null 2>&1 || return 1
  apptainer pull --force "$ENV_ROOT/$name.sif" "$image"
}

for name in "${TARGETS[@]}"; do
  log "creating environment $name"
  if create_env "$name"; then
    log "environment $name ready (virtual environment)"
  elif [ "$USE_FALLBACK" -eq 1 ] && pull_image "$name"; then
    log "environment $name ready (apptainer image)"
  elif is_optional "$name"; then
    log "optional environment $name skipped"
  else
    die "environment $name failed to install; retry with --apptainer-fallback for serving environments"
  fi
done
