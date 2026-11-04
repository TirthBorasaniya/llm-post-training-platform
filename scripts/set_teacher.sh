#!/usr/bin/env bash
# Point the teacher location at the merged checkpoint of an aligned stage.
# Usage: scripts/set_teacher.sh STAGE   (dpo, sft, orpo, or simpo)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../jobs/common.sh"
STAGE="${1:?usage: scripts/set_teacher.sh STAGE}"
SOURCE="$CKPT_ROOT/$STAGE/merged"
[ -f "$SOURCE/config.json" ] || die "no merged checkpoint at $SOURCE"
mkdir -p "$CKPT_ROOT/teacher"
ln -sfn "$SOURCE" "$CKPT_ROOT/teacher/merged"
echo "teacher -> $SOURCE"
