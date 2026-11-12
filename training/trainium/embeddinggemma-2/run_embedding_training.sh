#!/usr/bin/env bash
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
recipe_venv="${RECIPE_VENV:-$recipe_dir/.venv}"
if [[ ! -x "$recipe_venv/bin/python" ]]; then
  echo "Run setup_embedding_environment.py first, or set RECIPE_VENV to a configured environment." >&2
  exit 1
fi
# Cooperative lock only: unrelated workloads must be stopped by their owner.
command -v flock >/dev/null || { echo "Install util-linux (flock)." >&2; exit 1; }
runtime_dir="${TRAINIUM_RECIPE_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/tmp/trainium-recipe-$UID}}"
if [[ -L "$runtime_dir" ]]; then
  echo "Refusing symbolic-link runtime directory: $runtime_dir" >&2
  exit 1
fi
mkdir -p -m 700 -- "$runtime_dir"
read -r runtime_owner runtime_mode < <(stat -c '%u %a' -- "$runtime_dir")
if [[ "$runtime_owner" != "$UID" || "$runtime_mode" != "700" ]]; then
  echo "Runtime directory must be owned by UID $UID with mode 700: $runtime_dir" >&2
  exit 1
fi
lock_path="$runtime_dir/embeddinggemma-2.lock"
if [[ -L "$lock_path" ]]; then
  echo "Refusing symbolic-link lock file: $lock_path" >&2
  exit 1
fi
exec 9>"$lock_path"
flock -n 9 || { echo "Another recipe run owns the accelerator lock." >&2; exit 1; }
exec "$recipe_venv/bin/python" "$recipe_dir/train_embedding_standalone.py" "$@"
