inspect_args

# ---- Parse flags -----------------------------------------------------------

test_path=${args[test-path]:-}
level=${args[--level]}
keyword=${args[--keyword]:-}
mark=${args[--mark]:-}
tag=${args[--tag]}
coverage_path=${args[--coverage-path]}
verbose=${args[--verbose]}
workers=${args[--workers]}
no_build=${args[--no-build]}
model=${args[--model]}
live_target=${args[--live-target]}
live_timeout=${args[--live-timeout]}

if [ -z "$tag" ]; then
  tag="cray:latest"
fi

# Repo root, independent of the caller's CWD. The generated CLI lives at
# $repo/scripts/scalarlm; this partial is inlined into it.
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Node image used for the UI stage. Kept in sync with the Dockerfile's
# ui_builder stage so the same toolchain that produces the production bundle
# also runs the tests. A pinned tag is deliberate — tests should not drift
# when Docker Hub moves `latest`.
UI_NODE_IMAGE="node:24.2.0"

# ---- Pick stages based on --level -----------------------------------------
#
# Every stage runs inside a container. The two container types are:
#   - scalarlm CPU image ($tag)         → python -m pytest
#   - $UI_NODE_IMAGE                     → npm lint/typecheck/test
#
# An explicit test-path always short-circuits to the scalarlm container,
# preserving the prior behavior of `./scalarlm test <path>`.

declare -a container_pytest_paths=()
run_ui="no"
# Only component / e2e exercise training + the work queue and therefore
# need SLURM running inside the container. Unit and collectives are pure
# in-process tests; starting slurmctld for them is wasted boot time and
# noisy failures in logs when /app/cray state is stale.
needs_slurm="no"
run_live="no"

if [ -n "$test_path" ]; then
  container_pytest_paths=("$test_path")
  # Explicit paths don't tell us about the level; assume they need slurm,
  # since the previous behavior always started it.
  needs_slurm="yes"
else
  case "$level" in
    unit)
      container_pytest_paths=("test/unit")
      ;;
    component)
      container_pytest_paths=("test/component")
      needs_slurm="yes"
      ;;
    e2e)
      container_pytest_paths=("test/e2e")
      needs_slurm="yes"
      ;;
    ui)
      run_ui="yes"
      ;;
    collectives)
      container_pytest_paths=("test/collectives")
      ;;
    fast)
      container_pytest_paths=("test/unit" "test/collectives")
      run_ui="yes"
      ;;
    cpu)
      container_pytest_paths=("test/unit" "test/component" "test/e2e" "test/collectives")
      run_ui="yes"
      needs_slurm="yes"
      ;;
    live)
      run_live="yes"
      ;;
    all)
      container_pytest_paths=("test/unit" "test/component" "test/e2e" "test/collectives")
      run_ui="yes"
      needs_slurm="yes"
      ;;
    *)
      echo "$(red_bold "Unknown --level: $level")"
      exit 1
      ;;
  esac
fi

# ---- Shared pytest args ---------------------------------------------------

declare -a pytest_args=(
  "-rF" "--color=yes"
  "--durations=10" "--durations-min=10.0"
  "--verbose" "-vv"
  "-o" "log_cli=true" "-o" "log_cli_level=DEBUG"
)

if [ "yes" = "$verbose" ]; then
  pytest_args+=("-rP")
fi

if [ -n "$keyword" ]; then
  pytest_args+=("-k" "$keyword")
fi

if [ -n "$mark" ]; then
  pytest_args+=("-m" "$mark")
fi

# ---- Per-stage reporting --------------------------------------------------

exit_status=0

run_stage() {
  local name="$1"
  shift
  echo "$(blue_bold "== $name ==")"
  if "$@"; then
    echo "$(green_bold "-- $name passed --")"
  else
    echo "$(red_bold "-- $name FAILED --")"
    exit_status=1
  fi
}

tty_flags() {
  if test -t 0; then
    echo "-it"
  else
    echo "-t"
  fi
}

# ---- UI stage (runs inside $UI_NODE_IMAGE) --------------------------------

ui_stage() {
  if [ ! -d "$REPO_ROOT/ui" ]; then
    echo "$(red_bold "ui/ missing at $REPO_ROOT/ui")"
    return 1
  fi

  local TTY
  TTY=$(tty_flags)

  # Bind-mount ui/ into the node container. `npm install --no-audit --no-fund`
  # is used (not `npm ci`) so the dev workflow tolerates package.json
  # changes without forcing the contributor to regenerate package-lock.json
  # before every run. CI still pins a known lockfile via the fast-profile
  # image build.
  #
  # The UI stage runs lint / typecheck / test. Lint is conditionally
  # skipped when eslint is not installed — the repo's `lint` npm script
  # predates any eslint config, so the check stays opportunistic until the
  # UI eslint config lands.
  docker run --rm --init $TTY \
    -v "$REPO_ROOT/ui:/app/ui" \
    -w /app/ui \
    "$UI_NODE_IMAGE" sh -c '
      set -e
      npm install --no-audit --no-fund
      if [ -x node_modules/.bin/eslint ]; then
        npm run lint
      else
        echo "[ui] skipping lint: eslint not installed"
      fi
      npm run typecheck
      npm test
    '
}

if [ "yes" = "$run_ui" ]; then
  run_stage "UI tests ($UI_NODE_IMAGE)" ui_stage
fi

# ---- Pytest stage (runs inside the scalarlm CPU image) --------------------

pytest_stage() {
  local -a paths=("$@")

  if [ "yes" != "$no_build" ]; then
    "$REPO_ROOT/scalarlm" build-image
  fi

  for p in "${paths[@]}"; do
    base_test_path=${p%\*}
    if [ ! -e "$REPO_ROOT/$base_test_path" ]; then
      echo "$(red_bold "Path does not exist: $base_test_path")"
      return 1
    fi
  done

  # The scalarlm CPU image ships a venv but doesn't include pytest. Install
  # it (and plugins) lazily — pip is a no-op when everything is already
  # present so repeated invocations stay fast.
  local pip_install
  pip_install='pip install --quiet --disable-pip-version-check -r test/requirements-pytest.txt'

  # SLURM is only started for levels whose tests actually need it. For
  # unit/collectives we skip it — the reconfigure noise on a fresh
  # container just muddies the logs.
  local slurm_prelude=""
  if [ "yes" = "$needs_slurm" ]; then
    slurm_prelude="./scripts/start_slurm.sh ; "
  fi

  local -a pytest_cmd_parts=(
    "${slurm_prelude}${pip_install} &&"
    "python" "-m" "pytest" "--forked"
  )
  pytest_cmd_parts+=("${pytest_args[@]}")

  if [ "$workers" != "1" ]; then
    pytest_cmd_parts+=("-n" "$workers" "--dist=loadgroup")
  fi

  pytest_cmd_parts+=("${paths[@]}")

  local pytest_cmd="${pytest_cmd_parts[*]}"

  local TTY
  TTY=$(tty_flags)

  # Mirror docker-compose.yaml's bind mounts so tests run against local
  # source, not the (possibly stale) image-baked copy. pytest.ini is at the
  # repo root, so we bind it explicitly on top of $WORKDIR.
  local -a mounts=(
    "-v" "$REPO_ROOT/infra/cray_infra:/app/cray/infra/cray_infra"
    "-v" "$REPO_ROOT/scripts:/app/cray/scripts"
    "-v" "$REPO_ROOT/ml:/app/cray/ml"
    "-v" "$REPO_ROOT/test:/app/cray/test"
    "-v" "$REPO_ROOT/pytest.ini:/app/cray/pytest.ini"
  )

  local -a docker_cmd=(
    "docker" "run" "--rm" "--init" "$TTY"
    "${mounts[@]}"
    "-e" "PY_FORCE_COLOR=1"
    "-e" "PY_COLORS=1"
    "-e" "FORCE_COLOR=1"
    "$tag" "sh" "-c" "$pytest_cmd"
  )

  echo "${docker_cmd[*]}"
  "${docker_cmd[@]}"
}

if [ ${#container_pytest_paths[@]} -gt 0 ]; then
  run_stage "Container pytest (${container_pytest_paths[*]})" \
    pytest_stage "${container_pytest_paths[@]}"
fi

# ---- Live-server stage ------------------------------------------------------

live_stage() {
  local -a live_pytest_filters=()

  if ! [[ "$live_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "$(red_bold "--live-timeout must be a positive integer")"
    return 1
  fi

  if [ "yes" != "$no_build" ]; then
    if [ "cray:latest" != "$tag" ]; then
      echo "$(red_bold "A custom --tag requires --no-build yes; build-image produces cray:latest")"
      return 1
    fi
    "$REPO_ROOT/scalarlm" build-image "$live_target" || return 1
  fi

  if [ -n "$keyword" ]; then
    live_pytest_filters+=(--keyword "$keyword")
  fi
  if [ -n "$mark" ]; then
    live_pytest_filters+=(--mark "$mark")
  fi

  "$REPO_ROOT/test/live/run_live_server_tests.sh" \
    --tag "$tag" \
    --model "$model" \
    --target "$live_target" \
    --timeout "$live_timeout" \
    "${live_pytest_filters[@]}"
}

if [ "yes" = "$run_live" ]; then
  run_stage "Live server ($live_target, $model)" live_stage
fi

# ---- Summary ---------------------------------------------------------------

if [ "$exit_status" -eq 0 ]; then
  echo "$(green_bold "All requested test stages passed.")"
else
  echo "$(red_bold "One or more test stages failed.")"
fi

exit "$exit_status"
