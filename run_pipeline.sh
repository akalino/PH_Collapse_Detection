#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config.local.json}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[INFO] Using config: ${CONFIG_PATH}"

is_enabled() {
  python - "$CONFIG_PATH" "$1" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
stage = sys.argv[2]
print(str(cfg.get(stage, {}).get("enabled", False)).lower())
PY
}

run_stage() {
  local stage_name="$1"
  shift
  echo "[INFO] Starting ${stage_name}..."
  "$@"
  echo "[INFO] Finished ${stage_name}"
}

if [[ "$(is_enabled tau_parallel)" == "true" ]]; then
  run_stage "tau calibration" python tau_parallel.py --config "${CONFIG_PATH}"
fi

if [[ "$(is_enabled null_parallel)" == "true" ]]; then
  run_stage "null simulations" python null_parallel.py --config "${CONFIG_PATH}"
fi

if [[ "$(is_enabled alt_parallel)" == "true" ]]; then
  run_stage "alternative simulations" python alt_parallel.py --config "${CONFIG_PATH}"
fi

run_stage "statistic comparison" python compare_stats.py --config "${CONFIG_PATH}"
run_stage "covariance / intrinsic-dimension metrics" python cov_id_metrics.py --config "${CONFIG_PATH}"
run_stage "power aggregation" python power_vs_eps.py --config "${CONFIG_PATH}"
run_stage "table generation" python gen_tables.py --config "${CONFIG_PATH}"
run_stage "plot generation" python gen_plots.py --config "${CONFIG_PATH}"

echo "[INFO] Outputs are written directly to persistent storage."
echo "[INFO] Pipeline completed successfully."