#!/usr/bin/env bash
set -euo pipefail

# run dir
cd "$(dirname "$0")"

echo ">> Checking output directories"
mkdir -p calibration comparisons cov_id powers simulations

echo ">> Clearing past run csvs"
rm -f comparisons/*.csv cov_id/*.csv powers/*.csv simulations/*.csv calibration/*.csv

echo ">> Step 1/6: tau calibration"
python tau_calibration.py

echo ">> Step 2/6: null simulation"
python null_simulation.py

echo ">> Step 3/6: alt simulation"
python alt_simulation.py

echo ">> Step 4/6: compare statistics"
python compare_stats.py

echo ">> Step 5/6: spectral metrics"
python cov_id_metrics.py

echo ">> Step 6/6: power curves"
python power_vs_eps.py

echo ">> Complete, tables and plots can now be generated."