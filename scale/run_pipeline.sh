#!/usr/bin/env bash
set -euo pipefail

# run dir
cd "$(dirname "$0")"

echo ">> [SETUP] Checking output directories"
mkdir -p calibration comparisons cov_id powers simulations

echo ">> [SETUP] Clearing past run csvs, keeping calibration"
rm -f comparisons/*.csv cov_id/*.csv powers/*.csv simulations/*.csv results/*.csv

echo ">> [SIMULATE] Step 1/6: tau calibration (map pre-computed)"
python tau_parallel.py

echo ">> [SIMULATE] Step 2/6: null simulation"
python null_parallel.py

echo ">> [SIMULATE] Step 3/6: alt simulation"
python alt_parallel.py

echo ">> [SIMULATE] Step 4/6: compare statistics"
python compare_stats.py

echo ">> [SIMULATE] Step 5/6: spectral metrics"
python cov_id_metrics.py

echo ">> [SIMULATE] Step 6/6: power curves"
python power_vs_eps.py

echo ">> Complete, tables and plots can now be generated."

echo ">> [ASSETS] moving result csvs to single dir"
mkdir -p results

cp calibration/*.csv results/
cp comparisons/*.csv results/
cp cov_id/*.csv results/
cp powers/*.csv results/
cp simulations/*.csv results/

echo ">> [ASSETS] generating paper assets"
python gen_tables.py --in results/ --out assets/
python gen_plots.py --in results/ --out assets/
