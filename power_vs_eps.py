import argparse
import os

import pandas as pd

from config_utils import load_config, resolve_output


PRIMARY_FILTRATION = "vr"
PRIMARY_STAT = "total_persistence"


def run_level_power_any(tested: pd.DataFrame) -> pd.DataFrame:
    key = ["point_cloud", "eps", "n_pts", "dim", "seed"]
    per_run = (
        tested.groupby(key, dropna=False)["reject"]
        .max()
        .reset_index()
        .rename(columns={"reject": "run_reject_any"})
    )

    power = (
        per_run.groupby(["point_cloud", "eps", "n_pts", "dim"], dropna=False)["run_reject_any"]
        .mean()
        .reset_index()
        .rename(columns={"run_reject_any": "power_any"})
    )
    return power


def run_level_power_all(tested: pd.DataFrame) -> pd.DataFrame:
    key = ["point_cloud", "eps", "n_pts", "dim", "seed"]
    per_run = (
        tested.groupby(key, dropna=False)["reject"]
        .min()
        .reset_index()
        .rename(columns={"reject": "run_reject_all"})
    )

    power = (
        per_run.groupby(["point_cloud", "eps", "n_pts", "dim"], dropna=False)["run_reject_all"]
        .mean()
        .reset_index()
        .rename(columns={"run_reject_all": "power_all"})
    )
    return power


def run_level_power_primary(tested: pd.DataFrame, filtration: str, stat: str) -> pd.DataFrame:
    sub = tested[(tested["filtration"] == filtration) & (tested["stat"] == stat)].copy()
    key = ["point_cloud", "eps", "n_pts", "dim", "seed"]

    per_run = (
        sub.groupby(key, dropna=False)["reject"]
        .max()
        .reset_index()
        .rename(columns={"reject": "run_reject_primary"})
    )

    power = (
        per_run.groupby(["point_cloud", "eps", "n_pts", "dim"], dropna=False)["run_reject_primary"]
        .mean()
        .reset_index()
        .rename(columns={"run_reject_primary": "power_primary"})
    )
    return power


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    alt_tested_path = resolve_output(cfg, "comparisons/alternatives_tested.csv")
    out_any = resolve_output(cfg, "powers/power_any.csv")
    out_all = resolve_output(cfg, "powers/power_all.csv")
    out_primary = resolve_output(cfg, "powers/power_primary.csv")

    tested = pd.read_csv(alt_tested_path)

    required = {"point_cloud", "eps", "n_pts", "dim", "seed", "filtration", "stat", "reject"}
    missing = required - set(tested.columns)
    if missing:
        raise ValueError(f"{alt_tested_path} missing required columns: {missing}")

    os.makedirs(os.path.dirname(out_any), exist_ok=True)

    power_any = run_level_power_any(tested)
    power_all = run_level_power_all(tested)
    power_primary = run_level_power_primary(
        tested,
        filtration=PRIMARY_FILTRATION,
        stat=PRIMARY_STAT,
    )

    power_any.to_csv(out_any, index=False)
    power_all.to_csv(out_all, index=False)
    power_primary.to_csv(out_primary, index=False)

    print(f"Wrote: {out_any}, {out_all}, {out_primary}")