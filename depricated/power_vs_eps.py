import argparse

import numpy as np
import pandas as pd

from collections import defaultdict

from config_utils import load_config, resolve_output


ALT_TESTED_PATH = "comparisons/alternatives_tested.csv"  # leave None to recompute

GROUP_COLS = ["n_pts", "dim", "filtration"]
STAT_COLS  = ["total_persistence", "tail_count"]

DIRECTION_BY_STAT = {"total_persistence": "lower", "tail_count": "lower"}

PRIMARY_FILTRATION = "vr"          # "dtm" or "vr"
PRIMARY_STAT = "total_persistence"  # "total_persistence" or "tail_count"


# NOTE: could strip these out as duplicates to their own stat_utils.py
def empirical_pvalue(val, samples, direction):
    """

    :param val:
    :param samples:
    :param direction:
    :return:
    """
    samples = np.asarray(samples, dtype=float)
    if direction == "lower":
        return (1.0 + np.sum(samples <= val)) / (samples.size + 1.0)
    elif direction == "upper":
        return (1.0 + np.sum(samples >= val)) / (samples.size + 1.0)
    else:
        raise ValueError("direction must be 'lower' or 'upper'")


def critical_value(samples: np.ndarray, alpha: float, direction: str) -> float:
    samples = np.asarray(samples, dtype=float)
    q = alpha if direction == "lower" else (1 - alpha)
    return float(np.quantile(samples, q))


def build_null_samples(null_df: pd.DataFrame) -> dict:
    null_samples = defaultdict(dict)
    for keys, g in null_df.groupby(GROUP_COLS, dropna=False):
        for stat in STAT_COLS:
            null_samples[keys][stat] = g[stat].to_numpy(dtype=float)
    return null_samples


def calibrate(null_df: pd.DataFrame, alpha: float):
    null_samples = build_null_samples(null_df)
    crit_lookup = {}
    for key, stat_dict in null_samples.items():
        n_pts, dim, filtration = key
        for stat in STAT_COLS:
            direction = DIRECTION_BY_STAT.get(stat, "lower")
            crit = critical_value(stat_dict[stat], alpha=alpha, direction=direction)
            crit_lookup[(n_pts, dim, filtration, stat)] = (crit, direction)
    return crit_lookup, null_samples


def test_alternatives(alt_df: pd.DataFrame, crit_lookup: dict, null_samples: dict) -> pd.DataFrame:
    required = {"point_cloud","n_pts","dim","filtration","seed","eps"}
    missing = required - set(alt_df.columns)
    if missing:
        raise ValueError(f"alternatives.csv missing required columns: {missing}")

    out = []
    for _, r in alt_df.iterrows():
        key_base = (int(r["n_pts"]), int(r["dim"]), str(r["filtration"]))
        for stat in STAT_COLS:
            lk = (key_base[0], key_base[1], key_base[2], stat)
            if lk not in crit_lookup:
                continue

            crit, direction = crit_lookup[lk]
            val = float(r[stat])
            reject = (val <= crit) if direction == "lower" else (val >= crit)
            pval = empirical_pvalue(val, null_samples[key_base][stat], direction)

            out.append({
                "point_cloud": r["point_cloud"],
                "eps": float(r["eps"]),
                "n_pts": key_base[0],
                "dim": key_base[1],
                "filtration": key_base[2],
                "seed": int(r["seed"]),
                "stat": stat,
                "reject": int(reject),
                "p_value": float(pval),
            })
    return pd.DataFrame(out)


# Run-level aggregation rules
def run_level_power_any(tested: pd.DataFrame) -> pd.DataFrame:
    """
    For each run (seed), reject if ANY test rejects.
    """
    key = ["point_cloud","eps","n_pts","dim","seed"]
    per_run = (tested.groupby(key, dropna=False)["reject"]
               .max()
               .reset_index()
               .rename(columns={"reject":"run_reject_any"}))

    power = (per_run.groupby(["point_cloud","eps","n_pts","dim"], dropna=False)["run_reject_any"]
             .mean()
             .reset_index()
             .rename(columns={"run_reject_any":"power_any"}))
    return power


def run_level_power_all(tested: pd.DataFrame) -> pd.DataFrame:
    """
    For each run (seed), reject if ALL available tests reject.
    """
    key = ["point_cloud","eps","n_pts","dim","seed"]
    per_run = (tested.groupby(key, dropna=False)["reject"]
               .min()
               .reset_index()
               .rename(columns={"reject":"run_reject_all"}))

    power = (per_run.groupby(["point_cloud","eps","n_pts","dim"], dropna=False)["run_reject_all"]
             .mean()
             .reset_index()
             .rename(columns={"run_reject_all":"power_all"}))
    return power


def run_level_power_primary(tested: pd.DataFrame, filtration: str, stat: str) -> pd.DataFrame:
    """
    For each run (seed), use a single chosen filtration+stat.
    """
    sub = tested[(tested["filtration"] == filtration) & (tested["stat"] == stat)].copy()
    key = ["point_cloud","eps","n_pts","dim","seed"]
    # one row per run for that test; if duplicates exist, max is safe
    per_run = (sub.groupby(key, dropna=False)["reject"]
               .max()
               .reset_index()
               .rename(columns={"reject":"run_reject_primary"}))

    power = (per_run.groupby(["point_cloud","eps","n_pts","dim"], dropna=False)["run_reject_primary"]
             .mean()
             .reset_index()
             .rename(columns={"run_reject_primary":"power_primary"}))
    return power


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    shared = cfg["shared"]
    null_stage = cfg["null_parallel"]
    alt_stage = cfg["alt_parallel"]
    run = cfg["run"]

    NULL_PATH = resolve_output(cfg, null_stage["out_path"])
    ALT_PATH  = resolve_output(cfg, alt_stage["out_path"])
    ALPHA = shared["alpha"]

    if ALT_TESTED_PATH is not None:
        tested = pd.read_csv(ALT_TESTED_PATH)
        # must include point_cloud, epsilon, n_pts, dim, seed, filtration, stat, reject
        required = {"point_cloud","eps","n_pts","dim","seed","filtration","stat","reject"}
        missing = required - set(tested.columns)
        if missing:
            raise ValueError(f"{ALT_TESTED_PATH} missing required columns: {missing}")
    else:
        null_df = pd.read_csv(NULL_PATH)
        alt_df  = pd.read_csv(ALT_PATH)
        crit_lookup, null_samples = calibrate(null_df, alpha=ALPHA)
        tested = test_alternatives(alt_df, crit_lookup, null_samples)

    power_any = run_level_power_any(tested)
    power_all = run_level_power_all(tested)
    power_primary = run_level_power_primary(tested, filtration=PRIMARY_FILTRATION, stat=PRIMARY_STAT)

    power_any.to_csv("powers/power_any.csv", index=False)
    power_all.to_csv("powers/power_all.csv", index=False)
    power_primary.to_csv("powers/power_primary.csv", index=False)

    print("Wrote: power_any.csv, power_all.csv, power_primary.csv")
