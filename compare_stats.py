# analyze_ph_tests.py
import numpy as np
import pandas as pd
from collections import defaultdict

NULL_PATH = "null.csv"
ALT_PATH  = "alt_simulation.csv"

# --------------------------
# User settings
# --------------------------
ALPHA = 0.05

# Choose which null point clouds define "H0" for calibration.
# If you set this to BOTH, you are calibrating a mixture null.
CALIB_NULLS = ["gaussian", "noisy_sphere"]   # or ["gaussian"]

# One-sided direction per statistic:
# For collapse detection, you often want LOWER-tail (collapse => less high-dim topology).
# But if you find Type I is weird for tail_count, try setting tail_count to "upper".
DIRECTION_BY_STAT = {
    "total_persistence": "lower",
    "tail_count": "lower",
}

# Calibration granularity
GROUP_COLS = ["n_pts", "dim", "filtration"]
STAT_COLS  = ["total_persistence", "tail_count"]


# --------------------------
# Helpers
# --------------------------
def empirical_pvalue(val: float, samples: np.ndarray, direction: str) -> float:
    """Conservative empirical one-sided p-value."""
    samples = np.asarray(samples, dtype=float)
    if direction == "lower":
        return (1.0 + np.sum(samples <= val)) / (samples.size + 1.0)
    elif direction == "upper":
        return (1.0 + np.sum(samples >= val)) / (samples.size + 1.0)
    else:
        raise ValueError("direction must be 'lower' or 'upper'")

def critical_value(samples: np.ndarray, alpha: float, direction: str) -> float:
    """One-sided critical value."""
    samples = np.asarray(samples, dtype=float)
    q = alpha if direction == "lower" else (1 - alpha)
    return float(np.quantile(samples, q))

def build_null_samples(null_df: pd.DataFrame) -> dict:
    """
    Returns:
      null_samples[(n_pts, dim, filtration)][stat] -> np.array of samples
    """
    null_samples = defaultdict(dict)
    for keys, g in null_df.groupby(GROUP_COLS, dropna=False):
        for stat in STAT_COLS:
            null_samples[keys][stat] = g[stat].to_numpy(dtype=float)
    return null_samples

def calibrate(null_df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      crit_df: long DF with crit values per (n_pts,dim,filtration,stat)
      crit_lookup: dict[(n_pts,dim,filtration,stat)] = (crit, direction)
    """
    rows = []
    crit_lookup = {}

    null_samples = build_null_samples(null_df)

    for key, stat_dict in null_samples.items():
        n_pts, dim, filtration = key
        for stat in STAT_COLS:
            direction = DIRECTION_BY_STAT.get(stat, "lower")
            crit = critical_value(stat_dict[stat], alpha=alpha, direction=direction)

            rows.append({
                "n_pts": n_pts,
                "dim": dim,
                "filtration": filtration,
                "stat": stat,
                "alpha": alpha,
                "direction": direction,
                "crit": crit,
                "n_null": int(len(stat_dict[stat])),
            })
            crit_lookup[(n_pts, dim, filtration, stat)] = (crit, direction)

    crit_df = pd.DataFrame(rows).sort_values(GROUP_COLS + ["stat"]).reset_index(drop=True)
    return crit_df, crit_lookup, null_samples

def apply_tests(df: pd.DataFrame, crit_lookup: dict, null_samples: dict, label: str) -> pd.DataFrame:
    """
    Produces a long DF with one row per (original row × statistic).
    Adds reject + p_value.
    """
    out_rows = []

    for _, r in df.iterrows():
        key_base = (int(r["n_pts"]), int(r["dim"]), str(r["filtration"]))

        for stat in STAT_COLS:
            lk = (key_base[0], key_base[1], key_base[2], stat)
            if lk not in crit_lookup:
                # No calibration available for this (n,d,filtration)
                continue

            crit, direction = crit_lookup[lk]
            val = float(r[stat])

            reject = (val <= crit) if direction == "lower" else (val >= crit)
            pval = empirical_pvalue(val, null_samples[key_base][stat], direction)

            out_rows.append({
                "set": label,
                "point_cloud": r["point_cloud"],
                "n_pts": key_base[0],
                "dim": key_base[1],
                "filtration": key_base[2],
                "seed": int(r["seed"]),
                "stat": stat,
                "value": val,
                "crit": crit,
                "direction": direction,
                "reject": int(reject),
                "p_value": float(pval),
            })

    return pd.DataFrame(out_rows)

def summarize_rejections(tested: pd.DataFrame, by_point_cloud: bool = True) -> pd.DataFrame:
    group = (["point_cloud"] if by_point_cloud else []) + GROUP_COLS + ["stat"]
    return (tested.groupby(group, dropna=False)["reject"]
            .mean()
            .reset_index()
            .rename(columns={"reject": "rejection_rate"}))


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    null_all = pd.read_csv(NULL_PATH)
    alt_all  = pd.read_csv(ALT_PATH)

    # Choose calibration null subset
    null_calib = null_all[null_all["point_cloud"].isin(CALIB_NULLS)].copy()

    # Calibrate
    crit_df, crit_lookup, null_samples = calibrate(null_calib, alpha=ALPHA)

    # Identify missing calibration combos in alternatives
    null_groups = set(tuple(x) for x in null_calib[GROUP_COLS].drop_duplicates().to_numpy())
    alt_groups  = set(tuple(x) for x in alt_all[GROUP_COLS].drop_duplicates().to_numpy())
    missing = sorted(list(alt_groups - null_groups))

    if missing:
        print("\nWARNING: Some alternative (n_pts,dim,filtration) combos are not present in calibration nulls.")
        print("They will be skipped in testing. Missing combos:")
        for m in missing:
            print("  ", m)

    # Apply tests
    null_tested = apply_tests(null_all, crit_lookup, null_samples, label="null")
    alt_tested  = apply_tests(alt_all,  crit_lookup, null_samples, label="alt")

    # Summaries
    # Correct Type I check = on the SAME pooled null used for calibration:
    null_type1 = null_tested[null_tested["point_cloud"].isin(CALIB_NULLS)].copy()

    print("\n=== Critical values (first 20 rows) ===")
    print(crit_df.head(20).to_string(index=False))

    print("\n=== Type I error estimates (pooled over calibration nulls) ===")
    print(summarize_rejections(null_type1, by_point_cloud=False).to_string(index=False))

    print("\n=== Type I error by null point cloud (diagnostic) ===")
    print(summarize_rejections(null_type1, by_point_cloud=True).to_string(index=False))

    print("\n=== Alternative rejection rates (empirical power) ===")
    print(summarize_rejections(alt_tested, by_point_cloud=True).to_string(index=False))

    # Optional: save run-level tested outputs
    null_tested.to_csv("null_tested.csv", index=False)
    alt_tested.to_csv("alternatives_tested.csv", index=False)
    crit_df.to_csv("null_crit.csv", index=False)
    print("\nWrote: null_crit.csv, null_tested.csv, alternatives_tested.csv")
