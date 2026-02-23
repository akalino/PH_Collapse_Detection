import numpy as np
import pandas as pd
from collections import defaultdict

NULL_PATH = "null.csv"
ALT_PATH  = "alt_simulation.csv"
OUT_PATH  = "alternatives_agg_report.csv"

ALPHA = 0.05

# Calibration keys: epsilon is NOT part of null calibration
GROUP_COLS = ["n_pts", "dim", "filtration"]
STAT_COLS  = ["total_persistence", "tail_count"]

# One-sided direction per statistic
DIRECTION_BY_STAT = {
    "total_persistence": "lower",
    "tail_count": "lower",
}

# -----------------------
# Helpers
# -----------------------
def empirical_pvalue(val: float, samples: np.ndarray, direction: str) -> float:
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
    null_samples = defaultdict(dict)  # (n_pts, dim, filtration) -> stat -> array
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
    """
    Returns long-form tested rows: one per (original row × stat).
    """
    required = {"point_cloud","n_pts","dim","filtration","seed"}
    missing = required - set(alt_df.columns)
    if missing:
        raise ValueError(f"alternatives.csv missing required columns: {missing}")

    if "eps" not in alt_df.columns:
        raise ValueError("alternatives.csv does not have an 'epsilon' column (you said you added it).")

    out = []
    for _, r in alt_df.iterrows():
        key_base = (int(r["n_pts"]), int(r["dim"]), str(r["filtration"]))
        for stat in STAT_COLS:
            lk = (key_base[0], key_base[1], key_base[2], stat)
            if lk not in crit_lookup:
                # no calibration available for this (n,d,filtration)
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
                "value": val,
                "crit": float(crit),
                "direction": direction,
                "reject": int(reject),
                "p_value": float(pval),
            })

    return pd.DataFrame(out)

def summarize_alt(tested: pd.DataFrame) -> pd.DataFrame:
    """
    Summary keyed by (point_cloud, n_pts, dim, epsilon, filtration, stat).
    Includes reject counts/rate and mean p-values split by reject.
    """
    grp = ["point_cloud", "n_pts", "dim", "eps", "filtration", "stat"]
    g = tested.groupby(grp, dropna=False)

    base = g.agg(
        n_runs=("seed", "nunique"),
        n_rows=("reject", "size"),
        n_reject=("reject", "sum"),
        reject_rate=("reject", "mean"),
        p_mean=("p_value", "mean"),
    ).reset_index()

    # Mean p-values conditional on reject vs not
    def _mean_if(df, cond):
        sub = df.loc[cond, "p_value"]
        return float(sub.mean()) if len(sub) else np.nan

    split_rows = []
    for keys, df in tested.groupby(grp, dropna=False):
        split_rows.append({
            **dict(zip(grp, keys)),
            "p_mean_reject": _mean_if(df, df["reject"] == 1),
            "p_mean_not_reject": _mean_if(df, df["reject"] == 0),
        })
    split = pd.DataFrame(split_rows)

    return base.merge(split, on=grp, how="left")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    null_df = pd.read_csv(NULL_PATH)
    alt_df  = pd.read_csv(ALT_PATH)

    crit_lookup, null_samples = calibrate(null_df, alpha=ALPHA)
    alt_tested = test_alternatives(alt_df, crit_lookup, null_samples)

    report = summarize_alt(alt_tested)

    # Optional: also make an aggregated view across filtration/stat (one row per point_cloud,n,d,eps)
    agg_keys = ["point_cloud", "n_pts", "dim", "eps"]
    agg = (alt_tested.groupby(agg_keys, dropna=False)
           .agg(
                n_runs=("seed", "nunique"),
                n_tests=("reject", "size"),
                n_reject=("reject", "sum"),
                reject_rate=("reject", "mean"),
                p_mean=("p_value", "mean"),
           ).reset_index())

    report.to_csv(OUT_PATH, index=False)
    agg.to_csv("alternatives_report_agg.csv", index=False)

    print(f"Wrote {OUT_PATH} (by filtration/stat) and alternatives_report_agg.csv (aggregated).")
    print("\nHead of aggregated report:")
    print(agg.head(20).to_string(index=False))
