import pandas as pd
import numpy as np

NULL_PATH = "null.csv"
ALT_PATH  = "alternatives.csv"

# -----------------------------
# Configuration
# -----------------------------
ALPHA = 0.05

# If "collapse" should yield SMALLER values of the statistic, use lower-tail tests.
# (This is the usual choice for "dims > k total persistence" and "tail count in dims > k".)
TEST_DIRECTION = "lower"   # "lower" or "upper"

# Grouping columns that define a "regime" where you calibrate separately.
# Your files include dtm_n_pts and dtm_dim; we’ll use those as (n, d).
GROUP_COLS = ["dtm_n_pts", "dtm_dim"]

# Which point_cloud labels count as "null" for calibration (edit if your labels differ)
NULL_LABELS = {"gaussian", "noisy_sphere"}

# Which statistic columns to test (edit if you add Čech later)
STAT_COLS = [
    "vr_total_persistence",
    "vr_tail_count",
    "dtm_total_persistence",
    "dtm_dtm_tail_count",
]


# -----------------------------
# Core helpers
# -----------------------------
def critical_value(values: np.ndarray, alpha: float, direction: str) -> float:
    """Return the critical value for a one-sided test."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    q = alpha if direction == "lower" else (1 - alpha)
    return float(np.quantile(values, q))


def calibrate_criticals(null_df: pd.DataFrame,
                        stat_cols,
                        group_cols,
                        alpha: float,
                        direction: str) -> pd.DataFrame:
    """
    Calibrate critical values from null_df for each (group_cols) × stat.
    Returns a DataFrame with columns: group_cols + ["stat", "crit", "n_null"].
    """
    rows = []
    for keys, g in null_df.groupby(group_cols, dropna=False):
        # keys is a scalar if len(group_cols)==1 else a tuple
        key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for stat in stat_cols:
            crit = critical_value(g[stat].to_numpy(), alpha=alpha, direction=direction)
            rows.append({**key_dict, "stat": stat, "crit": crit, "n_null": int(g.shape[0])})
    return pd.DataFrame(rows)


def apply_tests(df: pd.DataFrame,
                crit_df: pd.DataFrame,
                stat_cols,
                group_cols,
                direction: str) -> pd.DataFrame:
    """
    Adds columns reject_<stat> to df using crit_df.
    """
    out = df.copy()

    # Reshape crit_df to wide form so we can merge once
    crit_wide = crit_df.pivot_table(index=group_cols, columns="stat", values="crit").reset_index()

    out = out.merge(crit_wide, on=group_cols, how="left", suffixes=("", "_crit"))

    for stat in stat_cols:
        crit_col = stat  # after pivot, crit values live in a column named like the stat
        val = out[stat].astype(float)
        crit = out[crit_col].astype(float)

        if direction == "lower":
            out[f"reject_{stat}"] = (val <= crit).astype(int)
        else:
            out[f"reject_{stat}"] = (val >= crit).astype(int)

    # Optional: drop the crit columns if you don’t want them lingering
    # out = out.drop(columns=stat_cols)

    return out


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    null_df = pd.read_csv(NULL_PATH)
    alt_df  = pd.read_csv(ALT_PATH)

    # Filter null calibration set by labels (adjust if needed)
    null_calib = null_df[null_df["point_cloud"].isin(NULL_LABELS)].copy()

    # Sanity check that group columns exist
    for c in GROUP_COLS:
        if c not in null_calib.columns or c not in alt_df.columns:
            raise ValueError(f"Missing grouping column {c} in one of the CSVs.")

    # 1) Calibrate
    crit_df = calibrate_criticals(
        null_df=null_calib,
        stat_cols=STAT_COLS,
        group_cols=GROUP_COLS,
        alpha=ALPHA,
        direction=TEST_DIRECTION,
    )
    print("\n=== Critical values (calibration) ===")
    print(crit_df.sort_values(GROUP_COLS + ["stat"]).to_string(index=False))

    # 2) Apply tests to alternatives
    alt_tested = apply_tests(
        df=alt_df,
        crit_df=crit_df,
        stat_cols=STAT_COLS,
        group_cols=GROUP_COLS,
        direction=TEST_DIRECTION,
    )

    print("\n=== Alternatives with reject flags (first few rows) ===")
    show_cols = ["point_cloud"] + GROUP_COLS + STAT_COLS + [f"reject_{s}" for s in STAT_COLS]
    print(alt_tested[show_cols].head(10).to_string(index=False))

    # 3) Summarize “power”: rejection rate by dataset and (n,d)
    reject_cols = [f"reject_{s}" for s in STAT_COLS]
    summary = (alt_tested
               .groupby(["point_cloud"] + GROUP_COLS, dropna=False)[reject_cols]
               .mean()
               .reset_index())

    print("\n=== Rejection rates (empirical power) ===")
    print(summary.to_string(index=False))
