import numpy as np
import pandas as pd
from collections import defaultdict

NULL_PATH = "simulations/null_simulation.csv"
ALT_PATH  = "simulations/alt_simulation.csv"

ALPHA = 0.05

# One-sided direction per statistic
DIRECTION_BY_STAT = {
    "total_persistence": "lower",
    "tail_count": "upper",
}
# CHECK THIS, tail_count might have to be upper


GROUP_COLS = ["n_pts", "dim", "filtration"]     # calibration keys
STAT_COLS  = ["total_persistence", "tail_count"]

M_TESTS_ANY = len(STAT_COLS) * 2
ALPHA_BONF = ALPHA / M_TESTS_ANY


def empirical_pvalue(val, samples, direction):
    """

    :param val:
    :param samples:
    :param direction:
    :return:
    """
    samples = np.asarray(samples, dtype=float)
    if direction == "lower":
        lt = np.sum(samples < val)
        eq = np.sum(samples == val)
        u = np.random.random()
        return (1.0 + lt + u * eq) / (samples.size + 1.0)
        #return (1.0 + np.sum(samples <= val)) / (samples.size + 1.0)
    elif direction == "upper":
        return (1.0 + np.sum(samples >= val)) / (samples.size + 1.0)
    else:
        raise ValueError("direction must be 'lower' or 'upper'")


def critical_value(samples, alpha, direction):
    """

    :param samples:
    :param alpha:
    :param direction:
    :return:
    """
    samples = np.asarray(samples, dtype=float)
    q = alpha if direction == "lower" else (1 - alpha)
    return float(np.quantile(samples, q))


def build_null_samples(null_df):
    """

    :param null_df:
    :return:
    """
    null_samples = defaultdict(dict)  # (n_pts, dim, filtration) -> stat -> array
    for keys, g in null_df.groupby(GROUP_COLS, dropna=False):
        for stat in STAT_COLS:
            null_samples[keys][stat] = g[stat].to_numpy(dtype=float)
    return null_samples


def calibrate(null_df, alpha):
    """

    :param null_df:
    :param alpha:
    :return:
    """
    null_samples = build_null_samples(null_df)
    crit_lookup = {}
    crit_rows = []

    for key, stat_dict in null_samples.items():
        n_pts, dim, filtration = key
        for stat in STAT_COLS:
            direction = DIRECTION_BY_STAT.get(stat, "lower")
            crit = critical_value(stat_dict[stat], alpha=alpha, direction=direction)

            crit_lookup[(n_pts, dim, filtration, stat)] = (crit, direction)
            crit_rows.append({
                "n_pts": n_pts,
                "dim": dim,
                "filtration": filtration,
                "stat": stat,
                "alpha": alpha,
                "direction": direction,
                "crit": crit,
                "n_null": int(len(stat_dict[stat])),
            })

    crit_df = pd.DataFrame(crit_rows).sort_values(GROUP_COLS + ["stat"]).reset_index(drop=True)
    return crit_df, crit_lookup, null_samples


def apply_tests(df, crit_lookup, null_samples, label):
    """

    :param df:
    :param crit_lookup:
    :param null_samples:
    :param label:
    :return:
    """
    out = []
    for _, r in df.iterrows():
        key_base = (int(r["n_pts"]), int(r["dim"]), str(r["filtration"]))
        for stat in STAT_COLS:
            lk = (key_base[0], key_base[1], key_base[2], stat)
            if lk not in crit_lookup:
                continue

            crit, direction = crit_lookup[lk]
            val = float(r[stat])
            #reject = (val <= crit) if direction == "lower" else (val >= crit)
            #pval = empirical_pvalue(val, null_samples[key_base][stat], direction)
            pval = empirical_pvalue(val, null_samples[key_base][stat], direction)
            # Minimal robust decision rule for discrete/tied stats
            # if stat == "tail_count":
            #     reject = (pval <= ALPHA)
            # else:
            #     reject = (val <= crit) if direction == "lower" else (val >= crit)
            if stat == "tail_count":
                reject = (pval <= ALPHA_BONF)
            else:
                reject = (pval <= ALPHA_BONF)

            row_out = {
                "set": label,
                "point_cloud": r.get("point_cloud", None),
                "n_pts": key_base[0],
                "dim": key_base[1],
                "filtration": key_base[2],
                "seed": int(r.get("seed", -1)),
                "stat": stat,
                "value": val,
                "crit": crit,
                "direction": direction,
                "reject": int(reject),
                "p_value": float(pval),
            }

            # Carry epsilon through if it exists (alternatives will have it)
            if "eps" in df.columns:
                row_out["eps"] = r.get("eps", np.nan)

            out.append(row_out)

    return pd.DataFrame(out)

if __name__ == "__main__":
    null_df = pd.read_csv(NULL_PATH)
    alt_df  = pd.read_csv(ALT_PATH)

    crit_df, crit_lookup, null_samples = calibrate(null_df, alpha=ALPHA)
    null_tested = apply_tests(null_df, crit_lookup, null_samples, label="null")
    alt_tested  = apply_tests(alt_df,  crit_lookup, null_samples, label="alt")
    crit_df.to_csv("comparisons/null_crit.csv", index=False)
    null_tested.to_csv("comparisons/null_tested.csv", index=False)
    alt_tested.to_csv("comparisons/alternatives_tested.csv", index=False)

    # Type I error (overall)
    null_summary = (null_tested.groupby(["point_cloud"] + GROUP_COLS + ["stat"], dropna=False)["reject"]
                    .mean().reset_index().rename(columns={"reject": "rejection_rate"}))

    # Power / rejection rates for alternatives
    if "eps" in alt_tested.columns:
        alt_group = ["point_cloud", "eps"] + GROUP_COLS + ["stat"]
    else:
        alt_group = ["point_cloud"] + GROUP_COLS + ["stat"]

    alt_summary = (alt_tested.groupby(alt_group, dropna=False)["reject"]
                   .mean().reset_index().rename(columns={"reject": "rejection_rate"}))

    null_summary.to_csv("comparisons/null_summary.csv", index=False)
    alt_summary.to_csv("comparisons/alternatives_summary.csv", index=False)

    print("Wrote: null_crit.csv, null_tested.csv, alternatives_tested.csv, null_summary.csv, alternatives_summary.csv")
