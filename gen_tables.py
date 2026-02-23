"""
Generates the initial results latex tables.
"""
import argparse
import os

import numpy as np
import pandas as pd


MECHANISM_MAP = {
    "contaminated_kcube": "C",
    "contaminated_kplane": "C",
    "contaminated_sphere": "C",
    "kplane": "A",
    "spiked_gaussian": "A",
    "torus": "B",
    "swiss": "B",
    "paraboloid": "B",
    "paraboloid_graph": "B",
}

STAT_NAMES = {
    "total_persistence": "TP",
    "mean_excess_tail": "MTE",
    "mean_tail_excess": "MTE",
    "tail_count": "MTE",
}

FILTRATION_NAMES = {"vr": "VR", "dtm": "DTM"}

# helpers

def fmt(x):
    """
    Rounding function.
    Parameters
    ----------
    x

    Returns
    -------

    """
    if pd.isna(x):
        return ""
    return f"{float(x):.3f}"


def latex_escape(s):
    """
    Fixes string formatting for tables.
    Parameters
    ----------
    s

    Returns
    -------

    """
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def to_latex_table(df, caption, label):
    """

    Parameters
    ----------
    df
    caption
    label

    Returns
    -------

    """
    cols = list(df.columns)
    header = " & ".join([f"\\textbf{{{latex_escape(c)}}}" for c in cols]) + " \\\\"
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append(" & ".join([latex_escape(x) for x in row.tolist()]) + " \\\\")
    body = "\n".join(body_lines)
    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular}}{{{"l" * len(cols)}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    return latex

# calibration


def make_calibration_table(_null_summary, _out_path):
    """
    Builds latex calibration table.
    Parameters
    ----------
    _null_summary: Null simulation dataframe.
    _out_path: Expected output path.

    Returns
    -------
    None, writes table to tex file.
    """
    df = _null_summary.copy()
    df["stat"] = df["stat"].map(lambda x: STAT_NAMES.get(x, x))
    df["filtration"] = df["filtration"].map(lambda x: FILTRATION_NAMES.get(x, x))

    # mean rejection rate per (null, filtration, stat), averaged over (n,d)
    g = df.groupby(["point_cloud", "filtration", "stat"], as_index=False)["rejection_rate"].mean()

    # pivot to columns: VR-TP, VR-MTE, DTM-TP, DTM-MTE
    g["col"] = g["filtration"] + "-" + g["stat"]
    wide = g.pivot(index="point_cloud", columns="col", values="rejection_rate").reset_index()

    # FWER(any) upper bound approx: 1 - Π(1 - p_i) over the four tests
    # using mean rates (quick, readable)
    cols = [c for c in wide.columns if c != "point_cloud"]
    for c in cols:
        if c not in wide.columns:
            wide[c] = np.nan
    pcols = [c for c in ["VR-TP", "VR-MTE", "DTM-TP", "DTM-MTE"]
             if c in wide.columns]
    p = wide[pcols].fillna(0.0).to_numpy()
    wide["FWER(any)~"] = 1.0 - np.prod(1.0 - p, axis=1)

    order_cols = ["point_cloud", "VR-TP", "VR-MTE", "DTM-TP", "DTM-MTE", "FWER(any)~"]
    for c in order_cols:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide[order_cols]

    wide = wide.sort_values("FWER(any)~", ascending=False)
    wide = wide.reset_index(drop=True)

    for c in wide.columns[1:]:
        wide[c] = wide[c].map(fmt)

    caption = (
        "Table [TODO]: Null calibration. Entries are mean empirical rejection rates "
        "averaged over the (n,d) grid for each null class and each test "
        "(filtration × statistic). The final column is an approximate FWER(any) computed "
        "from the four mean rates as $1-\\prod_i(1-p_i)$ (for quick comparison)."
    )
    label = "tab:calibration"

    tex = to_latex_table(wide, caption=caption, label=label)
    with open(_out_path, "w", encoding="utf-8") as f:
        f.write(tex)

# mechanism

def make_mechanism_map(_alt_summary, _out_path):
    """
    Builds latex mechanism map table.
    Parameters
    ----------
    _alt_summary: Alternatives simulation dataframe.
    _out_path: Expected output path.

    Returns
    -------
    None, writes table to tex file.
    """
    df = _alt_summary.copy()
    df["mechanism"] = df["point_cloud"].map(lambda pc: MECHANISM_MAP.get(pc, ""))
    df = df[df["mechanism"] != ""].copy()

    df["stat"] = df["stat"].map(lambda x: STAT_NAMES.get(x, x))
    df["filtration"] = df["filtration"].map(lambda x: FILTRATION_NAMES.get(x, x))

    g = df.groupby(["mechanism", "filtration", "stat"], as_index=False)["rejection_rate"].mean()
    g["col"] = g["filtration"] + "-" + g["stat"]
    wide = g.pivot(index="mechanism", columns="col", values="rejection_rate").reset_index()

    # add "best" per mechanism
    pcols = [c for c in wide.columns if c != "mechanism"]
    wide["best_test"] = wide[pcols].idxmax(axis=1)
    wide["best_power"] = wide[pcols].max(axis=1)
    for c in ["VR-TP", "VR-MTE", "DTM-TP", "DTM-MTE"]:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide[["mechanism", "VR-TP", "VR-MTE",
                 "DTM-TP", "DTM-MTE",
                 "best_test", "best_power"]]
    for c in ["VR-TP", "VR-MTE", "DTM-TP", "DTM-MTE", "best_power"]:
        wide[c] = wide[c].map(fmt)

    caption = (
        "Mechanism map summary. Mean power (empirical rejection rate) averaged over "
        "datasets within each mechanism, over the (n,d,\\epsilon) grid, by test (filtration × statistic). "
        "The final columns report the best-performing test per mechanism."
    )
    label = "tab:mechanism_map"

    tex = to_latex_table(wide, caption=caption, label=label)
    with open(_out_path, "w", encoding="utf-8") as f:
        f.write(tex)

# powers
def make_power_table(_power, _out_path):
    """
    Builds power versus collapse intensity table.
    Parameters
    ----------
    _power: Pre-computed powers dataframe.
    _out_path: Expected output path.

    Returns
    -------
    None, writes table to tex file.
    """
    df = _power.copy()
    # average over (n,d) at each eps per dataset
    g = df.groupby(["point_cloud", "eps"], as_index=False)["power_primary"].mean()
    # pivot eps to columns
    wide = g.pivot(index="point_cloud", columns="eps", values="power_primary").reset_index()

    # format eps columns
    eps_cols = [c for c in wide.columns if c != "point_cloud"]
    eps_cols_sorted = sorted(eps_cols, key=float)
    wide = wide[["point_cloud"] + eps_cols_sorted]
    for c in eps_cols_sorted:
        wide[c] = wide[c].map(fmt)

    caption = (
        "Primary-test power versus collapse strength $\\epsilon$. "
        "Entries are mean power averaged over the (n,d) grid."
    )
    label = "tab:power_primary"

    tex = to_latex_table(wide, caption=caption, label=label)
    with open(_out_path, "w", encoding="utf-8") as f:
        f.write(tex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Results directory")
    ap.add_argument("--out", dest="out_dir", default="assets/tables", help="Output dir for .tex")
    args = ap.parse_args()

    null_summary = pd.read_csv(f"{args.inp}/null_summary.csv")
    alt_summary = pd.read_csv(f"{args.inp}/alternatives_summary.csv")
    power_primary = pd.read_csv(f"{args.inp}/power_primary.csv")

    make_calibration_table(null_summary, os.path.join(args.out_dir, "calibration_table.tex"))
    make_mechanism_map(alt_summary, os.path.join(args.out_dir, "mechanism_map_table.tex"))
    make_power_table(power_primary, os.path.join(args.out_dir, "power_table.tex"))
