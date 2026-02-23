"""
Generates the initial results plots.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAT_NAMES = {
    "total_persistence": "TP",
    "mean_excess_tail": "MTE",
    "mean_tail_excess": "MTE",
    "tail_count": "MTE",
}

FILTRATION_NAMES = {"vr": "VR", "dtm": "DTM"}


def make_power_plots(_args):
    """
    Plots power vs. collapse strength for all alternatives.
    Parameters
    ----------
    _args: args passed from command line.

    Returns
    -------
    None, plots saved to results.
    """
    df = pd.read_csv(f"{_args.inp}/alternatives_summary.csv")
    df["stat"] = df["stat"].map(lambda x: STAT_NAMES.get(x, x))
    df["filtration"] = df["filtration"].map(lambda x: FILTRATION_NAMES.get(x, x))

    for pc in _args.datasets:
        sub = df[df['point_cloud'] == pc].copy()
        g = (
            sub.groupby(["eps", "filtration", "stat"], as_index=False)["rejection_rate"]
            .mean()
            .sort_values("eps")
        )

        fig = plt.figure()
        ax = plt.gca()

        for (filt, stat), gg in g.groupby(["filtration", "stat"]):
            ax.plot(gg["eps"].to_numpy(), gg["rejection_rate"].to_numpy(), marker="o", label=f"{filt}-{stat}")

        ax.set_xlabel("epsilon")
        ax.set_ylabel("power (rejection rate)")
        ax.set_title(f"Power vs epsilon: {pc}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(_args.out, f"power_vs_eps_{pc}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def make_baseline_plots(_args):
    """
    Plots power vs. collapse strength for all alternatives.
    Parameters
    ----------
    _args: args passed from command line.

    Returns
    -------
    None, plots saved to results.
    """
    df = pd.read_csv(f"{_args.inp}/cov_id_metrics.csv")

    metrics = [
        ("cov_participation_ratio", "Participation Ratio"),
        ("id_twonN", "TwoNN ID"),
        ("cov_cond", "Cov condition number"),
    ]
    for pc in _args.datasets:
        sub = df[df["pc"] == pc].copy()
        if sub.empty:
            print(f"[skip] no rows for pc={pc}")
            continue

        # average over (n,d) for each eps
        g = sub.groupby(["eps"], as_index=False).mean(numeric_only=True).sort_values("eps")

        for col, title in metrics:
            if col not in g.columns:
                continue

            fig = plt.figure()
            ax = plt.gca()
            ax.plot(g["eps"].to_numpy(), g[col].to_numpy(), marker="o")
            ax.set_xlabel("epsilon")
            ax.set_ylabel(title)
            ax.set_title(f"{title} vs epsilon: {pc}")
            fig.tight_layout()

            out_path = os.path.join(_args.out, f"baseline_{col}_{pc}.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Results directory")
    ap.add_argument("--out", default="assets/figures", help="Output dir for PNGs")
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=["contaminated_kcube", "contaminated_sphere", "contaminated_kplane",
                 "kplane", "spiked_gaussian",
                 "torus", "swiss", "paraboloid", "paraboloid_graph"],
        help="Datasets to plot (point_cloud names)",
    )
    args = ap.parse_args()
    make_power_plots(args)
    make_baseline_plots(args)
