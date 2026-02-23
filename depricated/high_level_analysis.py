import zipfile
import pandas as pd

ZIP_PATH = "results.zip"  # path to your zip
ALT_TESTED_CSV = "results/comparisons/alternatives_tested.csv"  # adjust if your zip paths differ

MECHANISM_MAP = {
    "contaminated_kcube": "C",
    "contaminated_kplane": "C",
    "contaminated_sphere": "C",
    "kplane": "A",
    "spiked_gaussian": "A",
    "torus": "B",
    "swiss": "B",
    "paraboloid": "B"
}

# Per-test columns in alternatives_tested.csv:
# filtration: 'vr'/'dtm'
# stat: 'total_persistence'/'tail_count'  (tail_count is your mean tail excess in code)
# reject: 0/1
# eps: epsilon level
# point_cloud: which alternative generator
# n_pts, dim: grid cell
# (seed, set, etc. may exist)

def load_alt_tested(zip_path=ZIP_PATH, inner_csv=ALT_TESTED_CSV) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(inner_csv) as f:
            df = pd.read_csv(f)
    return df

def per_cloud_per_test_table(df: pd.DataFrame, eps_values=(0.05, 0.20)) -> pd.DataFrame:
    # keep only alternative set (if present)
    if "set" in df.columns:
        df = df[df["set"].astype(str).str.contains("alt", case=False, na=False)].copy()

    # standardize epsilon col name
    eps_col = "eps" if "eps" in df.columns else ("epsilon" if "epsilon" in df.columns else None)
    if eps_col is None:
        raise ValueError("Could not find epsilon column ('eps' or 'epsilon') in alternatives_tested.csv")

    # map to nice labels
    fil_map = {"vr": "VR", "dtm": "DTM"}
    stat_map = {
        "total_persistence": "TP",
        "tail_count": "ME",  # tail_count key but it is mean tail excess in your current metrics
    }

    df = df[df[eps_col].isin(eps_values)].copy()
    df["Filtration"] = df["filtration"].map(lambda x: fil_map.get(str(x).lower(), str(x)))
    df["Stat"] = df["stat"].map(lambda x: stat_map.get(str(x), str(x)))

    # Per-test power = mean(reject) for that (point_cloud, eps, filtration, stat), averaged over n,d and seeds
    grp = (
        df.groupby(["point_cloud", eps_col, "Filtration", "Stat"], dropna=False)["reject"]
          .mean()
          .reset_index()
          .rename(columns={"reject": "power"})
    )

    # Pivot to columns: VR TP, VR ME, DTM TP, DTM ME
    grp["col"] = grp["Filtration"] + " " + grp["Stat"]
    wide = grp.pivot_table(index=["point_cloud", eps_col], columns="col", values="power")

    # Ensure consistent column order if present
    col_order = ["VR TP", "VR ME", "DTM TP", "DTM ME"]
    for c in col_order:
        if c not in wide.columns:
            wide[c] = pd.NA
    wide = wide[col_order].reset_index()

    wide["mechanism"] = wide["point_cloud"].apply(lambda x: MECHANISM_MAP.get(x, x))
    # Sort for readability
    wide = wide.sort_values(["mechanism", "point_cloud", eps_col]).reset_index(drop=True)

    return wide

def to_latex_table(df_wide: pd.DataFrame, eps_value: float, floatfmt="%.3f") -> str:
    eps_col = "eps" if "eps" in df_wide.columns else "epsilon"
    sub = df_wide[df_wide[eps_col] == eps_value].copy()

    # Pretty percent formatting (optional). Keep as decimals by default.
    # sub[col_order] = sub[col_order].astype(float).map(lambda x: floatfmt % x)

    sub = sub.drop(columns=[eps_col])
    return sub.to_latex(index=False, float_format=lambda x: floatfmt % x)

if __name__ == "__main__":
    df = load_alt_tested(ZIP_PATH, ALT_TESTED_CSV)
    wide = per_cloud_per_test_table(df, eps_values=(0.05, 0.10, 0.20, 0.50, 1.0, 2.0))

    print("Per-cloud per-test power (wide):")
    print(wide)
    wide.to_csv('plots/per_test_power.csv', index=False)

    print("\nLaTeX table for eps=0.05:")
    print(to_latex_table(wide, 0.05))

    print("\nLaTeX table for eps=0.20:")
    print(to_latex_table(wide, 0.20))
