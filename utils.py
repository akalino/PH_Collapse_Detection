import os
import time

import numpy as np
import pandas as pd

from metrics import compute_statistics


TAU_KEY_COLS = ["point_cloud", "filtration", "n_pts", "dim", "hom_dim"]


def empty_tau_map() -> pd.DataFrame:
    cols = TAU_KEY_COLS + ["tau_q", "tau", "calibration_run_id", "created_at"]
    return pd.DataFrame(columns=cols)


def load_tau_map(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return empty_tau_map()

    df = pd.read_csv(path)

    required = set(TAU_KEY_COLS + ["tau"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tau map at {path} missing required columns: {sorted(missing)}")

    return df


def required_tau_keys_from_config(cfg: dict, families: list[str] | None = None) -> pd.DataFrame:
    shared = cfg["shared"]
    stage = cfg["tau_parallel"]

    point_clouds = families if families is not None else stage["families"]

    rows = []
    for cloud in point_clouds:
        for n in shared["n_list"]:
            for d in shared["d_list"]:
                for hom_dim in shared["hom_dims"]:
                    for filtration in ("vr", "dtm"):
                        rows.append({
                            "point_cloud": cloud,
                            "filtration": filtration,
                            "n_pts": int(n),
                            "dim": int(d),
                            "hom_dim": int(hom_dim),
                        })

    out = pd.DataFrame(rows)
    if out.empty:
        return empty_tau_map()[TAU_KEY_COLS].copy()
    return out.drop_duplicates().reset_index(drop=True)


def find_missing_tau_keys(required_df: pd.DataFrame, tau_df: pd.DataFrame) -> pd.DataFrame:
    if tau_df.empty:
        return required_df.copy()

    existing = tau_df[TAU_KEY_COLS].drop_duplicates()
    merged = required_df.merge(existing, on=TAU_KEY_COLS, how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"][TAU_KEY_COLS].copy()
    return missing.reset_index(drop=True)


def append_to_master_tau_map(master_path: str, new_rows: pd.DataFrame) -> pd.DataFrame:
    if new_rows.empty:
        return load_tau_map(master_path)

    master_df = load_tau_map(master_path)
    combined = pd.concat([master_df, new_rows], axis=0, ignore_index=True)
    combined = combined.drop_duplicates(subset=TAU_KEY_COLS, keep="last")
    combined = combined.sort_values(TAU_KEY_COLS).reset_index(drop=True)

    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    combined.to_csv(master_path, index=False)
    return combined


def lookup_tau(_tau_df, _cloud, _n_pts, _dim, _filt, _hom_dim):
    sub = _tau_df[
        (_tau_df["point_cloud"] == _cloud) &
        (_tau_df["n_pts"] == _n_pts) &
        (_tau_df["dim"] == _dim) &
        (_tau_df["filtration"] == _filt) &
        (_tau_df["hom_dim"] == _hom_dim)
    ]
    if sub.empty:
        raise KeyError(
            f"No tau for cloud={_cloud}, n_pts={_n_pts}, dim={_dim}, "
            f"filtration={_filt}, hom_dim={_hom_dim}"
        )
    return float(sub["tau"].iloc[0])


def validate_required_tau_keys(cfg: dict, tau_df: pd.DataFrame, families: list[str]) -> None:
    required = required_tau_keys_from_config(cfg, families=families)
    missing = find_missing_tau_keys(required, tau_df)
    if not missing.empty:
        sample = missing.head(10).to_dict(orient="records")
        raise ValueError(
            f"Missing {len(missing)} tau entries. "
            f"Run tau extension first. Sample missing keys: {sample}"
        )


def shared_simulation(_generator, _n_pts, _dim,
                      _dims, _p, _tau,
                      _num_sim, _eps,
                      _landmark_m=None,
                      _tau_map=None,
                      _tau_cloud="gaussian",
                      _seed=None):
    r_num = np.random.default_rng(_seed)
    rows = []
    for _ in range(_num_sim):
        start = time.time()
        s = int(r_num.integers(0, 2 ** 20 - 1))
        try:
            x = _generator(_n_pts, _dim, _eps=_eps, _seed=s)
        except TypeError:
            x = _generator(_n_pts, _dim, _seed=s)
        if isinstance(x, tuple):
            x = x[0]

        cut = knn_persistence_param_estimation(x, 10, 0.95, 1.05, 'euclidean')

        tau_by_filtration_dim = {}
        if _tau_map is None:
            for hom_dim in _dims:
                tau_by_filtration_dim[("vr", hom_dim)] = _tau * cut
                tau_by_filtration_dim[("dtm", hom_dim)] = _tau * (2.0 * cut)
                tau_by_filtration_dim[("witness", hom_dim)] = _tau * (2.0 * cut)
        else:
            for hom_dim in _dims:
                tau_by_filtration_dim[("vr", hom_dim)] = lookup_tau(
                    _tau_map, _tau_cloud, _n_pts, _dim, "vr", hom_dim
                )
                tau_by_filtration_dim[("dtm", hom_dim)] = lookup_tau(
                    _tau_map, _tau_cloud, _n_pts, _dim, "dtm", hom_dim
                )
                tau_by_filtration_dim[("witness", hom_dim)] = lookup_tau(
                    _tau_map, _tau_cloud, _n_pts, _dim, "witness", hom_dim
                )

        tau_for_stats = {
            "vr": max(tau_by_filtration_dim[("vr", h)] for h in _dims),
            "dtm": max(tau_by_filtration_dim[("dtm", h)] for h in _dims),
        }

        stats = compute_statistics(x, _dims, _p, tau_for_stats, cut)
        elapsed = time.time() - start

        for filtr, sd in stats.items():
            row = {
                "point_cloud": None,
                "filtration": filtr,
                "n_pts": _n_pts,
                "dim": _dim,
                "eps": _eps,
                "seed": s,
                "time": elapsed,
                "cut": cut,
                **sd
            }
            for hom_dim in _dims:
                row[f"tau_{filtr}_h{hom_dim}"] = tau_by_filtration_dim[(filtr, hom_dim)]
            rows.append(row)

    return pd.DataFrame(rows)


def knn_persistence_param_estimation(_x, _k=10, _quantile=0.95, _scale=1.0, _metric='euclidean'):
    x = np.array(_x, dtype=float)
    n = x.shape[0]

    k_eff = int(_k)
    if k_eff >= n:
        k_eff = n - 1
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))

    np.fill_diagonal(d, np.inf)
    kth = np.partition(d, k_eff - 1, axis=1)[:, k_eff - 1]

    cutoff = float(np.quantile(kth, _quantile) * _scale)
    if not np.isfinite(cutoff) or cutoff < 0:
        cutoff = 0.0
    return cutoff