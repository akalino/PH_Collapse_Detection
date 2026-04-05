import argparse
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import skdim

from config_utils import load_config

# Nulls
from point_clouds import (
    # nulls
    generate_gaussian, generate_noisy_sphere,
    generate_elliptical_gaussian, generate_uniform_ball,
    generate_xavier_normal, generate_layer_mixes,
    # alts
    generate_collapsed_linear, generate_spiked_gaussian,
    generate_collapsed_swiss, generate_collapsed_torus, generate_paraboloid_graph,
    generate_contaminated_kplane, generate_k_cube, generate_noisy_sphere
)

# -----------------------------
# Metrics helpers (additions)
# -----------------------------
def _safe_float(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
        return np.nan
    except Exception:
        return np.nan


def compute_cov_id_metrics(
    _x,
    _id_methods=("TwoNN",),
    _mle_k=20,
    _eps_eig=1e-12,
    _explained_var=0.90,
    _compute_entropy=True,
):
    """
    Compute covariance diagnostics + intrinsic dimension estimates.
    Returns a flat dict of scalars suitable for CSV rows.
    """
    x = np.asarray(_x, dtype=float)
    n, d = x.shape

    out = {"n_pts": int(n), "dim": int(d)}

    # --- Covariance features ---
    if n <= 1 or d == 0:
        out.update({
            "cov_trace": np.nan,
            "cov_logdet": np.nan,
            "cov_cond": np.nan,
            "cov_participation_ratio": np.nan,
            "cov_eig_min": np.nan,
            "cov_eig_max": np.nan,
            "cov_eig_mean": np.nan,
            "cov_eig_median": np.nan,
            "cov_eig_q10": np.nan,
            "cov_eig_q90": np.nan,
            "cov_explained_dim_90": np.nan,
            "cov_eig_entropy": np.nan,
        })
    else:
        xc = x - x.mean(axis=0, keepdims=True)
        c = (xc.T @ xc) / max(n, 1)  # 1/n covariance

        eigvals = np.linalg.eigvalsh(c)
        eigvals = np.clip(eigvals, 0.0, None)

        tr = float(np.sum(eigvals))
        logdet = float(np.sum(np.log(eigvals + _eps_eig)))

        emax = float(np.max(eigvals)) if eigvals.size else np.nan
        emin = float(np.min(eigvals)) if eigvals.size else np.nan
        cond = float((emax + _eps_eig) / (emin + _eps_eig)) if np.isfinite(emax) and np.isfinite(emin) else np.nan

        denom = float(np.sum(eigvals ** 2))
        pr = float((tr ** 2) / denom) if denom > 0 else np.nan

        # Explained-variance dimension at threshold (default 90%)
        if tr > 0:
            p = eigvals / tr
            cdf = np.cumsum(np.sort(p)[::-1])
            k90 = int(np.searchsorted(cdf, _explained_var) + 1)
        else:
            k90 = np.nan

        # Eigen-entropy (optional)
        if _compute_entropy and tr > 0:
            p = eigvals / tr
            p = p[p > 0]
            ent = float(-np.sum(p * np.log(p)))
        else:
            ent = np.nan

        out.update({
            "cov_trace": tr,
            "cov_logdet": logdet,
            "cov_cond": cond, # use me
            "cov_participation_ratio": pr, # use me
            "cov_eig_min": emin,
            "cov_eig_max": emax,
            "cov_eig_mean": float(np.mean(eigvals)),
            "cov_eig_median": float(np.median(eigvals)),
            "cov_eig_q10": float(np.quantile(eigvals, 0.10)),
            "cov_eig_q90": float(np.quantile(eigvals, 0.90)),
            "cov_explained_dim_90": k90, # use me
            "cov_eig_entropy": ent,
        })

    # --- Intrinsic dimension (skdim) ---
    out["id_twonN"] = np.nan
    out["id_mle"] = np.nan
    out["id_mle_k"] = np.nan

    if n >= 5 and d >= 1:
        if "TwoNN" in _id_methods:
            est = skdim.id.TwoNN().fit(x)
            out["id_twonN"] = _safe_float(est.dimension_)

        if "MLE" in _id_methods:
            # NOTE: skdim MLE can be finicky; keep optional.
            k_eff = max(2, min(int(_mle_k), n - 1))
            try:
                est = skdim.id.MLE(K=k_eff).fit(x)  # FIXED: x not X
                out["id_mle"] = _safe_float(est.dimension_)
                out["id_mle_k"] = int(k_eff)
            except Exception:
                out["id_mle"] = np.nan
                out["id_mle_k"] = int(k_eff)

    return out


# -----------------------------
# Parallel task runner
# -----------------------------
_PC_GENERATORS = {
    "gaussian": lambda n, d, eps, seed: generate_gaussian(n, d, _seed=seed),
    "noisy_sphere": lambda n, d, eps, seed: generate_noisy_sphere(n, d, _seed=seed),
    "ell_gaussian": lambda n, d, eps, seed: generate_elliptical_gaussian(n, d, _seed=seed),
    "uniform_ball": lambda n, d, eps, seed: generate_uniform_ball(n, d, _seed=seed),
    "xavier": lambda n, d, eps, seed: generate_xavier_normal(n, d, _seed=seed),
    "layer_mix": lambda n, d, eps, seed: generate_layer_mixes(n, d, _seed=seed),
    "kplane": lambda n, d, eps, seed: generate_collapsed_linear(n, d, _eps=eps, _seed=seed),
    "spiked_gaussian": lambda n, d, eps, seed: generate_spiked_gaussian(n, d, _eps=eps, _seed=seed),
    "swiss": lambda n, d, eps, seed: generate_collapsed_swiss(n, d, _eps=eps, _seed=seed),
    "torus": lambda n, d, eps, seed: generate_collapsed_torus(n, d, _eps=eps, _seed=seed),
    "paraboloid_graph": lambda n, d, eps, seed: generate_paraboloid_graph(n, d, _eps=eps, _seed=seed),
    "contaminated_kplane": lambda n, d, eps, seed: generate_contaminated_kplane(n, d, _eps=eps, _seed=seed),
    "contaminated_sphere": lambda n, d, eps, seed: generate_noisy_sphere(n, d, _seed=seed),
    "contaminated_kcube": lambda n, d, eps, seed: generate_k_cube(n, d, _eps=eps, _seed=seed)
}


def _run_one(task):
    pc, n, d, eps, seed = task
    x = _PC_GENERATORS[pc](n, d, eps, seed)
    row = compute_cov_id_metrics(x, _id_methods=("TwoNN",), _compute_entropy=True)
    row["pc"] = pc
    row["eps"] = float(eps)
    row["seed"] = int(seed)
    return row


def build_cov_id_metrics_csv(
    out_csv="cov_id/cov_id_metrics.csv",
    n_list=(10, 50),
    d_list=(5, 10, 20),
    eps_list=(0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0),
    seeds=(17,),  # expand to multiple seeds for mean±std
    max_workers=None,
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    tasks = []
    for n, d, e, seed in product(n_list, d_list, eps_list, seeds):
        # base nulls use eps=0 for bookkeeping
        tasks.append(("gaussian", n, d, e, seed))
        # tasks.append(("noisy_sphere", n, d, e, seed))
        # alternatives sweep eps grid
        for pc in [
            "contaminated_kcube", "contaminated_sphere", "contaminated_kplane",
            "kplane", "spiked_gaussian",
            "torus", "swiss", "paraboloid_graph"
        ]:
                tasks.append((pc, n, d, e, seed))

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_one, t) for t in tasks]
        for fut in as_completed(futures):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    shared = cfg["shared"]
    null_stage = cfg["null_parallel"]
    alt_stage = cfg["alt_parallel"]
    run = cfg["run"]
    
    build_cov_id_metrics_csv(
        out_csv="cov_id/cov_id_metrics.csv",
        n_list=shared["n_list"],
        d_list=shared["d_list"],
        eps_list=shared["eps_list"],
        seeds=(run["base_seed"],),      # change to e.g. (11, 17, 23, 29, 31) when ready
        max_workers=run["max_workers"], # set to os.cpu_count() if you want explicit
    )
