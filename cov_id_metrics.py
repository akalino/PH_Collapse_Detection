# cov_id_metrics.py
import copy

import numpy as np
import pandas as pd
import skdim

from point_clouds import generate_gaussian, generate_noisy_sphere, generate_collapsed_linear, \
    generate_contaminated_kplane, generate_k_cube, generate_spiked_gaussian, generate_paraboloid_graph, \
    generate_collapsed_torus, generate_collapsed_swiss


def _safe_float(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
        return np.nan
    except Exception:
        return np.nan


# TODO: add MLE back to _id_methods later
def compute_cov_id_metrics(_x, _id_methods=("TwoNN"),
                           _mle_k=20, _eps_eig=1e-12):
    """
    Compute covariance-based collapse diagnostics + intrinsic dimension estimates.

    Returns a flat dict of scalars suitable for CSV rows:
      - cov_trace, cov_logdet, cov_cond, cov_participation_ratio, cov_eig_* summaries
      - id_twonN, id_mle (if skdim installed; else NaN)

    Parameters
    ----------
    X : (n, d) array
        Point cloud.
    id_methods : tuple[str]
        Any of ("TwoNN", "MLE"). Only used if skdim is installed.
    mle_k : int
        Neighborhood parameter for skdim.id.MLE (bounded to [2, n-1]).
    eps_eig : float
        Stabilizer for logdet and condition number when eigenvalues are tiny.
    """
    x = np.asarray(_x, dtype=float)
    n, d = x.shape

    out = {
        "n_pts": int(n),
        "dim": int(d),
    }

    # --- Covariance features ---
    if n <= 1 or d == 0:
        # Degenerate
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
        })
    else:
        xc = x - x.mean(axis=0, keepdims=True)
        # population covariance (1/n) is fine for diagnostics; avoids n-1 when n small
        c = (xc.T @ xc) / max(n, 1)

        # symmetric eigvals
        eigvals = np.linalg.eigvalsh(c)
        eigvals = np.clip(eigvals, 0.0, None)

        tr = float(np.sum(eigvals))
        # logdet stabilized (for near-singular)
        logdet = float(np.sum(np.log(eigvals + _eps_eig)))

        # condition number stabilized
        emax = float(np.max(eigvals)) if eigvals.size else np.nan
        emin = float(np.min(eigvals)) if eigvals.size else np.nan
        cond = float((emax + _eps_eig) / (emin + _eps_eig)) if np.isfinite(emax) and np.isfinite(emin) else np.nan

        # participation ratio: (sum λ)^2 / sum λ^2
        denom = float(np.sum(eigvals ** 2))
        pr = float((tr ** 2) / denom) if denom > 0 else np.nan

        out.update({
            "cov_trace": tr,
            "cov_logdet": logdet,
            "cov_cond": cond,
            "cov_participation_ratio": pr,
            "cov_eig_min": emin,
            "cov_eig_max": emax,
            "cov_eig_mean": float(np.mean(eigvals)),
            "cov_eig_median": float(np.median(eigvals)),
            "cov_eig_q10": float(np.quantile(eigvals, 0.10)),
            "cov_eig_q90": float(np.quantile(eigvals, 0.90)),
        })

    # --- Intrinsic dimension (skdim) ---
    out["id_twonN"] = np.nan
    out["id_mle"] = np.nan

    if n >= 5 and d >= 1:
        # TwoNN
        if "TwoNN" in _id_methods:
            est = skdim.id.TwoNN().fit(x)
            out["id_twonN"] = _safe_float(est.dimension_)

        # MLE (Levina–Bickel style)
        # TODO: this is broken in skdim
        if "MLE" in _id_methods:
            k_eff = int(_mle_k)
            k_eff = max(2, k_eff)
            k_eff = min(k_eff, n - 1)
            est = skdim.id.MLE(K=k_eff).fit(X)
            out["id_mle"] = _safe_float(est.dimension_)
            out["id_mle_k"] = int(k_eff)

    return out


if __name__ == "__main__":
    np_list = [10, 50, 100]
    dim_list = [5, 10, 20]
    eps = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    gauss = []
    noisy_sphere = []
    k_plane = []
    contaminated_plane = []
    k_cube = []
    spike_gauss = []
    parab = []
    torus = []
    swiss = []
    for n in np_list:
        for d in dim_list:
            gauss.append(compute_cov_id_metrics(generate_gaussian(n, d, _seed=17)))
            noisy_sphere.append(compute_cov_id_metrics(generate_noisy_sphere(n, d, _seed=17)))
            for e in eps:
                cl =compute_cov_id_metrics(generate_collapsed_linear(n, d, _eps=e, _seed=17))
                cl['eps'] = e
                k_plane.append(cl)

                cp = compute_cov_id_metrics(generate_contaminated_kplane(n, d, _eps=e, _seed=17))
                cp['eps'] = e
                contaminated_plane.append(cp)

                kc = compute_cov_id_metrics(generate_k_cube(n, d, _eps=e, _k=d, _seed=17))
                kc['eps'] = e
                k_cube.append(kc)

                sg = compute_cov_id_metrics(generate_spiked_gaussian(n, d, _eps=e, _seed=17))
                sg['eps'] = e
                spike_gauss.append(sg)

                pg = compute_cov_id_metrics(generate_paraboloid_graph(n, d, _eps=e, _seed=17))
                pg['eps'] = e
                parab.append(pg)

                ct = compute_cov_id_metrics(generate_collapsed_torus(n, d, _eps=e, _seed=17))
                ct['eps'] = e
                torus.append(ct)

                sr = compute_cov_id_metrics(generate_collapsed_swiss(n, d, _eps=e, _seed=17))
                sr['eps'] = e
                swiss.append(sr)

    gauss_metrics = pd.json_normalize(gauss, sep="_")
    gauss_metrics['pc'] = 'base_gauss'
    gauss_metrics['eps'] = 0

    sphere_metrics = pd.json_normalize(noisy_sphere, sep="_")
    sphere_metrics['pc'] = 'base_noisy_sphere'
    sphere_metrics['eps'] = 0

    col_linear = pd.json_normalize(k_plane, sep="_")
    col_linear['pc'] = 'alt_k_plane'

    contam = pd.json_normalize(contaminated_plane, sep="_")
    contam['pc'] = 'contaminated_k_plane'

    kcube_df = pd.json_normalize(k_cube, sep="_")
    kcube_df['pc'] = 'alt_k_cube'

    spike_df = pd.json_normalize(spike_gauss, sep="_")
    spike_df['pc'] = 'alt_spike_gauss'

    parab_df = pd.json_normalize(parab, sep="_")
    parab_df['pc'] = 'alt_parab'

    torus_df = pd.json_normalize(torus, sep="_")
    torus_df['pc'] = 'alt_torus'

    swiss_df = pd.json_normalize(swiss, sep="_")
    swiss_df['pc'] = 'alt_swiss'

    out = pd.concat([gauss_metrics, sphere_metrics,
                     col_linear, contam,
                     kcube_df, spike_df,
                     parab_df, torus_df, swiss_df])
    out = out.drop('id_mle', axis=1)
    out.to_csv('cov_id/cov_id_metrics.csv', index=False)
