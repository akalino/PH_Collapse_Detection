import numpy as np
import pandas as pd
import time

from tqdm import tqdm

from metrics import compute_statistics


def load_tau_map(path="calibration/tau_map.csv"):
    df = pd.read_csv(path)
    return df


def lookup_tau(_tau_df, _cloud, _n_pts, _dim, _filt):
    """
    :param _tau_df:
    :param _cloud:
    :param _n_pts:
    :param _dim:
    :param _filt:
    """
    sub = _tau_df[
        (_tau_df["point_cloud"] == _cloud) &
        (_tau_df["n_pts"] == _n_pts) &
        (_tau_df["dim"] == _dim) &
        (_tau_df["filtration"] == _filt)
    ]
    if sub.empty:
        raise KeyError(f"No tau for {_cloud=}, {_n_pts=}, {_dim=}, {_filt=}")
    return float(sub["tau"].iloc[0])


def shared_simulation(_generator, _n_pts, _dim,
                      _dims, _p, _tau,
                      _num_sim, _eps,
                      _landmark_m=None,
                      _tau_map=None,
                      _tau_cloud="gaussian",
                      _seed=None):
    r_num = np.random.default_rng(_seed)
    rows = []
    for _ in tqdm(range(_num_sim)):
        start = time.time()
        s = int(r_num.integers(0, 2 ** 20 - 1))
        try:
            x = _generator(_n_pts, _dim, _eps=_eps, _seed=s)
        except TypeError:
            x = _generator(_n_pts, _dim, _seed=s)
        if isinstance(x, tuple):
            x = x[0]

        cut = knn_persistence_param_estimation(x, 10, 0.95, 1.05, 'euclidean')
        if _tau_map is None:
            tau_vr = _tau * cut
            tau_dtm = _tau * (2.0 * cut)
        else:
            tau_vr = lookup_tau(_tau_map, _tau_cloud, _n_pts, _dim, "vr")
            tau_dtm = lookup_tau(_tau_map, _tau_cloud, _n_pts, _dim, "dtm")
        stats = compute_statistics(x, _dims, _p,
                                   {"vr": tau_vr, "dtm": tau_dtm}, cut)
        elapsed = time.time() - start

        for filtr, sd in stats.items():
            rows.append({
                "point_cloud": None,  # fill in caller
                "filtration": filtr,
                "n_pts": _n_pts,
                "dim": _dim,
                "eps": _eps,
                "seed": s,
                "time": elapsed,
                "cut": cut,
                "tau_vr": tau_vr,
                "tau_dtm": tau_dtm,
                **sd
            })

    return pd.DataFrame(rows)


def knn_persistence_param_estimation(_x, _k=10, _quantile=0.95, _scale=1.0, _metric='euclidean'):
    """

    :param _x: Point cloud.
    :param _k: k for nearest neighbors.
    :param _quantile: quantile cutoff.
    :param _scale: multiple of cutoff factor.
    :param _metric: needs to be euclidean.
    :return: data driven distance scale for max_edge_length and max_filtration.
    """
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
