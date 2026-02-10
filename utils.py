import numpy as np
import pandas as pd
import time

from tqdm import tqdm

from metrics import compute_statistics


def shared_simulation(_generator, _n_pts, _dim,
                      _dims, _p, _tau,
                      _num_sim, _eps,
                      _landmark_m=None, _seed=None):
    r_num = np.random.default_rng(_seed)
    rows = []
    for _ in tqdm(range(_num_sim)):
        start = time.time()
        s = int(r_num.integers(0, 2 ** 20 - 1))
        try:
            x = _generator(_n_pts, _dim, _eps=_eps, _seed=s)
        except TypeError:
            # for generators like generate_gaussian / generate_noisy_sphere that don't accept _eps
            x = _generator(_n_pts, _dim, _seed=s)
        # x = _generator(_n_pts, _dim, _eps=0.1, _seed=s)
        if isinstance(x, tuple):
            x = x[0]

        cut = knn_persistence_param_estimation(x, 10, 0.95, 1.05, 'euclidean')
        tau_local = 0.2 * cut
        stats = compute_statistics(x, _dims, _p, tau_local, cut)
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
                "tau": tau_local,
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
