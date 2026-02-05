import numpy as np


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
