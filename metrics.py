import numpy as np

from complex_persistence import compute_dtm_vr_diagrams, compute_vr_diagrams


def finite_lengths(_intervals):
    """
    Finds the finite length intervals from persistence diagram.

    :param _intervals: input birth/death intervals
    :return:
    """
    if _intervals.size == 0:
        return np.array([])
    births = _intervals[:, 0]
    deaths = _intervals[:, 1]
    finite = np.isfinite(deaths)
    return (deaths[finite] - births[finite]).clip(min=0.0)


def total_persistence(_diagrams_by_dim, _dims, _p=1):
    """
    Sum of lengths^p over selected homology dimensions.

    :param _diagrams_by_dim:
    :param _dims:
    :param _p:
    :return:
    """
    s = 0.0
    for d in _dims:
        l = finite_lengths(_diagrams_by_dim.get(d, np.empty((0, 2))))
        if l.size:
            s += np.sum(l ** _p)
    return float(s)


def tail_count(_diagrams_by_dim, _dims, _tau):
    """
    Count bars with length > tau for selected dims.

    :param _diagrams_by_dim:
    :param _dims:
    :param _tau:
    :return:
    """
    c = 0
    for d in _dims:
        l = finite_lengths(_diagrams_by_dim.get(d, np.empty((0, 2))))
        if l.size:
            c += int(np.sum(l > _tau))
    return int(c)


def choose_dtm_k(n, mass=0.1, min_k=5, max_k=None):
    k = int(np.ceil(mass * n))
    k = max(k, min_k)
    if max_k is not None:
        k = min(k, max_k)
    k = min(k, n - 1)      # DTMRips requires k <= n-1
    k = max(k, 1)          # and k >= 1
    return k


def compute_statistics(_x, _dims, _p, _tau, _knn_est):
    """
    Runs the two tests from a single persistence diagram.

    :param _x: Point cloud.
    :param _dims:
    :param _p:
    :param _tau:
    :param _knn_est:
    :return:
    """
    out = {}
    vr_res = compute_vr_diagrams(_x, _knn_est)
    out['vr'] = {'tail_count': tail_count(vr_res, _dims, _tau)}
    out['vr']['total_persistence'] = total_persistence(vr_res, _dims, _p=_p)

    dtm_k = choose_dtm_k(_x.shape[0], mass=0.1, min_k=5)
    dtm_max_f = 2.0 * _knn_est
    dtm_res = compute_dtm_vr_diagrams(_x, dtm_max_f, dtm_k)
    out['dtm'] = {'tail_count': tail_count(dtm_res, _dims, _tau)}
    out['dtm']['total_persistence'] = total_persistence(dtm_res, _dims, _p=_p)
    return out
