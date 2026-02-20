import numpy as np

from complex_persistence import compute_dtm_vr_diagrams, compute_vr_diagrams  # , compute_witness_diagrams


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


def concat_lengths_by_dim(_diagrams_by_dim, _dims):
    """
    :param _diagrams_by_dim:
    :param _dims:
    """
    all_len = []
    for d in _dims:
        ints = _diagrams_by_dim.get(d, np.empty((0, 2)))
        l = finite_lengths(ints)
        if l.size:
            all_len.append(l)
    return np.concatenate(all_len) if all_len else np.array([])


def compute_lengths(_x, _dims, _knn_est, _sparse=None):
    max_dim = int(max(_dims)) if len(_dims) else 0
    vr_res = compute_vr_diagrams(_x, _knn_est, _max_dim=max_dim, _sparse=_sparse)

    dtm_k = choose_dtm_k(_x.shape[0], mass=0.1, min_k=5)
    dtm_max_f = 2.0 * _knn_est
    dtm_res = compute_dtm_vr_diagrams(_x, dtm_max_f, dtm_k, _max_dim=max_dim)

    return {
        "vr": concat_lengths_by_dim(vr_res, _dims),
        "dtm": concat_lengths_by_dim(dtm_res, _dims),
    }


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


def tail_rate(_diagrams_by_dim, _dims, _tau):
    num = 0
    den = 0
    for d in _dims:
        l = finite_lengths(_diagrams_by_dim.get(d, np.empty((0, 2))))
        if l.size:
            den += int(l.size)
            num += int(np.sum(l > _tau))
    return float(num / den) if den > 0 else 0.0


def tail_excess(_diagrams_by_dim, _dims, _tau):
    s = 0.0
    for d in _dims:
        l = finite_lengths(_diagrams_by_dim.get(d, np.empty((0, 2))))
        if l.size:
            s += float(np.sum(np.maximum(l - _tau, 0.0)))
    return float(s)


def tail_mean_excess(_diagrams_by_dim, _dims, _tau, _eps=1e-12):
    """
    mean_excess = sum_i (l_i -tau) +/ ( #{i: l_i > tau} + eps )

    :param _diagrams_by_dim
    :param _dims
    :param _tau
    :param _eps
    """
    num_excess = 0
    sum_excess = 0.0

    for d in _dims:
        l = finite_lengths(_diagrams_by_dim.get(d, np.empty((0, 2))))
        if not l.size:
            continue
        excess = l - _tau
        mask = excess > 0.0
        if np.any(mask):
            num_excess += int(np.sum(mask))
            sum_excess += float(np.sum(excess[mask]))
    return float(sum_excess / (num_excess + _eps))


import numpy as np


def betti_curve(_diagrams_by_dim, _dims, _r_grid):
    """
    Compute Betti curve beta(r) pooled across selected homology dimensions.

    beta(r) = #{(b,d): b <= r < d} over finite bars, summed over dims in _dims.

    :param _diagrams_by_dim:
    :param _dims:
    :param _r_grid:
    :return: np.ndarray shape (len(_r_grid),)
    """
    r = np.asarray(_r_grid, dtype=float)
    if r.ndim != 1:
        raise ValueError("r_grid must be 1D")
    if r.size == 0:
        return np.zeros((0,), dtype=float)

    beta = np.zeros(r.shape, dtype=float)

    for d in _dims:
        ints = _diagrams_by_dim.get(d, np.empty((0, 2)))
        if ints.size == 0:
            continue
        births = ints[:, 0]
        deaths = ints[:, 1]
        finite = np.isfinite(deaths)
        if not np.any(finite):
            continue
        b = births[finite]
        e = deaths[finite]
        if b.size == 0:
            continue
        active = (b[:, None] <= r[None, :]) & (r[None, :] < e[:, None])
        beta += np.sum(active, axis=0).astype(float)

    return beta


def betti_curve_auc(_diagrams_by_dim, _dims, _r_grid):
    """
    Area under Betti curve (AUC) using trapezoid rule, pooled across dims.

    :param _diagrams_by_dim:
    :param _dims:
    :param _r_grid:
    :return: float
    """
    r = np.asarray(_r_grid, dtype=float)
    if r.size < 2:
        return 0.0
    beta = betti_curve(_diagrams_by_dim, _dims, r)
    order = np.argsort(r)
    r_sorted = r[order]
    beta_sorted = beta[order]
    return float(np.trapz(beta_sorted, r_sorted))


def betti_curve_peak(_diagrams_by_dim, _dims, _r_grid):
    """
    Peak (max) of Betti curve, pooled across dims.

    :param _diagrams_by_dim:
    :param _dims:
    :param _r_grid:
    :return: float
    """
    r = np.asarray(_r_grid, dtype=float)
    if r.size == 0:
        return 0.0
    beta = betti_curve(_diagrams_by_dim, _dims, r)
    return float(np.max(beta)) if beta.size else 0.0


def betti_curve_collapse(_diagrams_by_dim, _dims, _r_grid):
    """
    Minimal "Betti curve collapse" feature set:
      - auc: area under beta(r)
      - peak: max beta(r)

    :param _diagrams_by_dim:
    :param _dims:
    :param _r_grid:
    :return: dict
    """
    return {
        "betti_auc": betti_curve_auc(_diagrams_by_dim, _dims, _r_grid),
        "betti_peak": betti_curve_peak(_diagrams_by_dim, _dims, _r_grid),
    }


def choose_dtm_k(n, mass=0.1, min_k=5, max_k=None):
    k = int(np.ceil(mass * n))
    k = max(k, min_k)
    if max_k is not None:
        k = min(k, max_k)
    k = min(k, n - 1)      # DTMRips requires k <= n-1
    k = max(k, 1)          # and k >= 1
    return k


def pick_landmarks(_x, _m, _seed=None):
    """
    witness complex for scaling.

    :param _x: Point cloud.
    :param _m: Number of landmarks.
    :param _seed: Random seed.
    :return: Sub-sampled point cloud.
    """
    n = _x.shape[0]
    if _m is None or _m <= 0 or _m >= n:
        return _x
    rng = np.random.default_rng(_seed)
    idx = rng.choice(n, _m, replace=False)
    return _x[idx]


def compute_statistics(_x, _dims, _p, _tau, _knn_est,
                       _landmark_m=None, _landmark_seed=None):
    """
    Runs the two tests from a single persistence diagram.

    :param _x: Point cloud.
    :param _dims:
    :param _p:
    :param _tau:
    :param _knn_est:
    :param _landmark_m:
    :param _landmark_seed:
    :return:
    """
    # use_witness = False
    # _x = pick_landmarks(_x, _landmark_m, _landmark_seed)
    out = {}
    #if use_witness:
    #    max_alpha_sq = _knn_est**2
    #    dgms = compute_witness_diagrams(_x, _landmark_m, max_alpha_sq, max(_dims), _landmark_seed)
    #    out["witness"] = {
    #        "total_persistence": total_persistence(dgms, _dims, _p),
    #        "tail_count": tail_count(dgms, _dims, _tau)
    #    }

    tau_vr = _tau if not isinstance(_tau, dict) else _tau["vr"]
    tau_dtm = _tau if not isinstance(_tau, dict) else _tau["dtm"]

    vr_res = compute_vr_diagrams(_x, _knn_est)
    out['vr'] = {'tail_count': tail_mean_excess(vr_res, _dims, tau_vr)}
    out['vr']['total_persistence'] = total_persistence(vr_res, _dims, _p=_p)

    dtm_k = choose_dtm_k(_x.shape[0], mass=0.1, min_k=5)
    dtm_max_f = 2.0 * _knn_est
    dtm_res = compute_dtm_vr_diagrams(_x, dtm_max_f, dtm_k)
    out['dtm'] = {'tail_count': tail_mean_excess(dtm_res, _dims, tau_dtm)}
    out['dtm']['total_persistence'] = total_persistence(dtm_res, _dims, _p=_p)
    return out
