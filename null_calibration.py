import time
import numpy as np
import pandas as pd

from tqdm import tqdm

from point_clouds import generate_gaussian, generate_noisy_sphere
from metrics import compute_statistics


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

def run_single_dataset(_pc_generator, _n_pts, _dim, _seed, **kwargs):
    _x = _pc_generator(_n_pts, _dim, _seed, **kwargs)
    return _x


def calibrate_critical_values(_generator, _n_pts, _dim,
                              _dims, _p, _tau,
                              _num_sim, _alpha, _seed):
    r_num = np.random.default_rng(_seed)
    samples = {}
    for _ in tqdm(range(_num_sim)):
        start = time.time()
        s = int(r_num.integers(0, 2**20 - 1))
        x = _generator(_n_pts, _dim, s)
        if isinstance(x, tuple):
            x = x[0]
        cut = knn_persistence_param_estimation(x, 10, 0.95, 1.05, 'euclidean')
        stats = compute_statistics(x, _dims, _p, _tau, cut)
        for filtr, sd in stats.items():
            samples.setdefault(filtr, {})
            for stat_name, val in sd.items():
                samples[filtr].setdefault(stat_name, []).append(val)
            end = time.time() - start
            #samples[filtr]['time'] = end
            #samples[filtr]['seed'] = int(s)
            samples[filtr].setdefault("seed", []).append(int(s))
            samples[filtr].setdefault("seed", []).append(int(s))


    crit = {}
    #crit["n_pts"] = _n_pts
    #crit["dim"] = _dim
    for filtr, stat_dict in samples.items():
        crit[filtr] = {}
        for stat_name, vals in stat_dict.items():
            arr = np.array(vals, dtype=float)
            crit[filtr][stat_name] = float(np.quantile(arr, 1 - _alpha))
    crit[filtr]['n_pts'] = _n_pts
    crit[filtr]['dim'] = _dim
    return crit


if __name__ == "__main__":
    np_list = [10, 50]
    dim_list = [5, 10]
    gauss_benchmarks = []
    sphere_benchmarks = []
    for n in np_list:
        for d in dim_list:
            print('Running null calibration for {} points of dimension {}'.format(n, d))
            crit_gauss = calibrate_critical_values(generate_gaussian, n, d, [0, 1, 2],
                                                   0.1, 0.1, 100, 0.05, 17)
            crit_gauss['point_cloud'] = 'gaussian'
            gauss_benchmarks.append(crit_gauss)
            crit_sphere = calibrate_critical_values(generate_noisy_sphere, n, d, [0, 1, 2],
                                                    0.1, 0.1, 100, 0.05, 17)
            crit_sphere['point_cloud'] = 'noisy_sphere'
            sphere_benchmarks.append(crit_sphere)
    gauss_df = pd.json_normalize(gauss_benchmarks, sep='_')
    sphere_df = pd.json_normalize(sphere_benchmarks, sep='_')
    out = pd.concat([gauss_df, sphere_df], axis=0)
    out.to_csv('null.csv', index=False)