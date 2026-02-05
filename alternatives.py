import time
import numpy as np
import pandas as pd

from tqdm import tqdm

from point_clouds import generate_collapsed_linear, generate_collapsed_swiss, generate_collapsed_torus
from metrics import compute_statistics
from utils import knn_persistence_param_estimation


def critical_values(_generator, _n_pts, _dim,
                    _dims, _p, _tau,
                    _num_sim, _alpha, _seed):
    """

    :param _generator:
    :param _n_pts:
    :param _dim:
    :param _dims:
    :param _p:
    :param _tau:
    :param _num_sim:
    :param _alpha:
    :param _seed:
    :return:
    """
    r_num = np.random.default_rng(_seed)
    samples = {}
    for _ in tqdm(range(_num_sim)):
        start = time.time()
        s = int(r_num.integers(0, 2**20 - 1))
        x = _generator(_n_pts, _dim, _seed=s)
        if isinstance(x, tuple):
            x = x[0]
        cut = knn_persistence_param_estimation(x, 10, 0.95, 1.05, 'euclidean')
        _tau = 0.2 * cut
        stats = compute_statistics(x, _dims, _p, _tau, cut)
        for filtr, sd in stats.items():
            samples.setdefault(filtr, {})
            for stat_name, val in sd.items():
                samples[filtr].setdefault(stat_name, []).append(val)
            end = time.time() - start
            samples[filtr].setdefault("time", []).append(end)
            samples[filtr].setdefault("seed", []).append(int(s))
            samples[filtr].setdefault("n_pts", []).append(_n_pts)
            samples[filtr].setdefault("dim", []).append(_dim)

    crit = {}
    for filtr, stat_dict in samples.items():
        crit[filtr] = {}
        for stat_name, vals in stat_dict.items():
            arr = np.array(vals, dtype=float)
            crit[filtr][stat_name] = float(np.quantile(arr, _alpha))

    return crit


if __name__ == "__main__":
    np_list = [10, 50]
    dim_list = [5, 10]
    k_plane_benchmarks = []
    swiss_roll_benchmarks = []
    torus_benchmarks = []
    for n in np_list:
        for d in dim_list:
            print('Running simulations for {} points of dimension {}'.format(n, d))
            crit_kplane = critical_values(generate_collapsed_linear, n, d, [0, 1, 2],
                                         0.1, 0.1, 100, 0.05, 17)
            crit_kplane['point_cloud'] = 'kplane'
            k_plane_benchmarks.append(crit_kplane)
            crit_swiss = critical_values(generate_collapsed_swiss, n, d, [0, 1, 2],
                                          0.1, 0.1, 100, 0.05, 17)
            crit_swiss['point_cloud'] = 'swiss'
            swiss_roll_benchmarks.append(crit_swiss)
            crit_torus = critical_values(generate_collapsed_torus, n, d, [0, 1, 2],
                                         0.1, 0.1, 100, 0.05, 17)
            crit_torus['point_cloud'] = 'torus'
            torus_benchmarks.append(crit_torus)
    kplane_df = pd.json_normalize(k_plane_benchmarks, sep='_')
    swiss_df = pd.json_normalize(swiss_roll_benchmarks, sep='_')
    torus_df = pd.json_normalize(torus_benchmarks, sep='_')
    out = pd.concat([kplane_df, swiss_df, torus_df], axis=0)
    out.to_csv('alternatives.csv', index=False)
