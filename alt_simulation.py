import numpy as np
import pandas as pd

from tqdm import tqdm

from point_clouds import generate_collapsed_linear, generate_collapsed_swiss, generate_collapsed_torus
from utils import shared_simulation


if __name__ == "__main__":
    np_list = [10, 50, 100]
    dim_list = [5, 10, 20]
    eps = [0.05, 0.1, 0.2, 0.5]
    k_plane_benchmarks = []
    swiss_roll_benchmarks = []
    torus_benchmarks = []
    for n in np_list:
        for d in dim_list:
            for e in eps:
                print('Running simulations for {} points of dimension {} and eps {}'.format(n, d, e))
                crit_kplane = shared_simulation(generate_collapsed_linear, n, d, [0, 1, 2],
                                             0.1, 0.1, 100, e, 17)
                crit_kplane['point_cloud'] = 'kplane'
                k_plane_benchmarks.append(crit_kplane)
                crit_swiss = shared_simulation(generate_collapsed_swiss, n, d, [0, 1, 2],
                                              0.1, 0.1, 100, e, 17)
                crit_swiss['point_cloud'] = 'swiss'
                swiss_roll_benchmarks.append(crit_swiss)
                crit_torus = shared_simulation(generate_collapsed_torus, n, d, [0, 1, 2],
                                             0.1, 0.1, 100, e, 17)
                crit_torus['point_cloud'] = 'torus'
                torus_benchmarks.append(crit_torus)
    out = pd.concat(k_plane_benchmarks + swiss_roll_benchmarks + torus_benchmarks, axis=0)
    out.to_csv('alt_simulation.csv', index=False)
