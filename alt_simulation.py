import numpy as np
import pandas as pd

from tqdm import tqdm

from point_clouds import generate_collapsed_linear, generate_collapsed_swiss, generate_collapsed_torus, \
    generate_spiked_gaussian, generate_contaminated_kplane, generate_paraboloid_graph
from utils import shared_simulation


if __name__ == "__main__":
    np_list = [10, 50]
    dim_list = [5, 10, 20]
    eps = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    k_plane_benchmarks = []
    swiss_roll_benchmarks = []
    torus_benchmarks = []
    spiked_benchmark = []
    contaminated_kplane_benchmark = []
    paraboloid_graph_benchmark = []
    lm = False
    for n in np_list:
        for d in dim_list:
            for e in eps:
                print('Running simulations for {} points of dimension {} and eps {}'.format(n, d, e))
                landmark = n / 2 if lm else None
                crit_kplane = shared_simulation(generate_collapsed_linear, n, d, [0, 1, 2],
                                             0.1, 0.1, 100, e, landmark, 17)
                crit_kplane['point_cloud'] = 'kplane'
                k_plane_benchmarks.append(crit_kplane)
                crit_swiss = shared_simulation(generate_collapsed_swiss, n, d, [0, 1, 2],
                                              0.1, 0.1, 100, e, landmark, 17)
                crit_swiss['point_cloud'] = 'swiss'
                swiss_roll_benchmarks.append(crit_swiss)
                crit_torus = shared_simulation(generate_collapsed_torus, n, d, [0, 1, 2],
                                             0.1, 0.1, 100, e, landmark, 17)
                crit_torus['point_cloud'] = 'torus'
                torus_benchmarks.append(crit_torus)
                crit_spiked_gaussian = shared_simulation(generate_spiked_gaussian, n, d, [0, 1, 2],
                                                         0.1, 0.1,100, e, landmark, 17)
                crit_spiked_gaussian['point_cloud'] = 'spiked_gaussian'
                spiked_benchmark.append(crit_spiked_gaussian)
                crit_contaminated_kplane = shared_simulation(generate_contaminated_kplane,n, d, [0, 1, 2],
                                                             0.1, 0.1,100, e, landmark, 17)
                crit_contaminated_kplane['point_cloud'] = 'contaminated_kplane'
                contaminated_kplane_benchmark.append(crit_contaminated_kplane)
                crit_paraboloid = shared_simulation(generate_paraboloid_graph, n, d, [0, 1, 2],
                                                    0.1, 0.1, 100, e, landmark, 17)
                crit_paraboloid['point_cloud'] = 'paraboloid_graph'
                paraboloid_graph_benchmark.append(crit_paraboloid)
    out = pd.concat(k_plane_benchmarks +
                    swiss_roll_benchmarks +
                    torus_benchmarks +
                    spiked_benchmark +
                    contaminated_kplane_benchmark +
                    paraboloid_graph_benchmark, axis=0)
    out.to_csv('simulations/alt_simulation.csv', index=False)
