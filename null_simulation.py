import numpy as np
import pandas as pd

from tqdm import tqdm

from point_clouds import generate_gaussian, generate_noisy_sphere
from utils import shared_simulation


if __name__ == "__main__":
    np_list = [10, 50, 100]
    dim_list = [5, 10, 20]
    gauss_benchmarks = []
    sphere_benchmarks = []
    for n in np_list:
        for d in dim_list:
            print('Running null calibration for {} points of dimension {}'.format(n, d))
            crit_gauss = shared_simulation(generate_gaussian, n, d, [0, 1, 2],
                                                   0.1, 0.1, 100, 0.05, 17)
            crit_gauss['point_cloud'] = 'gaussian'
            gauss_benchmarks.append(crit_gauss)
            crit_sphere = shared_simulation(generate_noisy_sphere, n, d, [0, 1, 2],
                                                    0.1, 0.1, 100, 0.05, 17)
            crit_sphere['point_cloud'] = 'noisy_sphere'
            sphere_benchmarks.append(crit_sphere)
    out = pd.concat(gauss_benchmarks + sphere_benchmarks, axis=0)
    out.to_csv('null.csv', index=False)
