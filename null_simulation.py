import numpy as np
import pandas as pd

from tqdm import tqdm

from point_clouds import (generate_gaussian, generate_noisy_sphere,
                          generate_elliptical_gaussian, generate_uniform_ball)
from utils import shared_simulation, load_tau_map


def _wrap_elliptical(_eta, _k=None, _rotate=True):
    def _gen(_n, _d, _seed=17):
        kk = _k if _k is not None else max(1, min(3, _d))
        return generate_elliptical_gaussian(_n, _d, _k=kk,
                                            _eta=_eta, _scale=1.0, _seed=_seed, _rotate=_rotate)
    return _gen


def _wrap_ball(_radius=1.0, _noise_sig=0.0):
    def _gen(_n, _d, _seed=17):
        return generate_uniform_ball(_n, _d, _radius=_radius,
                                     _noise_sig=_noise_sig, _seed=_seed)
    return _gen


if __name__ == "__main__":
    np_list = [10, 50, 100]
    dim_list = [5, 10, 20]
    eps = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    eta_list = [1.0, 0.5, 0.2, 0.1, 0.05]
    gauss_benchmarks = []
    sphere_benchmarks = []
    ball_benchmark = []
    ell_benchmark = []
    lm = False
    tau_df = load_tau_map()
    for n in np_list:
        for d in dim_list:
            for e in eps:
                landmark = n / 2 if lm else None
                print('Running null calibration for {} points of dimension {}'.format(n, d))

                crit_gauss = shared_simulation(generate_gaussian,
                                               n, d, [0, 1, 2, 3],
                                               0.1, 0.1, 100, e, landmark,
                                               tau_df, "gaussian", 17)
                crit_gauss['point_cloud'] = 'gaussian'
                gauss_benchmarks.append(crit_gauss)

                crit_sphere = shared_simulation(generate_noisy_sphere, n, d, [0, 1, 2, 3],
                                                        0.1, 0.1,
                                                100, e, landmark,
                                                tau_df, "gaussian",17)
                crit_sphere['point_cloud'] = 'noisy_sphere'
                sphere_benchmarks.append(crit_sphere)

                crit_ball = shared_simulation(_wrap_ball(_radius=1.0, _noise_sig=0.0),
                                              n, d, [0, 1, 2, 3], 0.1, 0.1, 100,
                                              e, landmark,tau_df, "gaussian", 17)
                crit_ball['point_cloud'] = 'uniform_ball'
                ball_benchmark.append(crit_ball)

                for eta in eta_list:
                    crit_ell = shared_simulation(_wrap_elliptical(_eta=eta, _k=min(3, d), _rotate=True),
                                                 n, d, [0, 1, 2, 3], 0.1, 0.1, 100,
                                                 e, landmark,tau_df, "gaussian", 17)
                    crit_ell["point_cloud"] = "elliptical_{}".format(eta)
                    ell_benchmark.append(crit_ell)
    out = pd.concat(gauss_benchmarks + sphere_benchmarks +
                    ball_benchmark + ell_benchmark, axis=0)
    out.to_csv('simulations/null_simulation.csv', index=False)
