import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from point_clouds import (generate_gaussian, generate_noisy_sphere,
                          generate_elliptical_gaussian, generate_uniform_ball,
                          generate_xavier_normal, generate_layer_mixes)
from utils import load_tau_map, shared_simulation

GENS = {
    "gaussian": generate_gaussian,
    "noisy_sphere": generate_noisy_sphere,
    "ell_gaussian": generate_elliptical_gaussian,
    "uniform_ball": generate_uniform_ball,
    "xavier": generate_xavier_normal,
    "layer_mix": generate_layer_mixes
}


def run_one(_task):
    n, d, e, name = _task
    tau_df = load_tau_map()
    lm = False
    landmark = n / 2 if lm else None

    seed = abs(hash((n, d, e, name))) % (2**31 - 1)

    _df = shared_simulation(GENS[name], n, d, [0,1,2],
                           0.1, 0.1, 100,
                            e, landmark, tau_df, "gaussian", seed)
    _df['point_cloud'] = name
    return _df


if __name__ == '__main__':
    np_list = [10, 50]
    dim_list = [5, 10, 20]
    eps = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    names = list(GENS.keys())
    tasks = [(n, d, e, name) for n in np_list for d in dim_list for e in eps for name in names]

    out_dfs = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_one, t) for t in tasks]
        for f in as_completed(futures):
            try:
                out_dfs.append(f.result())
            except Exception as ex:
                print("Task failed:", ex)

    out = pd.concat(out_dfs, axis=0)
    out.to_csv("simulations/null_simulation.csv", index=False)
