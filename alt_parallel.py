import argparse
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from config_utils import load_config, resolve_output
from point_clouds import (
    generate_collapsed_linear, generate_spiked_gaussian,
    generate_collapsed_swiss, generate_collapsed_torus, generate_paraboloid_graph,
    generate_contaminated_kplane, generate_k_cube, generate_noisy_sphere
)
from utils import load_tau_map, shared_simulation


GENS = {
    "kplane": generate_collapsed_linear,
    "spiked_gaussian": generate_spiked_gaussian,
    "swiss": generate_collapsed_swiss,
    "torus": generate_collapsed_torus,
    "paraboloid": generate_paraboloid_graph,
    "contaminated_kplane": generate_contaminated_kplane,
    "contaminated_sphere": generate_noisy_sphere,
    "contaminated_kcube": generate_k_cube
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    shared = cfg["shared"]
    stage = cfg["alt_parallel"]
    run = cfg["run"]

    np_list = shared["n_list"]
    dim_list = shared["d_list"]
    eps = shared["eps_list"]
    out_path = resolve_output(cfg, stage["out_path"])
    max_workers = run["max_workers"]
    
    names = stage["families"]
    missing = [name for name in names if name not in GENS]
    if missing:
        raise ValueError(f"Unknown families in config: {missing}")
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
    out.to_csv("simulations/alt_simulation.csv", index=False)
