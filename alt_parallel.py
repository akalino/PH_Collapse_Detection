import argparse
import os

import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from config_utils import (load_config, resolve_output,
                          resolve_master_tau_map, stable_seed)
from point_clouds import (
    generate_collapsed_linear, generate_spiked_gaussian,
    generate_collapsed_swiss, generate_collapsed_torus, generate_paraboloid_graph,
    generate_contaminated_kplane, generate_k_cube, generate_noisy_sphere
)
from utils import load_tau_map, shared_simulation, validate_required_tau_keys


GENS = {
    "kplane": generate_collapsed_linear,
    "spiked_gaussian": generate_spiked_gaussian,
    "swiss": generate_collapsed_swiss,
    "torus": generate_collapsed_torus,
    "paraboloid_graph": generate_paraboloid_graph,
    "contaminated_kplane": generate_contaminated_kplane,
    "contaminated_sphere": generate_noisy_sphere,
    "contaminated_kcube": generate_k_cube
}


def run_one(_task):
    n, d, e, name = _task

    seed = stable_seed(BASE_SEED, "alt", name, n, d, e)

    _df = shared_simulation(GENS[name], n, d, HOM_DIMS,
                            P, ALPHA, N_SIM,
                            e, LANDMARK, TAU_DF, TAU_REF, seed)
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
    TAU_DF = load_tau_map(resolve_master_tau_map(cfg))
    BASE_SEED = run["base_seed"]
    HOM_DIMS = shared["hom_dims"]
    ALPHA = shared["alpha"]
    P = shared["p"]
    N_SIM = shared["n_sim"]
    LANDMARK = shared["landmark"]
    TAU_REF = shared["tau_reference_family"]

    np_list = shared["n_list"]
    dim_list = shared["d_list"]
    eps = shared["eps_list"]
    out_path = resolve_output(cfg, stage["out_path"])
    max_workers = run["max_workers"]
    
    names = stage["families"]
    validate_required_tau_keys(cfg, TAU_DF, families=cfg["tau_parallel"]["families"])

    missing = [name for name in names if name not in GENS]
    if missing:
        raise ValueError(f"Unknown families in config: {missing}")
    tasks = [(n, d, e, name) for n in np_list for d in dim_list for e in eps for name in names]

    out_dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, t) for t in tasks]
        for f in as_completed(futures):
            try:
                out_dfs.append(f.result())
            except Exception as ex:
                print("Task failed:", ex)

    out = pd.concat(out_dfs, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
