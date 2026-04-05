import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from config_utils import load_config, resolve_output, stable_seed
from point_clouds import (
    generate_gaussian,
    generate_noisy_sphere,
    generate_elliptical_gaussian,
    generate_uniform_ball,
    generate_xavier_normal,
    generate_layer_mixes,
)
from utils import load_tau_map, shared_simulation


def wrap(fn, **fixed_kwargs):
    def gen(n_pts, dim, _seed=None):
        return fn(n_pts, dim, _seed=_seed, **fixed_kwargs)
    return gen


def elliptical_gen(eta, rotate=True, scale=1.0):
    def gen(n_pts, dim, _seed=None):
        k = max(1, min(3, int(dim)))
        return generate_elliptical_gaussian(
            n_pts, dim, _k=k, _eta=eta, _scale=scale, _rotate=rotate, _seed=_seed
        )
    return gen


def build_gens():
    gens = {}
    gens["gaussian"] = generate_gaussian

    for sig in (0.1, 0.3, 0.5):
        gens[f"noisy_sphere_{sig}"] = wrap(generate_noisy_sphere, _radius=1.0, _noise_sig=sig)

    gens["uniform_ball"] = wrap(generate_uniform_ball, _radius=1.0, _noise_sig=0.0)
    gens["uniform_ball_noisy0.05"] = wrap(generate_uniform_ball, _radius=1.0, _noise_sig=0.05)

    for eta in (1.0, 0.5, 0.2, 0.1, 0.05):
        gens[f"ell_gaussian_eta{eta}_rot"] = elliptical_gen(eta, rotate=True, scale=1.0)

    try:
        gens["xavier_unit_norm"] = wrap(
            generate_xavier_normal, _scale="unit_norm", _f_in=0.0, _f_out=1.0
        )
    except Exception:
        pass

    try:
        gens["layer_mix_sigmas_0.5_1.0_1.5"] = wrap(
            generate_layer_mixes,
            _sigmas=(0.5, 1.0, 1.5),
            _base="gaussian",
            _scale="unit_norm",
        )
        gens["layer_mix_sigmas_0.25_1.0_2.0"] = wrap(
            generate_layer_mixes,
            _sigmas=(0.25, 1.0, 2.0),
            _base="gaussian",
            _scale="unit_norm",
        )
    except Exception:
        pass

    return gens


def run_one(task):
    n, d, e, name = task
    gens = build_gens()
    gen = gens[name]
    seed = BASE_SEED
    df = shared_simulation(
        gen,
        n,
        d,
        HOM_DIMS,
        P,
        ALPHA,
        N_SIM,
        e,
        LANDMARK,
        TAU_DF,
        TAU_REF,
        seed,
    )
    df["point_cloud"] = name
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    shared = cfg["shared"]
    stage = cfg["null_parallel"]
    run = cfg["run"]
    TAU_DF = load_tau_map(resolve_output(cfg, cfg["tau_parallel"]["out_path"]))
    BASE_SEED = run["base_seed"]
    HOM_DIMS = shared["hom_dims"]
    ALPHA = shared["alpha"]
    P = shared["p"]
    N_SIM = shared["n_sim"]
    LANDMARK = shared["landmark"]
    TAU_REF = shared["tau_reference_family"]

    np_list = shared["n_list"]
    dim_list = shared["d_list"]
    eps_list = shared["eps_list"]
    out_path = resolve_output(cfg, stage["out_path"])
    max_workers = run["max_workers"]
    
    names = stage["families"]
    GENS = build_gens()
    missing = [name for name in names if name not in GENS]
    if missing:
        raise ValueError(f"Unknown families in config: {missing}")
    tasks = [(n, d, e, name) for n in np_list for d in dim_list for e in eps_list for name in names]

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for f in as_completed(futs):
            rows.append(f.result())

    out = pd.concat(rows, axis=0)
    out.to_csv(out_path, index=False)
