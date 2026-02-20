from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

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
    def gen(n_pts, dim, seed=None):
        return fn(n_pts, dim, _seed=seed, **fixed_kwargs)
    return gen


def elliptical_gen(eta, rotate=True, scale=1.0):
    def gen(n_pts, dim, seed=None):
        k = max(1, min(3, int(dim)))
        return generate_elliptical_gaussian(
            n_pts, dim, _k=k, _eta=eta, _scale=scale, _rotate=rotate, _seed=seed
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
    tau_df = load_tau_map()
    lm = False
    landmark = n / 2 if lm else None
    gens = build_gens()
    gen = gens[name]
    seed = abs(hash((n, d, float(e), name))) % (2**31 - 1)
    df = shared_simulation(
        gen,
        n,
        d,
        [0, 1, 2],
        0.1,
        0.1,
        100,
        e,
        landmark,
        tau_df,
        "gaussian",
        seed,
    )
    df["point_cloud"] = name
    return df


if __name__ == "__main__":
    np_list = [10, 50]
    dim_list = [5, 10, 20]
    eps_list = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    gens = build_gens()
    names = list(gens.keys())
    tasks = [(n, d, e, name) for n in np_list for d in dim_list for e in eps_list for name in names]

    rows = []
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for f in as_completed(futs):
            rows.append(f.result())

    out = pd.concat(rows, axis=0)
    out.to_csv("simulations/null_simulation.csv", index=False)
