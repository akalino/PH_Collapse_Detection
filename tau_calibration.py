import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from point_clouds import (
    generate_gaussian,
    generate_noisy_sphere,
    generate_elliptical_gaussian,
    generate_uniform_ball,
    generate_xavier_normal,
    generate_layer_mixes,
)
from utils import knn_persistence_param_estimation
from metrics import compute_lengths

TAU_Q = 0.95
N_SIM = 100


def wrap(fn, **fixed_kwargs):
    def gen(n_pts, dim, _seed=None):
        return fn(n_pts, dim, _seed=_seed, **fixed_kwargs)
    return gen


def elliptical_gen(eta, rotate=True, scale=1.0):
    def gen(n_pts, dim, _seed=None):
        k = max(1, min(3, int(dim)))
        return generate_elliptical_gaussian(
            n_pts, dim, _k=k, _eta=eta, _scale=scale, _seed=_seed, _rotate=rotate
        )
    return gen


def build_tau_map(_n_list, _d_list, _dims, _out, _null_clouds):
    rows = []
    r_num = np.random.default_rng(123)

    for name, gen in _null_clouds:
        for n in _n_list:
            for d in _d_list:
                all_vr = []
                all_dtm = []

                for _ in tqdm(range(N_SIM), desc=f"{name} (n={n}, d={d})", leave=False):
                    s = int(r_num.integers(0, 2**31 - 1))
                    try:
                        x = gen(n, d, _seed=s)
                    except TypeError:
                        x = gen(n, d, _eps=0.0, _seed=s)

                    if isinstance(x, tuple):
                        x = x[0]

                    cut = knn_persistence_param_estimation(x, 10, 0.95, 1.05, "euclidean")
                    l = compute_lengths(x, _dims, cut)

                    if l["vr"].size:
                        all_vr.append(l["vr"])
                    if l["dtm"].size:
                        all_dtm.append(l["dtm"])

                vr_cat = np.concatenate(all_vr) if all_vr else np.array([])
                dtm_cat = np.concatenate(all_dtm) if all_dtm else np.array([])

                tau_vr = float(np.quantile(vr_cat, TAU_Q)) if vr_cat.size else 0.0
                tau_dtm = float(np.quantile(dtm_cat, TAU_Q)) if dtm_cat.size else 0.0

                rows.append(
                    {"point_cloud": name, "n_pts": n, "dim": d, "filtration": "vr", "tau_q": TAU_Q, "tau": tau_vr}
                )
                rows.append(
                    {"point_cloud": name, "n_pts": n, "dim": d, "filtration": "dtm", "tau_q": TAU_Q, "tau": tau_dtm}
                )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(_out), exist_ok=True)
    df.to_csv(_out, index=False)
    print(f"Wrote tau map: {_out}")


if __name__ == "__main__":
    n_list = [10, 50, 100]
    d_list = [5, 10, 20]
    dims = [0, 1, 2, 3]

    null_clouds = [("gaussian", generate_gaussian)]

    for sig in (0.1, 0.3, 0.5):
        null_clouds.append((f"noisy_sphere_{sig}", wrap(generate_noisy_sphere, _noise_sig=sig, _radius=1.0)))

    null_clouds.append(("uniform_ball", wrap(generate_uniform_ball, _radius=1.0, _noise_sig=0.0)))
    null_clouds.append(("uniform_ball_noisy0.05", wrap(generate_uniform_ball, _radius=1.0, _noise_sig=0.05)))

    for eta in (1.0, 0.5, 0.2, 0.1, 0.05):
        null_clouds.append((f"ell_gaussian_eta{eta}_rot", elliptical_gen(eta, rotate=True, scale=1.0)))

    null_clouds.append(("xavier_unit_norm", wrap(generate_xavier_normal, _scale="unit_norm", _f_in=0.0, _f_out=1.0)))

    null_clouds.append(
        ("layer_mix_sigmas_0.5_1.0_1.5",
         wrap(generate_layer_mixes, _sigmas=(0.5, 1.0, 1.5), _base="gaussian", _scale="unit_norm"))
    )
    null_clouds.append(
        ("layer_mix_sigmas_0.25_1.0_2.0",
         wrap(generate_layer_mixes, _sigmas=(0.25, 1.0, 2.0), _base="gaussian", _scale="unit_norm"))
    )

    build_tau_map(
        n_list,
        d_list,
        dims,
        _out="calibration/tau_map.csv",
        _null_clouds=null_clouds,
    )
