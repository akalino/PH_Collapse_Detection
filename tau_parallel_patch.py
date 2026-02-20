import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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

KNN_K = 10
KNN_Q = 0.95
KNN_B1 = 1.05
KNN_METRIC = "euclidean"

N_LIST = [10, 50, 100]
D_LIST = [5, 10, 20]
DIMS = [0, 1, 2]

OUT_PATH = "calibration/tau_map.csv"
CACHE_ROOT = "calibration/tau_cache"

MAX_WORKERS = 3


# -----------------------
# Generator suite
# -----------------------
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


def build_null_suite():
    gens = {}
    gens["gaussian"] = wrap(generate_gaussian)

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


# Per-process globals (set by initializer)
_WORKER_GENS = None


def _init_worker():
    global _WORKER_GENS
    _WORKER_GENS = build_null_suite()


def _seed_stream(base_seed, n_sims):
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 2**31 - 1, size=n_sims, dtype=np.int64)


def _cache_path(name, n, d, seed):
    return os.path.join(CACHE_ROOT, name, f"n{n}_d{d}", f"seed{int(seed)}.npz")


def _atomic_save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def run_one_seed(task):
    name, n, d, seed = task
    out_path = _cache_path(name, n, d, seed)

    if os.path.exists(out_path):
        return out_path

    gen = _WORKER_GENS[name]
    x = gen(n, d, _seed=int(seed))
    if isinstance(x, tuple):
        x = x[0]

    cut = knn_persistence_param_estimation(x, KNN_K, KNN_Q, KNN_B1, KNN_METRIC)

    # keep your sparse setting from the current tau_parallel (cheap win, no landmarks)
    l = compute_lengths(x, DIMS, cut, _sparse=cut * 0.5)

    vr = l.get("vr", np.array([], dtype=float))
    dtm = l.get("dtm", np.array([], dtype=float))

    _atomic_save_npz(out_path, vr=vr, dtm=dtm)
    return out_path


def aggregate_group(name, n, d, seeds):
    vr_all = []
    dtm_all = []

    for s in seeds:
        p = _cache_path(name, n, d, s)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        vr = z["vr"]
        dtm = z["dtm"]
        if getattr(vr, "size", 0):
            vr_all.append(vr)
        if getattr(dtm, "size", 0):
            dtm_all.append(dtm)

    vr_cat = np.concatenate(vr_all) if vr_all else np.array([])
    dtm_cat = np.concatenate(dtm_all) if dtm_all else np.array([])

    tau_vr = float(np.quantile(vr_cat, TAU_Q)) if vr_cat.size else 0.0
    tau_dtm = float(np.quantile(dtm_cat, TAU_Q)) if dtm_cat.size else 0.0

    return [
        {"point_cloud": name, "n_pts": n, "dim": d, "filtration": "vr", "tau_q": TAU_Q, "tau": tau_vr},
        {"point_cloud": name, "n_pts": n, "dim": d, "filtration": "dtm", "tau_q": TAU_Q, "tau": tau_dtm},
    ]


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    os.makedirs(CACHE_ROOT, exist_ok=True)

    gens = build_null_suite()
    names = list(gens.keys())

    seed_map = {}
    for name in names:
        for n in N_LIST:
            for d in D_LIST:
                base_seed = abs(hash((name, int(n), int(d), "tau"))) % (2**31 - 1)
                seeds = _seed_stream(base_seed, N_SIM).tolist()
                seed_map[(name, n, d)] = seeds

    tasks = []
    for (name, n, d), seeds in seed_map.items():
        for s in seeds:
            tasks.append((name, n, d, int(s)))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker) as ex:
        futs = [ex.submit(run_one_seed, t) for t in tasks]
        for _ in tqdm(as_completed(futs), total=len(futs)):
            pass

    rows = []
    for (name, n, d), seeds in seed_map.items():
        rows.extend(aggregate_group(name, n, d, seeds))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")
