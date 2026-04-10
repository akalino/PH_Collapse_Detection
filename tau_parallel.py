import argparse
import os
import time
import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from complex_persistence import (compute_dtm_vr_diagrams,
                                 compute_vr_diagrams,
                                 compute_witness_diagrams)
from config_utils import (load_config, resolve_output,
                          resolve_master_tau_map, stable_seed, utc_now_iso)
from metrics import finite_lengths
from point_clouds import (
    generate_gaussian,
    generate_noisy_sphere,
    generate_elliptical_gaussian,
    generate_uniform_ball,
    generate_xavier_normal,
    generate_layer_mixes,
)
from utils import (
    append_to_master_tau_map,
    find_missing_tau_keys,
    load_tau_map,
    required_tau_keys_from_config,
    knn_persistence_param_estimation,
)

warnings.filterwarnings("ignore", category=UserWarning)


def log(msg):
    print(f"[TAU] {msg}", flush=True)


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


_WORKER_GENS = None


def _init_worker():
    global _WORKER_GENS
    _WORKER_GENS = build_null_suite()


def _seed_stream(base_seed, n_sims):
    return [stable_seed(base_seed, i) for i in range(n_sims)]


def _cache_path(name, n, d, seed):
    return os.path.join(CACHE_ROOT, name, f"n{n}_d{d}", f"seed{int(seed)}.npz")


def _atomic_save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}.npz"
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

    max_dim = max(DIMS)
    max_edge = float(2.0 * cut)
    dtm_max_f = float(4.0 * cut)
    dtm_k = 10
    max_lands = 100

    vr_dgms = compute_vr_diagrams(x, max_edge, _max_dim=max_dim,
                                  _sparse=None, _backend="ripser")
    dtm_dgms = compute_dtm_vr_diagrams(x, dtm_max_f, _k=dtm_k, _max_dim=max_dim)
    wit_dgms = compute_witness_diagrams(x, max_lands)

    payload = {"cut": np.array([cut], dtype=float)}
    for hom_dim in DIMS:
        vr_lengths = finite_lengths(vr_dgms.get(hom_dim, np.empty((0, 2))))
        dtm_lengths = finite_lengths(dtm_dgms.get(hom_dim, np.empty((0, 2))))
        wit_lengths = finite_lengths(wit_dgms.get(hom_dim, np.empty((0, 2))))
        payload[f"vr_h{hom_dim}"] = vr_lengths
        payload[f"dtm_h{hom_dim}"] = dtm_lengths
        payload[f"witness_h{hom_dim}"] = wit_lengths

    _atomic_save_npz(out_path, **payload)
    return out_path


def aggregate_group(name, n, d, seeds):
    rows = []

    for filtration in ("vr", "dtm"):
        for hom_dim in DIMS:
            vals = []
            arr_name = f"{filtration}_h{hom_dim}"

            for s in seeds:
                p = _cache_path(name, n, d, s)
                if not os.path.exists(p):
                    continue
                z = np.load(p, allow_pickle=False)
                arr = z[arr_name]
                if getattr(arr, "size", 0):
                    vals.append(arr)

            cat = np.concatenate(vals) if vals else np.array([])
            tau = float(np.quantile(cat, TAU_Q)) if cat.size else 0.0

            rows.append({
                "point_cloud": name,
                "filtration": filtration,
                "n_pts": int(n),
                "dim": int(d),
                "hom_dim": int(hom_dim),
                "tau_q": TAU_Q,
                "tau": tau,
                "calibration_run_id": RUN_ID,
                "created_at": utc_now_iso(),
            })

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    shared = cfg["shared"]
    stage = cfg["tau_parallel"]
    run = cfg["run"]

    N_LIST = shared["n_list"]
    D_LIST = shared["d_list"]
    DIMS = shared["hom_dims"]
    N_SIM = shared["n_sim"]
    TAU_Q = shared["tau_q"]
    KNN_K = shared["knn_k"]
    KNN_Q = shared["knn_q"]
    KNN_B1 = shared["knn_b1"]
    KNN_METRIC = shared["knn_metric"]
    SEED = run["base_seed"]
    RUN_ID = run["run_id"]
    OUT_PATH = resolve_output(cfg, stage["out_path"])
    MASTER_TAU_PATH = resolve_master_tau_map(cfg)
    CACHE_ROOT = stage["cache_root"]
    MAX_WORKERS = run["max_workers"]
    FAMILIES = stage["families"]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MASTER_TAU_PATH), exist_ok=True)
    os.makedirs(CACHE_ROOT, exist_ok=True)

    log(f"starting tau calibration, run_id={RUN_ID}")
    log(f"master_tau_map_path={MASTER_TAU_PATH}")
    log(f"cache_root={CACHE_ROOT}")

    master_df = load_tau_map(MASTER_TAU_PATH)
    required_df = required_tau_keys_from_config(cfg, families=FAMILIES)
    missing_df = find_missing_tau_keys(required_df, master_df)

    if missing_df.empty:
        log("no missing tau keys; master tau map already covers requested config")
        master_df.to_csv(OUT_PATH, index=False)
        log(f"wrote local tau snapshot to {OUT_PATH}")
        raise SystemExit(0)

    missing_groups = (
        missing_df[["point_cloud", "n_pts", "dim"]]
        .drop_duplicates()
        .sort_values(["point_cloud", "n_pts", "dim"])
        .reset_index(drop=True)
    )

    log(f"missing tau rows={len(missing_df)}, missing groups={len(missing_groups)}")
    for _, row in missing_groups.iterrows():
        log(f"missing group point_cloud={row['point_cloud']} n={row['n_pts']} d={row['dim']}")

    seed_map = {}
    for _, row in missing_groups.iterrows():
        name = row["point_cloud"]
        n = int(row["n_pts"])
        d = int(row["dim"])
        seed_map[(name, n, d)] = _seed_stream(SEED, N_SIM)

    tasks = []
    for (name, n, d), seeds in seed_map.items():
        for s in seeds:
            tasks.append((name, n, d, int(s)))

    log(f"running {len(tasks)} seed tasks with max_workers={MAX_WORKERS}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker) as ex:
        futs = [ex.submit(run_one_seed, t) for t in tasks]
        for _ in tqdm(as_completed(futs), total=len(futs)):
            pass

    rows = []
    for (name, n, d), seeds in seed_map.items():
        log(f"aggregating point_cloud={name} n={n} d={d}")
        rows.extend(aggregate_group(name, n, d, seeds))

    new_df = pd.DataFrame(rows)
    updated_master = append_to_master_tau_map(MASTER_TAU_PATH, new_df)
    updated_master.to_csv(OUT_PATH, index=False)

    log(f"appended {len(new_df)} rows to master tau map")
    log(f"master rows now={len(updated_master)}")
    log(f"wrote local tau snapshot to {OUT_PATH}")