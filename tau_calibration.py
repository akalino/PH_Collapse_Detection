import numpy as np
import pandas as pd

from tqdm import tqdm

from point_clouds import generate_gaussian, generate_noisy_sphere
from utils import knn_persistence_param_estimation
from metrics import compute_lengths


TAU_Q = 0.95
N_SIM = 200


def build_tau_map(_n_list, _d_list, _dims, _out, _null_clouds):
    rows = []
    r_num = np.random.default_rng(123)

    for name, gen in _null_clouds:
        for n in _n_list:
            for d in _d_list:
                all_vr = []
                all_dtm = []

                for _ in tqdm(range(N_SIM)):
                    s = int(r_num.integers(0, 2**10 - 1))
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

                rows.append({"point_cloud": name, "n_pts": n, "dim": d,
                             "filtration": "vr", "tau_q": TAU_Q, "tau": tau_vr})
                rows.append({"point_cloud": name, "n_pts": n, "dim": d,
                             "filtration": "dtm", "tau_q": TAU_Q, "tau": tau_dtm})

    df = pd.DataFrame(rows)
    df.to_csv(_out, index=False)
    print(f"Wrote tau map: {_out}")


if __name__ == "__main__":
    n_list = [10, 50, 100]
    d_list = [5, 10, 20]
    dims = [0, 1, 2, 3]

    null_clouds = [("gaussian", generate_gaussian)]
                   #("noisy_sphere", generate_noisy_sphere)]

    build_tau_map(n_list, d_list, dims,
                  _out="comparisons/tau_map.csv",
                  _null_clouds=null_clouds)
