import numpy as np
import gudhi as gd

from gudhi.dtm_rips_complex import DTMRipsComplex


def _diag_by_dim(_tree, _max_dim):
    """

    :param _tree: Simplex tree.
    :param _max_dim: Maximum dimension to compute persistence diagram.
    :return: out, dictionary with dimension: birth/death pairs.
    """
    st = _tree
    st.compute_persistence()
    out = {}
    for d in range(_max_dim + 1):
        pairs = st.persistence_intervals_in_dimension(d)
        out[d] = np.array(pairs) if len(pairs) else np.empty((0, 2))
    return out


def compute_vr_diagrams(_points, _max_edge_length, _max_dim=3, _sparse=None):
    """

    :param _points:
    :param _max_edge_length:
    :param _max_dim:
    :param _sparse:
    :return:
    """
    rips = gd.RipsComplex(points=_points, max_edge_length=_max_edge_length, sparse=_sparse)
    st = rips.create_simplex_tree(max_dimension=_max_dim + 1)
    return _diag_by_dim(st, _max_dim)


def compute_dtm_vr_diagrams(_points, _max_filtration=100, _k=10, _q=2, _max_dim=3):
    """

    :param _points:
    :param _max_filtration:
    :param _k:
    :param _q:
    :param _max_dim:
    :return:
    """
    dtm_rips = DTMRipsComplex(points=_points, k=_k, q=_q, max_filtration=_max_filtration)
    st = dtm_rips.create_simplex_tree(max_dimension=_max_dim + 1)
    return _diag_by_dim(st, _max_dim)
