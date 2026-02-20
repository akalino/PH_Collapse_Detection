# Code for generating all point clouds (null/alt).
import numpy as np

# Helper functions

def _rng(_seed=None):
    """
    Random number generator wrapper.

    :param _seed: Seeding rng.
    :return: seed.
    """
    if isinstance(_seed, np.random.Generator):
        return _seed
    return np.random.default_rng(_seed)


def _ortohnorm_basis(_n_pts, _dim, _seed=None):
    """
    Orthonormal basis using QR factorization.
    :param _n_pts: Number of points in the point cloud.
    :param _dim: Dimension of the point cloud.
    :param _seed: Seed for the random number generator.
    :return:
    """
    r_num = _rng(_seed)
    a = r_num.normal(size=(_n_pts, _dim))
    q, _ = np.linalg.qr(a)
    return q[:, :_dim]


def _add_isotropic_noise(_y, _sigma, _seed=None):
    """

    :param _y:
    :param _sigma:
    :param _seed:
    :return:
    """
    r_num = _rng(_seed)
    return _y + _sigma * r_num.normal(size=_y.shape)


# Null baselines

def generate_gaussian(_n_pts, _dim, _seed=None):
    """
    Generates Gaussian distribution with given dimension.

    :param _n_pts: Number of points in the point cloud.
    :param _dim: The ambient dimension.
    :param _seed: Seed for the random number generator.
    :return: numpy matrix of Gaussian distribution.
    """
    r_num = _rng(_seed)
    x = r_num.normal(size=(_n_pts, _dim))
    return x


def generate_noisy_sphere(_n_pts, _dim, _noise_sig=0.3, _radius=1.0, _seed=None):
    """
    Generate non-linear point cloud from sphere plus isotropic noise.

    :param _n_pts: Number of points in the point cloud.
    :param _dim: The ambient dimension.
    :param _noise_sig: Amount of isotropic noise.
    :param _radius: Radius of the sphere.
    :param _seed: Random seed for the random number generator.
    :return:
    """
    r_num = _rng(_seed)
    z = r_num.normal(size=(_n_pts, _dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = _radius * z
    x = _add_isotropic_noise(y, _noise_sig, _seed)
    return x


def generate_elliptical_gaussian(_n_pts, _dim, _k=3, _eta=0.1, _scale=1.0,
                                 _seed=None, _rotate=True):
    """
    :param _n_pts: Number of points.
    :param _dim: Ambient dimension.
    :param _k: Number of high-variance directions.
    :param _eta: Shrink factor for remaining directions in (0,1].
    :param _scale: Global scale factor (std dev in top-k directions).
    :param _seed: RNG seed or np.random.Generator.
    :param _rotate: If True, apply a random orthonormal rotation to avoid axis alignment.
    """
    r_num = _rng(_seed)
    scales = np.full(_dim, _scale, dtype=float)
    scales[_k:] = _scale * _eta
    x = r_num.normal(size=(_n_pts, _dim))
    if _rotate and _dim > 1:
        u = _ortohnorm_basis(_dim, _dim, r_num)
        x = x @ u.T
    return x


def generate_uniform_ball(_n_pts, _dim, _radius=1.0, _noise_sig=0.0, _seed=None):

    r_num = _rng(_seed)
    z = r_num.normal(size=(_n_pts, _dim))
    z_norm = np.linalg.norm(z, axis=1, keepdims=True)
    z_norm = np.maximum(z_norm, 1e-12)
    u = z / z_norm

    rad = _radius * r_num.uniform(size=(_n_pts, 1)) ** (1.0 / _dim)
    x = rad * u

    if _noise_sig > 0:
        x = _add_isotropic_noise(x, _noise_sig, r_num)
    return x

# Nulls that match neural network initializations

def generate_xavier_normal(_n_pts, _dim, _scale="unit_norm",
                           _f_in=0.0, _f_out=1.0, _seed=17):
    """
    Var[Unif(-a,a)] = a^2/3 = sigma2
    """
    r_num = _rng(_seed)
    if _scale == "unit_norm":
        sigma2 = 1.0 / float(_dim)
    elif _scale == "glorot":
        sigma2 = 2.0 / float(_f_in + _f_out)
    else:
        raise ValueError("scale_mode must be one of {'unit_norm','glorot'}")
    a = np.sqrt(3.0 * sigma2)
    x = r_num.uniform(low=-a, high=a, size=(_n_pts, _dim))
    return x


def generate_layer_mixes(_n_pts, _dim, _sigmas=(0.5, 1.0, 1.5),
                         _weights=None, _base="gaussian",
                         _scale="unit_norm", _seed=None):
    """
    In early training embeddings are mostly isotropic but have norms based on token types or contexts.
    Keep the covariance isotropic but introduce radial structure.
    """
    r_num = _rng(_seed)
    k = len(_sigmas)
    if _weights is None:
        w = np.full(k, 1.0 / k, dtype=float)
    else:
        w = np.asarray(_weights, dtype=float)

    if _scale == "unit_norm":
        base_std = 1.0 / np.sqrt(float(_dim))
    elif _scale == "none":
        base_std = 1.0
    components = r_num.choice(k, size=_n_pts, p=w)

    x = np.empty((_n_pts, _dim), dtype=float)
    if _base == "gaussian":
        for j in range(k):
            idx = np.where(components == j)[0]
            if idx.size == 0:
                continue
            std = base_std * float(_sigmas[j])
            x[idx, :] = r_num.normal(loc=0.0, scale=std, size=(idx.size, _dim))

    elif _base == "xavier_uniform":
        for j in range(k):
            idx = np.where(components == j)[0]
            if idx.size == 0:
                continue
            std = base_std * float(_sigmas[j])
            a = np.sqrt(3.0) * std
            x[idx, :] = r_num.uniform(low=-a, high=a, size=(idx.size, _dim))
    return x

# Alternate / collapse tests

def generate_collapsed_linear(_n_pts, _dim, _k=3, _eps=0.1, _spread=1.0, _seed=None):
    """
    Generates a collapsed linear manifold from a k-plane in R^d plus epsilon noise.

    :param _n_pts: Number of points in the point cloud.
    :param _dim: The ambient dimension.
    :param _k: dimension of the k-plane.
    :param _eps: Controls concentration (-> 0 ~ more concentration).
    :param _spread: Spread of points on sphere.
    :param _seed: Random seed for the random number generator.
    :return:
    """
    if not (1 <= _k <= _dim):
        raise ValueError("k must be between 1 and d.")
    r_num = _rng(_seed)
    u = _ortohnorm_basis(_dim, _k, r_num)
    s = _spread * r_num.normal(size=(_n_pts, _k))
    y = s @ u.T
    x = _add_isotropic_noise(y, _eps, r_num)
    return x  # , y, u


def generate_collapsed_swiss(_n_pts, _dim, _eps=0.1,
                           _t_range=(1.5 * np.pi, 4.5 * np.pi),
                           _h_range=(0.0, 10.0), _seed=None):
    """
    Creating a swiss roll y in two-dimensions, then embedding in R^3.
    After, padding with zeros to embed in R^d.
    Finally, compute x as y + eps * N(0, I).

    :param _n_pts: Number of points in the point cloud.
    :param _dim: Ambient dimension.
    :param _eps: Noise.
    :param _t_range: Tightness of the roll.
    :param _h_range: Height range of the roll.
    :param _seed: Random seed for the random number generator.
    :return: x, swiss roll embedded in R^d
    """
    r_num = _rng(_seed)
    t = r_num.uniform(_t_range[0], _t_range[1], size=_n_pts)
    h = r_num.uniform(_h_range[0], _h_range[1], size=_n_pts)

    # swiss roll in R^3
    y3 = np.zeros((_n_pts, 3))
    y3[:, 0] = t * np.cos(t)
    y3[:, 1] = h
    y3[:, 2] = t * np.sin(t)

    # transform to R^d
    y = np.zeros((_n_pts, _dim))
    y[:, :3] = y3

    # add noise
    x = _add_isotropic_noise(y, _eps, r_num)
    return x  #, y


def generate_collapsed_torus(_n_pts, _dim, _eps=0.1,
                           _major_r=2.0, _minor_r=0.5, _seed=None):
    """
    Same as the swiss roll; create in R^3, embed in R^d, add noise.
    :param _n_pts: Number of points in the point cloud.
    :param _dim: Ambient dimension.
    :param _eps: Noise.
    :param _major_r: Major radius of the torus.
    :param _minor_r: Minor radius of the torus.
    :param _seed: Random seed for the random number generator.
    :return:
    """
    if _dim < 3:
        raise ValueError("dim must be >= 3.")
    r_num = _rng(_seed)
    theta = r_num.uniform(0, 2 * np.pi, size=_n_pts)
    phi = r_num.uniform(0, 2 * np.pi, size=_n_pts)

    y3 = np.zeros((_n_pts, 3))
    y3[:, 0] = (_major_r + _minor_r * np.cos(phi)) * np.cos(theta)
    y3[:, 1] = (_major_r + _minor_r * np.cos(phi)) * np.sin(theta)
    y3[:, 2] = _minor_r * np.sin(phi)

    y = np.zeros((_n_pts, _dim))
    y[:, :3] = y3
    x = _add_isotropic_noise(y, _eps, r_num)
    return x  #, y


def generate_spiked_gaussian(_n_pts, _dim, _k=3, _eps=0.1,
                             _seed=None, _rotate=True):
    """

    :param _n_pts:
    :param _dim:
    :param _k:
    :param _eps:
    :param _seed:
    :param _rotate:
    :return:
    """
    if not (1 <= _k <= _dim):
        raise ValueError("k must be in range [1, dim]")
    r_num = _rng(_seed)

    scales = np.ones(_dim)
    scales[_k:] = _eps
    x = r_num.normal(size=(_n_pts, _dim)) * scales[None, :]
    if _rotate and _dim > 1:
        u = _ortohnorm_basis(_dim, _dim, r_num)
        x = x @ u.T
    return x


def generate_contaminated_kplane(_n_pts, _dim, _k=3,
                                 _eps=0.1, _eta=0.1, _seed=None):
    """

    :param _n_pts:
    :param _dim:
    :param _k:
    :param _eps:
    :param _eta:
    :param _seed:
    :return:
    """
    if not (0.0 <= _eta <= 1.0):
        raise ValueError("eta must be in [0,1].")
    if not (1 <= _k <= _dim):
        raise ValueError("k must be between 1 and dim.")
    r_num = _rng(_seed)

    m = int(np.round((1.0 - _eta) * _n_pts))
    o = _n_pts - m

    # Inliers: k-plane + eps isotropic noise
    u = _ortohnorm_basis(_dim, _k, r_num)
    s = r_num.normal(size=(m, _k))
    y = s @ u.T
    x_in = y + _eps * r_num.normal(size=(m, _dim))

    # Outliers: full-dim clutter
    x_out = r_num.normal(size=(o, _dim))

    x = np.vstack([x_in, x_out])
    r_num.shuffle(x, axis=0)
    return x


def generate_paraboloid_graph(_n_pts, _dim, _k=2, _eps=0.1, _amp=0.5, _seed=None):
    """

    :param _n_pts:
    :param _dim:
    :param _k:
    :param _eps:
    :param _amp:
    :param _seed:
    :return:
    """
    if _dim < _k + 1:
        raise ValueError("Need dim >= k + 1 to embed paraboloid graph.")
    r_num = _rng(_seed)

    u = r_num.normal(size=(_n_pts, _k))
    height = _amp * np.sum(u ** 2, axis=1, keepdims=True)

    y = np.zeros((_n_pts, _dim))
    y[:, :_k] = u
    y[:, _k: _k + 1] = height

    x = y + _eps * r_num.normal(size=(_n_pts, _dim))
    return x


def generate_k_cube(_n_pts, _dim, _k=3, _eps=0.1, _seed=None, _rotate=True, _center=True):
    """

    :param _n_pts:
    :param _dim:
    :param _k:
    :param _eps:
    :param _seed:
    :param _rotate:
    :param _center:
    :return:
    """
    if not (1 <= _k <= _dim):
        raise ValueError("k must be in range [1, dim]")
    r_num = _rng(_seed)

    if _center:
        u = r_num.uniform(low=0.5, high=0.5, size=(_n_pts, _k))
    else:
        u = r_num.uniform(low=0.0, high=1.0, size=(_n_pts, _k))
    y = np.zeros((_n_pts, _dim))
    y[:, :_k] = u

    if _rotate and _dim > 1:
        r = _ortohnorm_basis(_dim, _dim, r_num)
        y = y @ r.T

    x = y + _eps * r_num.normal(size=(_n_pts, _dim))
    return x
