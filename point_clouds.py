# Code for generating Gaussians.
import numpy as np


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
        u = r_num.uniform(low=0.5, high=0.5, size=(_n_pts, _dim))
    else:
        u = r_num.uniform(low=0.0, high=1.0, size=(_n_pts, _dim))
    y = np.zeros((_n_pts, _dim))
    y[:, :_k] = u

    if _rotate and _dim > 1:
        r = _ortohnorm_basis(_dim, _dim, r_num)
        y = y @ r.T

    x = y + _eps * r_num.normal(size=(_n_pts, _dim))
    return x
