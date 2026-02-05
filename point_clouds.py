# Code for generating Gaussians.
import numpy as np


def _rng(_seed=None):
    """
    Builds random number generator object for a seed.
    :param _seed: random seed (int).
    :return: numpy random generator.
    """
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
    return x, y


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
    return x, y, u


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
    return x, y


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
    return x, y
