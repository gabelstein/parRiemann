from numba import njit, prange
import numpy as np
import parriemann as pr


@njit
def riemann_kernel(A, B, G_invsq):
    A_ = G_invsq@A@G_invsq
    B_ = G_invsq@B@G_invsq
    return np.trace(A_@B_)


@njit(parallel=True)
def riemann_kernel_matrix(X, Y):
    G = pr.utils.mean.mean_riemann(X)
    G_invsq = pr.utils.base.invsqrtm(G)
    Ntx, Ne, Ne = X.shape

    X_ = np.zeros((Ntx, Ne, Ne))
    for index in prange(Ntx):
        X_[index] = G_invsq@X[index]@G_invsq

    Nty, Ne, Ne = Y.shape
    Y_ = np.zeros((Nty, Ne, Ne))
    for index in prange(Nty):
        Y_[index] = G_invsq @ Y[index] @ G_invsq

    res = np.zeros((Nty, Ntx))
    for i in prange(Nty):
        for j in prange(Ntx):
            res[i][j] = np.trace(Y_[i]@X_[j])
    return res


def check_version(library, min_version):
    """Check minimum library version required

    Parameters
    ----------
    library : str
        The library name to import. Must have a ``__version__`` property.
    min_version : str
        The minimum version string. Anything that matches
        ``'(\\d+ | [a-z]+ | \\.)'``

    Returns
    -------
    ok : bool
        True if the library exists with at least the specified version.

    Adapted from MNE-Python: http://github.com/mne-tools/mne-python
    """
    from distutils.version import LooseVersion
    ok = True
    try:
        library = __import__(library)
    except ImportError:
        ok = False
    else:
        this_version = LooseVersion(library.__version__)
        if this_version < min_version:
            ok = False
    return ok
