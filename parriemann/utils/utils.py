from numba import njit, prange
import numpy as np


def sliding_windows(data, step_size, shift):
    return np.array([data[i * step_size:i * step_size + shift].T
                     for i in range(data.shape[0] // step_size - shift // step_size + 1)])\
        .reshape((-1, data.shape[1], shift))


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
