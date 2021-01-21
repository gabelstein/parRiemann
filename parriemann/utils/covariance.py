import numpy
from numba import njit, prange
from .mean import mean_euclid


# Mapping different estimator on the sklearn toolbox


@njit
def _lwf(X):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = ledoit_wolf(X.T)
    return C


@njit
def _oas(X):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = oas(X.T)
    return C


@njit
def _scm(X):
    """Wrapper for sklearn sample covariance estimator"""
    return empirical_covariance(X.T)


def _check_est(est):
    """Check if a given estimator is valid"""

    # Check estimator exist and return the correct function
    estimators = {
        'cov': numpy.cov,
        'scm': _scm,
        'lwf': _lwf,
        'oas': _oas,
        'corr': numpy.corrcoef
    }

    if callable(est):
        # All good (cross your fingers)
        pass
    elif est in estimators.keys():
        # Map the corresponding estimator
        est = estimators[est]
    else:
        # raise an error
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function""" % (est, (' , ').join(estimators.keys())))
    return est


@njit
def covariances(X, estimator='cov'):
    """Estimation of covariance matrix."""
    if estimator == 'cov':
        return _cov_loop(X, numpy.cov)
    elif estimator == 'scm':
        return _cov_loop(X, _scm)
    elif estimator == 'lwf':
        return _cov_loop(X, _lwf)
    elif estimator == 'oas':
        return _cov_loop(X, _oas)
    elif estimator == 'corr':
        return _cov_loop(X, numpy.corrcoef)
    else:
        raise ValueError("Not a valid estimator.")


@njit(parallel=True)
def _cov_loop(X, est):
    Nt, Ne, Ns = X.shape
    covmats = numpy.zeros((Nt, Ne, Ne))
    for i in prange(Nt):
        covmats[i] = est(X[i])
    return covmats


def covariances_EP(X, P, estimator='cov'):
    """Special form covariance matrix."""
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    Np, Ns = P.shape
    covmats = numpy.zeros((Nt, Ne + Np, Ne + Np))
    for i in range(Nt):
        covmats[i] = est(numpy.concatenate((P, X[i]), axis=0))
    return covmats


def eegtocov(sig, window=128, overlapp=0.5, padding=True, estimator='cov'):
    """Convert EEG signal to covariance using sliding window"""
    est = _check_est(estimator)
    X = []
    if padding:
        padd = numpy.zeros((int(window / 2), sig.shape[1]))
        sig = numpy.concatenate((padd, sig, padd), axis=0)

    Ns, Ne = sig.shape
    jump = int(window * overlapp)
    ix = 0
    while (ix + window < Ns):
        X.append(est(sig[ix:ix + window, :].T))
        ix = ix + jump

    return numpy.array(X)


def coherence(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute coherence."""
    cosp = cospectrum(X, window, overlap, fmin, fmax, fs)
    coh = numpy.zeros_like(cosp)
    for f in range(cosp.shape[-1]):
        psd = numpy.sqrt(numpy.diag(cosp[..., f]))
        coh[..., f] = cosp[..., f] / numpy.outer(psd, psd)
    return coh


def cospectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute Cospectrum."""
    Ne, Ns = X.shape
    number_freqs = int(window / 2)

    step = int((1.0 - overlap) * window)
    step = max(1, step)

    number_windows = int((Ns - window) / step + 1)
    # pre-allocation of memory
    fdata = numpy.zeros((number_windows, Ne, number_freqs), dtype=complex)
    win = numpy.hanning(window)

    # Loop on all frequencies
    for window_ix in range(int(number_windows)):
        # time markers to select the data
        # marker of the beginning of the time window
        t1 = int(window_ix * step)
        # marker of the end of the time window
        t2 = int(t1 + window)
        # select current window and apodize it
        cdata = X[:, t1:t2] * win

        # FFT calculation
        fdata[window_ix, :, :] = numpy.fft.fft(
            cdata, n=window, axis=1)[:, 0:number_freqs]

    # Adjust Frequency range to specified range (in case it is a parameter)
    if fmin is not None:
        f = numpy.arange(0, 1, 1.0 / number_freqs) * (fs / 2.0)
        Fix = (f >= fmin) & (f <= fmax)
        fdata = fdata[:, :, Fix]

    # fdata = fdata.real
    Nf = fdata.shape[2]
    S = numpy.zeros((Ne, Ne, Nf), dtype=complex)
    normval = numpy.linalg.norm(win) ** 2
    for i in range(Nf):
        S[:, :, i] = numpy.dot(fdata[:, :, i].conj().T, fdata[:, :, i]) / (
                number_windows * normval)

    return numpy.abs(S) ** 2


@njit
def empirical_covariance(X, assume_centered=False):
    """Computes the Maximum likelihood covariance estimator


    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).

    """
    X = numpy.asarray(X)

    if X.ndim == 1:
        X = numpy.reshape(X, (1, -1))

    if X.shape[0] == 1:
        raise ValueError("Only one sample available. You may want to reshape your data array")

    if assume_centered:
        covariance = numpy.dot(X.T, X) / X.shape[0]
    else:
        covariance = numpy.cov(X.T, bias=1)

    if covariance.ndim == 0:
        raise ValueError("Empty covariance.")
    return covariance


@njit
def oas(X, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
      If True, data will not be centered before computation.
      Useful to work with data whose mean is significantly equal to
      zero but is not exactly zero.
      If False, data will be centered before computation.

    Returns
    -------
    shrunk_cov : array-like of shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularised (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * numpy.identity(n_features)

    where mu = trace(cov) / n_features

    The formula we used to implement the OAS is slightly modified compared
    to the one given in the article. See :class:`OAS` for more details.
    """
    X = numpy.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        return numpy.array((X ** 2).mean()).reshape((1, 1)), 0.
    if X.ndim == 1:
        X = numpy.reshape(X, (1, -1))
        raise ValueError("Only one sample available. You may want to reshape your data array")
        n_samples = 1
        n_features = X.size
    else:
        n_samples, n_features = X.shape

    emp_cov = empirical_covariance(X, assume_centered=assume_centered)
    mu = numpy.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = numpy.mean(emp_cov ** 2)
    num = alpha + mu ** 2
    den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)

    shrinkage = 1. if den == 0 else min(num / den, 1.)
    shrunk_cov = (1. - shrinkage) * emp_cov

    add_shrink_mu = numpy.zeros((n_features ** 2))
    add_shrink_mu[::n_features + 1] += shrinkage * mu
    add_shrink_mu = add_shrink_mu.reshape((n_features, n_features))
    shrunk_cov += add_shrink_mu

    return shrunk_cov, shrinkage


@njit
def ledoit_wolf(X, assume_centered=False, block_size=1000):
    """Estimates the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split.
        This is purely a memory optimization and does not affect results.

    Returns
    -------
    shrunk_cov : ndarray of shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * numpy.identity(n_features)

    where mu = trace(cov) / n_features
    """
    X = numpy.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        return numpy.array((X ** 2).mean()).reshape((1, 1)), 0.
    if X.ndim == 1:
        X = numpy.reshape(X, (1, -1))
        raise ValueError("Only one sample available. You may want to reshape your data array")
        n_features = X.size
    else:
        _, n_features = X.shape

    # get Ledoit-Wolf shrinkage
    shrinkage = ledoit_wolf_shrinkage(
        X, assume_centered=assume_centered, block_size=block_size)
    emp_cov = empirical_covariance(X, assume_centered=assume_centered)
    trace = numpy.trace(emp_cov)
    mu = trace / n_features
    shrunk_cov = (1. - shrinkage) * emp_cov

    add_shrink_mu = numpy.zeros((n_features ** 2))
    add_shrink_mu[::n_features + 1] += shrinkage * mu
    add_shrink_mu = add_shrink_mu.reshape((n_features, n_features))
    shrunk_cov += add_shrink_mu

    return shrunk_cov, shrinkage


@njit
def ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
    """Estimates the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split.

    Returns
    -------
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * numpy.identity(n_features)

    where mu = trace(cov) / n_features
    """
    X = numpy.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        return 0.
    if X.ndim == 1:
        X = numpy.reshape(X, (1, -1))

    if X.shape[0] == 1:
        raise ValueError("Only one sample available. You may want to reshape your data array")
    n_samples, n_features = X.shape

    # optionally center data
    if not assume_centered:
        X = X - mean_euclid(X)

    # A non-blocked version of the computation is present in the tests
    # in tests/test_covariance.py

    # number of blocks to split the covariance matrix into
    n_splits = int(n_features / block_size)
    X2 = X ** 2
    emp_cov_trace = numpy.sum(X2, axis=0) / n_samples
    mu = numpy.sum(emp_cov_trace) / n_features
    beta_ = 0.  # sum of the coefficients of <X2.T, X2>
    delta_ = 0.  # sum of the *squared* coefficients of <X.T, X>
    # starting block computation
    for i in range(n_splits):
        for j in range(n_splits):
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            beta_ += numpy.sum(numpy.dot(X2.T[rows], X2[:, cols]))
            delta_ += numpy.sum(numpy.dot(X.T[rows], X[:, cols]) ** 2)
        rows = slice(block_size * i, block_size * (i + 1))
        beta_ += numpy.sum(numpy.dot(X2.T[rows], X2[:, block_size * n_splits:]))
        delta_ += numpy.sum(
            numpy.dot(X.T[rows], X[:, block_size * n_splits:]) ** 2)
    for j in range(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        beta_ += numpy.sum(numpy.dot(X2.T[block_size * n_splits:], X2[:, cols]))
        delta_ += numpy.sum(
            numpy.dot(X.T[block_size * n_splits:], X[:, cols]) ** 2)
    delta_ += numpy.sum(numpy.dot(X.T[block_size * n_splits:],
                                  X[:, block_size * n_splits:]) ** 2)
    delta_ /= n_samples ** 2
    beta_ += numpy.sum(numpy.dot(X2.T[block_size * n_splits:],
                                 X2[:, block_size * n_splits:]))
    # use delta_ to compute beta
    beta = 1. / (n_features * n_samples) * (beta_ / n_samples - delta_)
    # delta is the sum of the squared coefficients of (<X.T,X> - mu*Id) / p
    delta = delta_ - 2. * mu * emp_cov_trace.sum() + n_features * mu ** 2
    delta /= n_features
    # get final beta as the min between beta and delta
    # We do this to prevent shrinking more than "1", which would invert
    # the value of covariances
    beta = min(beta, delta)
    # finally get shrinkage
    shrinkage = 0 if beta == 0 else beta / delta
    return shrinkage
