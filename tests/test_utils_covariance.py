from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_equal, assert_raises
import numpy as np
from scipy.signal import coherence as coh_sp
from parriemann.utils.covariance import (covariances, covariances_EP, eegtocov,
                                         cospectrum, coherence)
from pyriemann.utils.covariance import covariances as cov2


def test_covariances():
    """Test covariance for multiple estimator"""
    x = np.random.randn(2, 3, 100)
    cov = covariances(x)
    cov = covariances(x, estimator='oas')
    assert_array_almost_equal(cov, cov2(x, estimator='oas'))
    cov = covariances(x, estimator='lwf')
    assert_array_almost_equal(cov, cov2(x, estimator='lwf'))
    cov = covariances(x, estimator='scm')
    assert_array_almost_equal(cov, cov2(x, estimator='scm'))
    cov = covariances(x, estimator='corr')
    assert_array_almost_equal(cov, cov2(x, estimator='corr'))
    assert_raises(ValueError, covariances, x, estimator='truc')


def test_covariances_EP():
    """Test covariance_EP for multiple estimator"""
    x = np.random.randn(2, 3, 100)
    p = np.random.randn(3, 100)
    cov = covariances_EP(x, p)
    cov = covariances_EP(x, p, estimator='oas')
    cov = covariances_EP(x, p, estimator='lwf')
    cov = covariances_EP(x, p, estimator='scm')
    cov = covariances_EP(x, p, estimator='corr')


def test_covariances_eegtocov():
    """Test eegtocov"""
    x = np.random.randn(1000, 3)
    cov = eegtocov(x)
    assert_equal(cov.shape[1], 3)


def test_covariances_cospectrum():
    """Test cospectrum"""
    x = np.random.randn(3, 1000)
    cospectrum(x)
    cospectrum(x, fs=128, fmin=2, fmax=40)


def test_covariances_coherence():
    """Test coherence"""
    x = np.random.randn(2, 2048)
    coh = coherence(x, fs=128, window=256)

    _, coh2 = coh_sp(
        x[0],
        x[1],
        fs=128,
        nperseg=256,
        noverlap=int(0.75 * 256),
        window='hanning',
        detrend=False)
    assert_array_almost_equal(coh[0, 1], coh2[:-1], 0.1)
