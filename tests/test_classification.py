import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises
from parriemann.classification import (MDM, FgMDM, KNearestNeighbor,
                                       TSclassifier, RSVC, KNNRegression)


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose."""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats


def test_MDM_init():
    """Test init of MDM"""
    MDM(metric='riemann')

    # Should raise if metric not string or dict
    assert_raises(TypeError, MDM, metric=42)

    # Should raise if metric is not contain bad keys
    assert_raises(KeyError, MDM, metric={'universe': 42})

    # should works with correct dict
    MDM(metric={'mean': 'riemann', 'distance': 'logeuclid'})


def test_MDM_fit():
    """Test Fit of MDM"""
    covset = generate_cov(100, 3)
    labels = np.array([0, 1]).repeat(50)
    mdm = MDM(metric='riemann')
    mdm.fit(covset, labels)


def test_MDM_predict():
    """Test prediction of MDM"""
    covset = generate_cov(100, 3)
    labels = np.array([0, 1]).repeat(50)
    mdm = MDM(metric='riemann')
    mdm.fit(covset, labels)
    mdm.predict(covset)

    # test fit_predict
    mdm = MDM(metric='riemann')
    mdm.fit_predict(covset, labels)

    # test transform
    mdm.transform(covset)

    # predict proba
    mdm.predict_proba(covset)

    # test n_jobs
    mdm = MDM(metric='riemann', n_jobs=2)
    mdm.fit(covset, labels)
    mdm.predict(covset)


def test_KNN():
    """Test KNearestNeighbor"""
    covset = generate_cov(30, 3)
    labels = np.array([0, 1, 2]).repeat(10)

    knn = KNearestNeighbor(1, metric='riemann')
    knn.fit(covset, labels)
    preds = knn.predict(covset)
    assert_array_equal(labels, preds)


def test_TSclassifier():
    """Test TS Classifier"""
    covset = generate_cov(40, 3)
    labels = np.array([0, 1]).repeat(20)

    assert_raises(TypeError, TSclassifier, clf='666')
    clf = TSclassifier()
    clf.fit(covset, labels)
    clf.predict(covset)
    clf.predict_proba(covset)


def test_FgMDM_init():
    """Test init of FgMDM"""
    FgMDM(metric='riemann')

    # Should raise if metric not string or dict
    assert_raises(TypeError, FgMDM, metric=42)

    # Should raise if metric is not contain bad keys
    assert_raises(KeyError, FgMDM, metric={'universe': 42})

    # should works with correct dict
    FgMDM(metric={'mean': 'riemann', 'distance': 'logeuclid'})


def test_FgMDM_predict():
    """Test prediction of FgMDM"""
    covset = generate_cov(100, 3)
    labels = np.array([0, 1]).repeat(50)
    fgmdm = FgMDM(metric='riemann')
    fgmdm.fit(covset, labels)
    fgmdm.predict(covset)
    fgmdm.transform(covset)


def test_RSVC():
    """Test RSVC"""
    covset = np.concatenate((generate_cov(20, 8), np.array([np.eye(8) for i in range(20)])))
    labels = np.array([0, 1]).repeat(20)
    rsvc = RSVC(C=100)
    rsvc.fit(covset, labels)
    preds = rsvc.predict(covset)
    assert_array_equal(labels, preds)


def test_KNNRegression():
    """Test KNNRegression"""
    covset = np.concatenate((generate_cov(20, 8), np.array([np.eye(8) for i in range(20)])))
    labels = np.array([0., 1.]).repeat(20)
    knnreg = KNNRegression(n_neighbors=2)
    knnreg.fit(covset, labels)
    preds = knnreg.predict(covset)
    assert_array_equal(labels, preds)
