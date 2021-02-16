import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal, assert_array_equal
from parriemann.utils.utils import _sliding_windows, cv_idx, cv_split_by_labels, SlidingWindow


def test_sliding_windows():
    data = np.random.rand(12, 3)
    assert_array_equal([i.T for i in np.split(data, 6)], _sliding_windows(data, 2, 2))


def test_labeled_windows():
    data = np.random.rand(20, 3)
    labels = np.concatenate((np.repeat(1, 8), np.repeat(0, 12)))
    clf = SlidingWindow(4, 2)
    clf.transform(data, labels)


def test_labeled_windows_raise():
    data = np.random.rand(10, 3)
    labels = np.concatenate((np.repeat(1,10), np.repeat(0,10)))
    clf = SlidingWindow(4, 2)
    assert_raises(ValueError, clf.transform, data, labels)


def test_cv_idx():
    cv_idx(np.array([5, 5, 4, 5]))


def test_cv_split_by_labels():
    cv_idx(np.array([5, 5, 4, 5]))
    cv_split_by_labels(labels = np.concatenate((np.repeat(1,10), np.repeat(0,10), np.repeat(1,10), np.repeat(0,10))), cv=4)
