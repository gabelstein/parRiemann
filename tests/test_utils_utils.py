import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal, assert_array_equal
from parriemann.utils.utils import sliding_windows
from sklearn.datasets import make_spd_matrix


def test_sliding_windows():
    data = np.random.rand(12,3)
    assert_array_equal([i.T for i in np.split(data, 6)], sliding_windows(data,2,2))