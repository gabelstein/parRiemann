from numba import njit, prange
import numpy as np


def cv_split_by_labels(labels, cv):
    all_idx = np.arange(len(labels))
    idx_split, label_split = _split_data_by_class(all_idx, labels)

    classes = np.unique(labels)
    idx = [[] for i in range(cv)]

    for i in range(len(classes)):
        tmp_idx = label_split==classes[i]
        class_idx = idx_split[tmp_idx]
        for tmp in class_idx:
            idx[i % cv].append(tmp)
    return idx


@njit
def cv_idx(data):
    cv = len(data)
    fullidx = np.arange(np.sum(data))
    train = list()
    test = list()
    start = 0
    end = 0
    for i in range(cv):
        end += data[i]
        test.append(fullidx[start:end])
        train.append(np.delete(fullidx, fullidx[start:end]))
        start += data[i]
    return train, test


@njit
def labeled_windows(data, label, window_size=200, shift=20, adjust_class_size=True):
    """
    This function creates sliding windows for multivariate data.
    ----------
    data : ndarray
        multivariate data to turn into sliding windows based on the labels provided.
        Dimension: nSamples x nChannels
    label : ndarray
        labels for the data.
    intlength : int,
        lenght of the sliding windows. The default is 200.
    step_size : int,
        size of step until next sliding window is calculated. The default is 20. Only produces exact overlapping windows
        if intlength is an integer multiple of step_size.

    Returns
    -------
    X,y : ndarray, ndarray
        data turned into sliding windows and corresponding label for each data point.
        Dimension: nSamples x nChannels x intlength
    """
    if len(data) != len(label):
        raise ValueError("Data and labels must have same length.")

    Nt, Ne = data.shape
    if adjust_class_size:
        ratios = _adjust_classes(label)

    datasplit, labelsplit = _split_data_by_class(data, label)
    n_chunks = len(labelsplit)
    X = list()
    y = list()

    for i in range(n_chunks):
        if adjust_class_size:
            step = int(shift*ratios[labelsplit[i]])
        else:
            step = shift
        X.append(sliding_windows(datasplit[i], window_size, step))
        y.append(np.ones(X[i].shape[0]) * labelsplit[i])

    _sub_len = np.array([len(X[i]) for i in range(n_chunks)])
    n_windows = np.sum(_sub_len)
    X_ = np.zeros((n_windows, Ne, window_size))
    y_ = np.zeros(n_windows)
    start = 0
    for i in range(n_chunks):
        sub_len = _sub_len[i]
        X_[start:start + sub_len] = X[i]
        y_[start:start + sub_len] = y[i]
        start += sub_len

    #X = np.array([((x.T-x.mean(axis=1)).T/x.std(axis=1)[:,None]) for x in X])

    return X_, y_


@njit(parallel=True)
def sliding_windows(data, window_size, shift):
    Nt, Ne = data.shape
    Nw = (data.shape[0] - window_size) // shift + 1
    _dat = np.zeros((Nw, Ne, window_size))
    for i in prange(Nw):
        _dat[i] = data[i * shift:i * shift + window_size].T
    return _dat


@njit
def _adjust_classes(labels):
    classes = np.unique(labels)
    Nc = classes.size
    counts = np.zeros(Nc)
    for i in range(Nc):
        counts[i] = np.count_nonzero(labels == classes[i])
    min_class = np.min(counts)
    class_ratios = counts / min_class
    ratio_dict = dict()
    for i in range(Nc):
        ratio_dict[classes[i]] = class_ratios[i]
    return ratio_dict


@njit
def _split_data_by_class(data, label):
    splitter = np.argwhere(np.diff(label) != 0)[:, 0] + 1
    datasplit = _array_split(data, splitter)
    _labelsplit = _array_split(label, splitter)
    labelsplit = np.array([x[0] for x in _labelsplit])
    return datasplit, labelsplit


@njit#throws errors if parallel=True \_('_')_/
def _array_split(ary, indices_or_sections):
    """
    Split an array into multiple sub-arrays.

    Please refer to the ``split`` documentation.  The only difference
    between these functions is that ``array_split`` allows
    `indices_or_sections` to be an integer that does *not* equally
    divide the axis. For an array of length l that should be split
    into n sections, it returns l % n sub-arrays of size l//n + 1
    and the rest of size l//n.

    See Also
    --------
    split : Split array into multiple sub-arrays of equal size.

    """
    Ntotal = ary.shape[0]

    Nsections = len(indices_or_sections) + 1
    div_points = [0] + list(indices_or_sections) + [Ntotal]

    sub_arys = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(ary[st:end])

    return sub_arys


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
