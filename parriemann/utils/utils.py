import mne as mne
import scipy
from numba import njit, prange
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import math


class SlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, step_size, adjust_class_size=True):
        self.window_size = window_size
        self.step_size = step_size
        self.adjust_class_size = adjust_class_size

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        if y is None:
            return _sliding_windows(X,self.window_size,self.step_size)
        else:
            X_, y_ = _slide(X, y, self.window_size, self.step_size, self.adjust_class_size)
            return np.array(X_), np.array(y_)

    def fit_transform(self, X, y):
        if y is None:
            return _sliding_windows(X,self.window_size,self.step_size)
        else:
            X_, y_ = _slide(X, y, self.window_size, self.step_size, self.adjust_class_size)
            return np.array(X_), np.array(y_)

    def _labeled_windows(self, data, label):
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
        if self.adjust_class_size:
            ratios = _adjust_classes(label)

        datasplit, labelsplit = _split_data_by_class(data, label)
        n_chunks = len(labelsplit)
        X = list()
        y = list()

        for i in range(n_chunks):
            if self.adjust_class_size:
                step = int(self.step_size * ratios[labelsplit[i]])
            else:
                step = self.step_size
            X.append(_sliding_windows(datasplit[i], self.window_size, step))
            y.append(np.ones(X[i].shape[0]) * labelsplit[i])

        _sub_len = np.array([len(X[i]) for i in range(n_chunks)])
        n_windows = np.sum(_sub_len)
        X_ = np.zeros((n_windows, Ne, self.window_size))
        y_ = np.zeros(n_windows)
        start = 0
        for i in range(n_chunks):
            sub_len = _sub_len[i]
            X_[start:start + sub_len] = X[i]
            y_[start:start + sub_len] = y[i]
            start += sub_len

        return X_, y_


# TODO: rewrite for numba
def _slide(data, label, window_size, step_size, adjust_class_size):
    if len(data) != len(label):
        raise ValueError("Data and labels must have same length.")
    Nt = (len(data) - window_size) // step_size + 1
    Nc = data.shape[1]
    Nw = window_size
    X_ = np.zeros((Nt, Nc, Nw))
    y_ = np.zeros((Nt))

    for i in range(Nt):
        X_[i] = data[i * step_size:i * step_size + window_size].T
        tmpy = label[i * step_size:i * step_size + window_size]
        if np.unique(tmpy).size == 1:
            y_[i] = tmpy.mean() * math.copysign(1, np.diff(tmpy).sum())
        else:
            y_[i] = np.inf

    X_ = X_[np.isfinite(y_)]
    y_ = y_[np.isfinite(y_)]

    if adjust_class_size:
        classes = np.unique(y_)
        Nc = classes.size
        counts = []
        for i in range(Nc):
            counts.append(np.count_nonzero(y_ == classes[i]))
        min_count = np.min(np.array(counts))
        X_adj = []
        y_adj = []
        for i in range(Nc):
            y_c = y_[y_ == classes[i]]
            X_c = X_[y_ == classes[i]]
            subsamples = np.random.choice(len(y_c), min_count, replace=False)
            y_adj.extend(y_c[subsamples])
            X_adj.extend(X_c[subsamples])
        return np.array(X_adj), np.array(y_adj)

    return X_, y_


@njit(parallel=True)
def _sliding_windows(data, window_size, shift):
    Nw = (data.shape[0] - window_size) // shift + 1
    if data.ndim == 1:
        _dat = np.zeros((Nw, window_size))
    if data.ndim == 2:
        Nt, Ne = data.shape
        _dat = np.zeros((Nw, Ne, window_size))
    for i in range(Nw):
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


@njit
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
    Ntotal = len(ary)

    Nsections = len(indices_or_sections) + 1
    div_points = [0] + list(indices_or_sections) + [Ntotal]

    sub_arys = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(ary[st:end])

    return sub_arys


#@njit
def cv_split_by_labels(labels, cv):
    all_idx = np.arange(len(labels))
    idx_split, label_split = _split_data_by_class(all_idx, labels)

    classes = np.unique(labels)
    idx = [[] for i in range(cv)]

    for i in range(len(classes)):
        tmp_idx = (label_split == classes[i])
        class_idx = []
        for k in range(len(tmp_idx)):
            if tmp_idx[k]:
                class_idx.append(idx_split[k])
        for tmp in class_idx:
            res = i % cv
            idx[res].append(tmp)
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


class BandPassFilter(BaseEstimator, TransformerMixin):
    """Band Pass filtering.

    ----------
    filter_bands : list
        bands to filter signal with
    sample_rate : int
        Signal sample rate
    filter_len : int,
        lenght of the filter. The default is 1001.
    l_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.
    h_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.
    """

    def __init__(self, filter_bands, sample_rate, filter_len='1000ms', l_trans_bandwidth=4, h_trans_bandwidth=4):
        self.filter_bands = filter_bands
        self.sample_rate = sample_rate
        self.filter_len = filter_len
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.filters = self._calc_band_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.filter_len,
                                               self.l_trans_bandwidth,
                                               self.h_trans_bandwidth)

    def fit(self, X, y):
        self.filters = self._calc_band_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.filter_len,
                                               self.l_trans_bandwidth,
                                               self.h_trans_bandwidth)
        return self

    def transform(self, X, y=None):
        X_ = self._apply_filter(X, self.filters)
        return X_

    def fit_transform(self, X, y):
        self.filters = self._calc_band_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.filter_len,
                                               self.l_trans_bandwidth,
                                               self.h_trans_bandwidth)
        X_ = self._apply_filter(X, self.filters)
        return X_

    def _calc_band_filters(self, f_ranges, sample_rate, filter_len="1000ms", l_trans_bandwidth=4, h_trans_bandwidth=4):
        """
        This function returns for the given frequency band ranges filter coefficients with with length "filter_len"
        Thus the filters can be sequentially used for band power estimation
        Parameters
        ----------
        f_ranges : TYPE
            DESCRIPTION.
        sample_rate : float
            sampling frequency.
        filter_len : int,
            lenght of the filter. The default is 1001.
        l_trans_bandwidth : TYPE, optional
            DESCRIPTION. The default is 4.
        h_trans_bandwidth : TYPE, optional
            DESCRIPTION. The default is 4.
        Returns
        -------
        filter_fun : array
            filter coefficients stored in rows.
        """
        filter_fun = []

        for a, f_range in enumerate(f_ranges):
            h = mne.filter.create_filter(None, sample_rate, l_freq=f_range[0], h_freq=f_range[1],
                                         fir_design='firwin', l_trans_bandwidth=l_trans_bandwidth,
                                         h_trans_bandwidth=h_trans_bandwidth, filter_length=filter_len, verbose=False)

            filter_fun.append(h)

        return np.array(filter_fun)

    def _apply_filter(self, dat_, filter_fun):
        """
        For a given channel, apply previously calculated filters

        Parameters
        ----------
        dat_ : array (ns,)
            segment of data at a given channel and downsample index.
        sample_rate : float
            sampling frequency.
        filter_fun : array
            output of calc_band_filters.
        line_noise : int|float
            (in Hz) the line noise frequency.
        seglengths : list
            list of ints with the leght to which variance is calculated.
            Used only if variance is set to True.
        variance : bool,
            If True, return the variance of the filtered signal, else
            the filtered signal is returned.
        Returns
        -------
        filtered : array
            if variance is set to True: (nfb,) array with the resulted variance
            at each frequency band, where nfb is the number of filter bands used to decompose the signal
            if variance is set to False: (nfb, filter_len) array with the filtered signal
            at each freq band, where nfb is the number of filter bands used to decompose the signal
        """
        filtered = []

        for filt in range(filter_fun.shape[0]):
            for ch in dat_.T:
                filtered.append(scipy.signal.convolve(ch, filter_fun[filt, :], mode='same'))

        return np.array(filtered).T


class NotchFilter(BaseEstimator, TransformerMixin):
    """Notch filtering.

    ----------
    sample_rate : int
        Signal sample rate.
    line_noise : int
        Line noise
    """

    def __init__(self, line_noise, sample_rate):
        self.sample_rate = sample_rate
        self.line_noise = line_noise

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        X_ = self._notch_filter(X)
        return X_

    def fit_transform(self, X, y=None):
        X_ = self._notch_filter(X)
        return X_

    def _notch_filter(self, dat_):
        dat_notch_filtered = mne.filter.notch_filter(x=dat_.T, Fs=self.sample_rate, trans_bandwidth=7,
                                                     freqs=np.arange(self.line_noise, 4 * self.line_noise,
                                                                     self.line_noise),
                                                     fir_design='firwin', verbose=False, notch_widths=1,
                                                     filter_length=dat_.shape[0] - 1)
        return dat_notch_filtered.T


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
