import mne as mne
import scipy
from numba import njit, prange
import numpy as np
from scipy.stats import zscore
from sklearn.base import TransformerMixin, BaseEstimator
import math
from scipy.signal import butter, lfilter


class ZScore(BaseEstimator, TransformerMixin):
    """Calculate z-score.

        ----------
        mean : str
            Rereferencing method.
        line_noise : int
            Line noise
        """

    def __init__(self, mean=None, std=None, axis=0):
        self.mean = mean
        self.std = std
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return zscore(X, axis=self.axis)


class Rereferencing(BaseEstimator, TransformerMixin):
    """Rereferencing.

        ----------
        method : str
            Rereferencing method.
        reref_idx : list
            list of rereferencing electrodes for each channel for custom rereferencing.
        """

    def __init__(self, method, reref_idx=None):
        self.method = method
        self.reref_idx = reref_idx
        self.reref_function = self.reref_methods[method]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.reref_function(self, X)

    def fit_transform(self, X, y=None):
        return self.reref_function(self, X)

    def reref_avg(self, X, reref_idx=None):
        return (X.T - X.mean(axis=1)).T

    def reref_bipolar(self, X, reref_idx=None):
        X_ = np.roll(X, 1)
        return X - X_

    def reref_custom(self, X):
        X_T = X.T
        for i, x in enumerate(X_T):
            x -= np.mean(X[self.reref_idx[i]], axis=1)
        return X_T.T

    reref_methods = {
        'average': reref_avg,
        'bipolar': reref_bipolar,
        'custom': reref_custom
    }


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
        self.filters = _calc_band_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.filter_len,
                                               self.l_trans_bandwidth,
                                               self.h_trans_bandwidth)

    def fit(self, X, y=None):
        self.filters = _calc_band_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.filter_len,
                                               self.l_trans_bandwidth,
                                               self.h_trans_bandwidth)
        return self

    def transform(self, X, y=None):
        X_ = _apply_filter(X, self.filters)
        return X_

    def fit_transform(self, X, y=None):
        self.filters = _calc_band_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.filter_len,
                                               self.l_trans_bandwidth,
                                               self.h_trans_bandwidth)
        X_ = _apply_filter(X, self.filters)
        return X_


def _calc_band_filters(f_ranges, sample_rate, filter_len="1000ms", l_trans_bandwidth=4, h_trans_bandwidth=4):
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


def _apply_filter(dat_, filter_fun):
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


class ButterBandPassFilter(BaseEstimator, TransformerMixin):
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

    def __init__(self, filter_bands, sample_rate, order=5):
        self.filter_bands = filter_bands
        self.sample_rate = sample_rate
        self.order = order
        self.filters = _calc_butter_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.order)

    def fit(self, X, y=None):
        self.filters = _calc_butter_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.order)
        return self

    def transform(self, X, y=None):
        X_ = _apply_butter_filter(X, self.filters)
        return X_

    def fit_transform(self, X, y=None):
        self.filters = _calc_butter_filters(self.filter_bands,
                                               self.sample_rate,
                                               self.order)
        X_ = _apply_butter_filter(X, self.filters)
        return X_



def butter_bandpass(filter_band, fs, order=5):
    nyq = 0.5 * fs
    high = filter_band[1] / nyq
    low = filter_band[0] / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _calc_butter_filters(filters, sample_rate, order=5):
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

    for f_range in filters:
        h = butter_bandpass(f_range, sample_rate, order)
        filter_fun.append(h)

    return np.array(filter_fun)

def _apply_butter_filter(dat_, filter_fun):
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

    for filt in filter_fun:
        for ch in dat_.T:
            filtered.append(lfilter(filt[0], filt[1], ch))

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

    def fit(self, X, y=None):
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


class SlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, step_size, adjust_class_size=True, allow_overlap=False, overlap_handler='avg', verbose=False):
        self.window_size = window_size
        self.step_size = step_size
        self.adjust_class_size = adjust_class_size
        self.allow_overlap = allow_overlap
        self.overlap_handler = overlap_handler
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is None:
            windows = _sliding_windows(X, self.window_size, self.step_size)
            if self.verbose:
                print(f'# sliding windows: {len(windows)}')
            return windows
        else:
            X_, y_ = _slide(X, y, self.window_size, self.step_size, self.adjust_class_size, self.allow_overlap,
                            self.overlap_handler)
            if self.verbose:
                print(f'# sliding windows: {len(X_)}')
            return np.array(X_), np.array(y_)

    def fit_transform(self, X, y):
        if y is None:
            windows = _sliding_windows(X, self.window_size, self.step_size)
            if self.verbose:
                print(f'# sliding windows: {len(windows)}')
            return windows
        else:
            X_, y_ = _slide(X, y, self.window_size, self.step_size, self.adjust_class_size, self.allow_overlap,
                            self.overlap_handler)
            if self.verbose:
                print(f'# sliding windows: {len(X_)}')
            return np.array(X_), np.array(y_)


# TODO: rewrite for numba
def _slide(data, label, window_size, step_size, adjust_class_size, allow_overlap=False, overlap_handler='avg'):
    if len(data) != len(label):
        raise ValueError("Data and labels must have same length.")
    Nt = (len(data) - window_size) // step_size + 1
    Nc = data.shape[1]
    Nw = window_size
    X_ = np.zeros((Nt, Nc, Nw))
    y_ = np.zeros((Nt))
    if allow_overlap:
        for i in range((len(data) - window_size) // step_size):
            X_[i] = data[i * step_size:i * step_size + window_size].T
            if overlap_handler == 'avg':
                y_[i] = label[i * step_size:i * step_size + window_size].mean() * math.copysign(1, np.diff(
                    label[i * step_size:i * step_size + window_size]).sum())
            elif overlap_handler == 'last':
                y_[i] = label[i * step_size:i * step_size + window_size][-1]
        return X_, y_

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
            subsamples = np.round(np.linspace(0, len(y_c) - 1, min_count)).astype(int)
            y_adj.extend(y_c[subsamples])
            X_adj.extend(X_c[subsamples])
        return np.array(X_adj), np.array(y_adj)

    return X_, y_

@njit(parallel=True)
def _sliding_windows(data, window_size, shift):
    Nw = (data.shape[0] - window_size) // shift + 1
    if Nw < 0:
        print(f"Array too short for window.")
        return

    if data.ndim == 1:
        _dat = np.zeros((Nw, window_size))
    if data.ndim == 2:
        Nt, Ne = data.shape
        _dat = np.zeros((Nw, Ne, window_size))
    for i in range(Nw):
        _dat[i] = data[i * shift:i * shift + window_size].T
    return _dat