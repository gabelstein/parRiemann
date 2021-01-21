import numpy
from .base import sqrtm, invsqrtm, logm, expm
from .mean import mean_covariance
from numba import njit, prange


###############################################################
# Tangent Space
###############################################################


@njit(parallel=True)
def tangent_space(covmats, Cref):
    """Project a set of covariance matrices in the tangent space. according to
    the reference point Cref

    :param covmats: np.ndarray
        Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)

    """
    Nt, Ne, Ne = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = numpy.triu_indices_from(Cref)
    flatidx = idx[0] * Ne + idx[1]  # because numba cant handle double indexing
    Nf = int(Ne * (Ne + 1) / 2)
    coeffs = (numpy.sqrt(2) * numpy.triu(numpy.ones((Ne, Ne)), 1) +
              numpy.eye(Ne)).flatten()[flatidx]

    T = numpy.empty((Nt, Nf))
    for index in prange(Nt):
        tmp = numpy.dot(numpy.dot(Cm12, covmats[index]), Cm12)
        tmp = logm(tmp).flatten()[flatidx]
        T[index] = numpy.multiply(coeffs, tmp)
    return T


@njit(parallel=True)
def untangent_space(T, Cref):
    """Project a set of Tangent space vectors back to the manifold.

    :param T: np.ndarray
        the Tangent space , a matrix of Ntrials X (channels * (channels + 1)/2)
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        A set of Covariance matrix, Ntrials X Nchannels X Nchannels

    """
    Nt, Nd = T.shape
    Ne = int((numpy.sqrt(1 + 8 * Nd) - 1) / 2)
    C12 = sqrtm(Cref)

    idx1, idx2 = numpy.triu_indices_from(Cref)
    covmats = numpy.empty((Nt, Ne, Ne))
    for i in prange(Nt):
        for j in prange(Nd):
            covmats[i, idx1[j], idx2[j]] = T[i, j]

    for i in prange(Nt):
        triuc = numpy.triu(covmats[i], 1) / numpy.sqrt(2)
        covmats[i] = (numpy.diag(numpy.diag(covmats[i])) + triuc + triuc.T)
        covmats[i] = expm(covmats[i])
        covmats[i] = numpy.dot(numpy.dot(C12, covmats[i]), C12)

    return covmats


@njit(parallel=True)
def transport(Covs, Cref, metric='riemann'):
    """Parallel transport of two set of covariance matrix.

    """
    Nt, Ne, Ne = Covs.shape
    C = mean_covariance(Covs, metric=metric)
    iC = invsqrtm(C)
    E = sqrtm(numpy.dot(numpy.dot(iC, Cref), iC))
    out = numpy.zeros((Nt, Ne, Ne))
    for index in prange(Nt):
        out[index] = numpy.dot(numpy.dot(E, Covs[index]), E.T)
    return out
