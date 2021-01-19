import numpy
from numba import njit
from .base import sqrtm, invsqrtm, powm, logm, expm

###############################################################
# geodesic
###############################################################


@njit
def geodesic(A, B, alpha, metric='riemann'):
    """Return the matrix at the position alpha on the geodesic between A and B according to the metric :


    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' , 'logeuclid' , 'euclid'
    :returns: the covariance matrix on the geodesic

    """
    if metric == 'euclid':
        C = geodesic_euclid(A, B, alpha)
    elif metric == 'logeuclid':
        C = geodesic_logeuclid(A, B, alpha)
    elif metric == 'riemann':
        C = geodesic_riemann(A, B, alpha)
    else:
        raise NotImplementedError("Geodesic function not implemented.")

    return C


@njit
def geodesic_riemann(A, B, alpha=0.5):
    """Return the matrix at the position alpha on the riemannian geodesic between A and B  :

    .. math::
            \mathbf{C} = \mathbf{A}^{1/2} \left( \mathbf{A}^{-1/2} \mathbf{B} \mathbf{A}^{-1/2} \\right)^\\alpha \mathbf{A}^{1/2}

    C is equal to A if alpha = 0 and B if alpha = 1

    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :returns: the covariance matrix

    """
    sA = sqrtm(A)
    isA = invsqrtm(A)
    C = numpy.dot(numpy.dot(isA, B), isA)
    D = powm(C, alpha)
    E = numpy.dot(numpy.dot(sA, D), sA)
    return E


@njit
def geodesic_euclid(A, B, alpha=0.5):
    """Return the matrix at the position alpha on the euclidean geodesic between A and B  :

    .. math::
            \mathbf{C} = (1-\\alpha) \mathbf{A} + \\alpha \mathbf{B}

    C is equal to A if alpha = 0 and B if alpha = 1

    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :returns: the covariance matrix

    """
    return (1 - alpha) * A + alpha * B

@njit
def geodesic_logeuclid(A, B, alpha=0.5):
    """Return the matrix at the position alpha on the log euclidean geodesic between A and B  :

    .. math::
            \mathbf{C} =  \exp \left( (1-\\alpha) \log(\mathbf{A}) + \\alpha \log(\mathbf{B}) \\right)

    C is equal to A if alpha = 0 and B if alpha = 1

    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :returns: the covariance matrix

    """
    return expm((1 - alpha) * logm(A) + alpha * logm(B))
