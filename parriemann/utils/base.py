import numpy
from numba import njit

@njit
def _check_mat(Ci):
    if (Ci.dtype is numpy.dtype(numpy.float64) or Ci.dtype is numpy.dtype(numpy.float32)) and not numpy.isfinite(
            Ci).all():
        raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error.")


@njit
def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    _check_mat(Ci)
    eigvals, eigvects = numpy.linalg.eigh(Ci)
    eigvals = numpy.diag(operator(eigvals))
    Out = numpy.dot(numpy.dot(eigvects, eigvals), eigvects.T)
    return Out


@njit
def sqrtm(Ci):
    """Return the matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix square root

    """
    return _matrix_operator(Ci, numpy.sqrt)


@njit
def logm(Ci):
    """Return the matrix logarithm of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix logarithm

    """
    return _matrix_operator(Ci, numpy.log)


@njit
def expm(Ci):
    """Return the matrix exponential of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix exponential

    """
    return _matrix_operator(Ci, numpy.exp)


@njit
def invsqrtm(Ci):
    """Return the inverse matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root

    """
    isqrt = lambda x: 1. / numpy.sqrt(x)
    return _matrix_operator(Ci, isqrt)


@njit
def powm(Ci, alpha):
    """Return the matrix power :math:`\\alpha` of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :param alpha: the power to apply
    :returns: the matrix power

    """
    _check_mat(Ci)
    eigvals, eigvects = numpy.linalg.eigh(Ci)
    eigvals = numpy.diag(eigvals**alpha)
    Out = numpy.dot(numpy.dot(eigvects, eigvals), eigvects.T)
    return Out
