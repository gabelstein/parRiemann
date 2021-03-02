"""Embedding covariance matrices via manifold learning techniques."""

import numpy as np
from scipy.linalg import svd, eigh, solve
from scipy.sparse import csr_matrix, eye
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import spectral_embedding
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold._locally_linear import null_space, barycenter_kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import FLOAT_DTYPES

from parriemann.utils.distance import pairwise_distance, distance
from parriemann.tangentspace import TangentSpace
import umap
from numba import njit


class Embedding(BaseEstimator):
    """Embed SPD matrices into an Euclidean space of smaller dimension.

    It uses Laplacian Eigenmaps [1] to embed SPD matrices into an Euclidean
    space. The basic hypothesis is that high-dimensional data lives in a
    low-dimensional manifold, whose intrinsic geometry can be described
    via the Laplacian matrix of a graph. The vertices of this graph are
    the SPD matrices and the weights of the links are determined by the
    Riemannian distance between each pair of them.

    Parameters
    ----------
    n_components : integer, default: 2
        The dimension of the projected subspace.
    metric : string | dict (default: 'riemann')
        The type of metric to be used for defining pairwise distance between
        covariance matrices.
    eps:  float (default: None)
        The scaling of the Gaussian kernel. If none is given
        it will use the square of the median of pairwise distances between
        points.

    References
    ----------
    [1] M. Belkin and P. Niyogi, "Laplacian Eigenmaps for dimensionality
    reduction and data representation," in Journal Neural Computation,
    vol. 15, no. 6, p. 1373-1396 , 2003

    """

    def __init__(self, n_components=2, metric='riemann', eps=None):
        """Init."""
        self.metric = metric
        self.n_components = n_components
        self.eps = eps

    def _get_affinity_matrix(self, X, eps):

        # make matrix with pairwise distances between points
        distmatrix = pairwise_distance(X, metric=self.metric)

        # determine which scale for the gaussian kernel
        if self.eps is None:
            eps = np.median(distmatrix)**2 / 2

        # make kernel matrix from the distance matrix
        kernel = np.exp(-distmatrix**2 / (4 * eps))

        # normalize the kernel matrix
        q = np.dot(kernel, np.ones(len(kernel)))
        kernel_n = np.divide(kernel, np.outer(q, q))

        return kernel_n

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        affinity_matrix = self._get_affinity_matrix(X, self.eps)
        embd = spectral_embedding(adjacency=affinity_matrix,
                                  n_components=self.n_components,
                                  norm_laplacian=True)

        # normalize the embedding between -1 and +1
        embdn = 2*(embd - embd.min(0)) / embd.ptp(0) - 1

        self.embedding_ = embdn

        return self

    def fit_transform(self, X, y=None):
        """Calculate the coordinates of the embedded points.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)

        """
        self.fit(X)
        return self.embedding_


class UMapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for UMAP for dimensionality reduction on positive definite matrices with a Riemannian metric

    Parameters
    ----------
    metric : str (default 'riemann')
        string code for the metric from .utils.distance
    **kwargs : dict
        arguments to pass to umap.

    """

    def __init__(self, distance_metric='riemann', **kwargs):
        self.distance_metric = distance_metric
        self.umap_args = kwargs
        self.umapfitter = umap.UMAP(
            metric=_umap_metric_helper,
            metric_kwds={'distance_metric': self.distance_metric},
            **kwargs
        )

    def fit(self, X, y):
        Xre = np.reshape(X, (len(X), -1))
        self.umapfitter.fit(Xre, y)
        return self

    def transform(self, X, y=None):
        Xre = np.reshape(X, (len(X), -1))
        X_ = self.umapfitter.transform(Xre)
        return X_

    def fit_transform(self, X, y=None):
        Xre = np.reshape(X, (len(X), -1))
        self.umapfitter.fit(Xre, y)
        X_ = self.umapfitter.transform(Xre)
        return X_

@njit
def _umap_metric_helper(A, B, distance_metric='riemann'):
    dim = int(np.sqrt(len(A)))
    A_ = np.reshape(A, (dim, dim)).astype(np.float64) # umap casts to float32 for some reason, crashing the metric
    B_ = np.reshape(B, (dim, dim)).astype(np.float64)

    return distance(A_, B_, distance_metric)


class RiemannLLE(LocallyLinearEmbedding):
    def __init__(self, distance_metric='riemann', **kwargs):
        self.distance_metric = distance_metric
        super().__init__(**kwargs)

    def _fit_transform(self, X):
        self.embedding_, self.reconstruction_error_ = \
            locally_linear_embedding(
                X=X, n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                eigen_solver=self.eigen_solver, tol=self.tol,
                max_iter=self.max_iter, method=self.method, reg=self.reg, n_jobs=self.n_jobs)

def locally_linear_embedding(
        X, *, n_neighbors, n_components, reg=1e-3, eigen_solver='auto',
        tol=1e-6, max_iter=100, method='standard',
        random_state=None, n_jobs=None):
    """Perform a Locally Linear Embedding analysis on the data.
    Read more in the :ref:`User Guide <locally_linear_embedding>`.
    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.
    n_neighbors : int
        number of neighbors to consider for each point.
    n_components : int
        number of coordinates for the manifold.
    reg : float, default=1e-3
        regularization constant, multiplies the trace of the local covariance
        matrix of the distances.
    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.
    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.
    max_iter : int, default=100
        maximum number of iterations for the arpack solver.
    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        standard : use the standard locally linear embedding algorithm.
                   see reference [1]_
        hessian  : use the Hessian eigenmap method.  This method requires
                   n_neighbors > n_components * (1 + (n_components + 1) / 2.
                   see reference [2]_
        modified : use the modified locally linear embedding algorithm.
                   see reference [3]_
        ltsa     : use local tangent space alignment algorithm
                   see reference [4]_
    random_state : int, RandomState instance, default=None
        Determines the random number generator when ``solver`` == 'arpack'.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.
    n_jobs : int or None, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Returns
    -------
    Y : array-like, shape [n_samples, n_components]
        Embedding vectors.
    squared_error : float
        Reconstruction error for the embedding vectors. Equivalent to
        ``norm(Y - W Y, 'fro')**2``, where W are the reconstruction weights.
    References
    ----------
    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
        linear embedding techniques for high-dimensional data.
        Proc Natl Acad Sci U S A.  100:5591 (2003).
    .. [3] Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
        Embedding Using Multiple Weights.
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
    .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
        dimensionality reduction via tangent space alignment.
        Journal of Shanghai Univ.  8:406 (2004)
    """
    if eigen_solver not in ('auto', 'arpack', 'dense'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

    if method not in ('standard', 'hessian', 'modified', 'ltsa'):
        raise ValueError("unrecognized method '%s'" % method)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError("output dimension must be less than or equal "
                         "to input dimension")
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples, "
            " but n_samples = %d, n_neighbors = %d" %
            (N, n_neighbors)
        )

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    M_sparse = (eigen_solver != 'dense')

    n_samples = nbrs.n_samples_fit_
    indices = nbrs.kneighbors(X, return_distance=False)[:, 1:]

    X = check_array(X, dtype=FLOAT_DTYPES)
    Y = X
    indices = check_array(indices, dtype=int)

    n_samples, n_neighbors = indices.shape
    assert X.shape[0] == n_samples

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, ind in enumerate(indices):
        A = Y[ind]
        C = A - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::n_neighbors + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    data = B

    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    W = csr_matrix((data.ravel(), indices.ravel(), indptr),
                      shape=(n_samples, n_samples))

    # we'll compute M = (I-W)'(I-W)
    # depending on the solver, we'll do this differently
    if M_sparse:
        M = eye(*W.shape, format=W.format) - W
        M = (M.T * M).tocsr()
    else:
        M = (W.T * W - W.T - W).toarray()
        M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I
    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                      tol=tol, max_iter=max_iter, random_state=random_state)

