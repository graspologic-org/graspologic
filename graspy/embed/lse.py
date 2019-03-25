# ase.py
# Created by Ben Pedigo on 2018-09-26.
# Email: bpedigo@jhu.edu
import warnings

from .base import BaseEmbed
from .svd import selectSVD
from ..utils import import_graph, to_laplace, get_lcc, is_fully_connected


class LaplacianSpectralEmbed(BaseEmbed):
    r"""
    Class for computing the laplacian spectral embedding of a graph 
    
    The laplacian spectral embedding (LSE) is a k-dimensional Euclidean representation of 
    the graph based on its Laplacian matrix [1]_. It relies on an SVD to reduce the dimensionality
    to the specified k, or if k is unspecified, can find a number of dimensions automatically.

    Parameters
    ----------
    form : {'DAD' (default), 'I-DAD', 'R-DAD'}, optional
        Specifies the type of Laplacian normalization to use.

    n_components : int or None, default = None
        Desired dimensionality of output data. If "full", 
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        ``select_dimension`` using ``n_elbows`` argument.
    
    n_elbows : int, optional, default: 2
        If `n_components=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.

    algorithm : {'randomized' (default), 'full', 'truncated'}, optional
        SVD solver to use:

        - 'randomized'
            Computes randomized svd using 
            ``sklearn.utils.extmath.randomized_svd``
        - 'full'
            Computes full svd using ``scipy.linalg.svd``
        - 'truncated'
            Computes truncated svd using ``scipy.sparse.linalg.svd``
    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or 
        'truncated'. The default is larger than the default in randomized_svd 
        to handle sparse matrices that may have large slowly decaying spectrum.

    check_lcc : bool , optional (defult = True)
        Whether to check if input graph is connected. May result in non-optimal 
        results if the graph is unconnected. If True and input is unconnected,
        a UserWarning is thrown. Not checking for connectedness may result in 
        faster computation.

    regularizer: int, float or None, optional (default=None)
        Constant to be added to the diagonal of degree matrix. If None, average 
        node degree is added. If int or float, must be >= 0. Only used when 
        ``form`` == 'R-DAD'.

    Attributes
    ----------
    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph.
    latent_right_ : array, shape (n_samples, n_components), or None
        Only computed when the graph is directed, or adjacency matrix is assymetric.
        Estimated right latent positions of the graph. Otherwise, None.
    singular_values_ : array, shape (n_components)
        Singular values associated with the latent position matrices.

    See Also
    --------
    graspy.embed.selectSVD
    graspy.embed.select_dimension
    graspy.utils.to_laplace

    Notes
    -----
    The singular value decomposition: 

    .. math:: A = U \Sigma V^T

    is used to find an orthonormal basis for a matrix, which in our case is the Laplacian
    matrix of the graph. These basis vectors (in the matrices U or V) are ordered according 
    to the amount of variance they explain in the original matrix. By selecting a subset of these
    basis vectors (through our choice of dimensionality reduction) we can find a lower dimensional 
    space in which to represent the graph.

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    """

    def __init__(
        self,
        form="DAD",
        n_components=None,
        n_elbows=2,
        algorithm="randomized",
        n_iter=5,
        check_lcc=True,
        regularizer=None,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
            check_lcc=check_lcc,
        )
        self.form = form
        self.regularizer = regularizer

    def fit(self, graph, y=None):
        """
        Fit LSE model to input graph

        By default, uses the Laplacian normalization of the form:

        .. math:: L = D^{-1/2} A D^{-1/2}

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to embed. see graphstats.utils.import_graph

        y : Ignored

        Returns
        -------
        self : returns an instance of self.
        """
        A = import_graph(graph)

        if self.check_lcc:
            if not is_fully_connected(A):
                msg = (
                    "Input graph is not fully connected. Results may not"
                    + "be optimal. You can compute the largest connected component by"
                    + "using ``graspy.utils.get_lcc``."
                )
                warnings.warn(msg, UserWarning)

        L_norm = to_laplace(A, form=self.form, regularizer=self.regularizer)
        self._reduce_dim(L_norm)
        return self
