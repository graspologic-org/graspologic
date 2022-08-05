import numpy as np 
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
import time
from .base import BaseAlign

class SeededProcrustes(BaseAlign):

    """
    Matches two datasets using an orthogonal matrix and a '2d' matrix of seeds. Unlike
    :class:`~graspologic.align.OrthogonalProcrustes` or :class:`~graspologic.align.SeedlessProcrustes`, 
    this requires only a partial matching between entries. It can even be used in 
    the settings where the two datasets do not have the same number of entries.

    In the graph setting, it is used to align the embeddings of two different
    graphs, which requires some simultaneous inference task and partial 1-1
    matching between the vertices of the two graphs.

    Attributes
    ----------
    Q_ : array, size (d, d)
        Final orthogonal matrix, used to modify ``X``.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    .. [2] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.

    .. [3] Agterberg, J., Tang, M., Priebe., C. E. (2020).
        "Nonparametric Two-Sample Hypothesis Testing for Random Graphs with Negative and Repeated Eigenvalues"
        arXiv:2012.09828

    .. [4] Agterberg, J., Tang, M., Priebe., C. E. (2020).
        "On Two Distinct Sources of Nonidentifiability in Latent Position Random Graph Models"
        arXiv:2003.14250

    Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    -----
    The goal of this process is to find a correspondence between
    the vertices of two datasets as well as the orthogonal alignment 
    between them. If the two datasets are represented with 
    matrices :math:`X \in M_{n, d}` and :math:`Y \in M_{m, d}`, 
    then the correspondence is a matrix :math:`P \in M_{n, m}` that is 
    soft assignment matrix (that is, its rows sum to :math:`1/n`, and columns 
    sum to :math:`1/m`) and the orthogonal alignment is an 
    orthogonal matrix :math:`Q \in M_{d, d}`. An orthogonal
    matrix is any matrix that satisfies :math:`Q^T Q = Q Q^T = I`. 
    The global objective function is :math:`|| X Q - P Y ||_F`.
    
    Note that both :math:`X` and :math:`PY` are matrices in :math:`M_{n, d}`.
    Thus, if one knew :math:`P`, it would be simple to obtain an estimate for
    :math:`Q`, using the regular orthogonal procrustes. On the other hand, if
    :math:`Q` was known, then :math:`XQ` and :math:`Y` could be thought of
    distributions over a finite number of masses, each with weight :math:`1/n`
    or :math:`1/m`, respectively. These distributions could be "matched" via
    solving an optimal transport problem.

    However, both :math:`Q` and :math:`P` are simultaneously unknown here, so
    the algorithm performs a sequence of alternating steps, obtaining
    iteratively improving estimates of :math:`Q` and :math:`P`, similarly to an
    expectation-maximization (EM) procedure. It is not known whether this
    procedure is formally an EM, but the analogy can be drawn as follows: after
    obtaining an initial guess of of :math:`\hat{Q}_{0}`, obtaining an
    assignment matrix :math:`\hat{P}_{i+1} | \hat{Q}_{i}` ("E-step") is done by
    solving an optimal transport problem via Sinkhorn algorithm, whereas
    obtaining an orthogonal alignment matrix :math:`Q_{i+1} | P_{i}` 
    is done via regular orthogonal procurstes. These steps are further simplified
    by the use of a seeded matrix that contains rows of seeds that align a set of 
    indeses in ``X`` and ``Y``.
    """

    def __init__(
        self,
        X: ,
        Y,
        seeds
    ):


    def fit(
        self,
        X,
        Y,
        seeds,
        verbose=False,
    )->"SeededProcrustes": 

        """
        Overrides the fit method from the parent class ``BaseAlign`` to find
        a value for ``Q`` given two datasets, ``X`` and ``Y``, and a matrix of seeds.
        
        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Dataset to be mapped to ``Y``, must have same number of dimensions
            (axis 1) as ``Y``.

        Y : np.ndarray, shape (m, d)
            Target dataset, must have same number of dimensions (axis 1) as ``X``.

        seeds : np.ndarray, shape (?,2)
            Matrix of pairs, demonstrates the relation between rows of ``X`` and ``Y``.
            The first column pertains to the ``X`` component of the pairs, and the second column 
            pertains to the ``Y`` component. The pairs are stored in the rows of ``seeds``,
            each row containing an ``X`` value and a corresponding ``Y`` value.

        verbose : bool
            Degree to which the code records and produces feedback when run.

        Returns
        -------
        self : returns an instance of self
        """

        n = len(X[0])
        init_Q = _find_Q_from_pairs(X,Y,seeds)
        procruster = SeedlessProcrustes(
            init="custom",
            initial_Q=init_Q,
            optimal_transport_eps=1.0,
            optimal_transport_num_reps=100,
            iterative_num_reps=10,
        )
        currtime = time.time()
        X_mapped = procruster.fit_transform(X,Y) 
        self.Q_ = procruster.Q_

        if verbose > 1:
            print(f"{time.time() - currtime:.3f} seconds elapsed for SeedlessProcrustes.")
        X = (X_mapped[:n], X_mapped[n:])
        return self

def _find_Q_from_pairs(X: np.ndarray,Y: np.ndarray,seeds: np.ndarray) -> np.ndarray:
    """
    This function uses the seeds matrix to sort the matrices X and Y 
    by their paired indices. 

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Dataset to be mapped to ``Y``, must have same number of dimensions
        (axis 1) as ``Y``.

    Y : np.ndarray, shape (m, d)
        Target dataset, must have same number of dimensions (axis 1) as ``X``.
    
    seeds : np.ndarray, shape (?, 2)
        Matrix of pairs between rows of ``X`` and ``Y``

    Returns
    -------
    op.Q_ : a call of the ``Q_`` attribute of the OrthogonalProcrustes class

    """
    paired_inds1 = seeds[:,0] #first col of seeds relates to X
    paired_inds2 = seeds[:,1] #second col of seeds relates to Y
    Xp = X[paired_inds1, :] #X is sorted in terms of the first col indexes
    Yp = Y[paired_inds2, :] #Y is sorted in terms of the second col indexes
    op = OrthogonalProcrustes() #new instance of orthogonal pro.
    op.fit(Xp, Yp)  #call fit on instance of orthogonal pro. to get Q
    return op.Q_ #return Q

def fit_transform(self, X: np.ndarray, Y: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Uses the two datasets to learn the matrix :attr:`~graspologic.align.OrthogonalProcrustes.Q_` 
    that aligns the first dataset with the second. Then, transforms the first dataset ``X``
    using the learned matrix :attr:`~graspologic.align.OrthogonalProcrustes.Q_`.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Dataset to be mapped to ``Y``, must have the same shape as ``Y``.

    Y : np.ndarray, shape (m, d)
        Target dataset, must have the same shape as ``X``.

    Returns
    -------
    X_prime : np.ndarray, shape (n, d)
        First dataset of vectors, aligned to second. Equal to
        ``X`` @ :attr:`~graspologic.align.BaseAlign.Q_`.
    """
    self.fit(X,Y, seeds)
    return self.transform(X)


    