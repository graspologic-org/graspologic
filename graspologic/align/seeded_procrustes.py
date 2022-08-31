import numpy as np
from beartype import beartype
from sklearn.utils import check_scalar

from .base import BaseAlign
from .orthogonal_procrustes import OrthogonalProcrustes
from .seedless_procrustes import SeedlessProcrustes


class SeededProcrustes(BaseAlign):
    """
    Aligns two datasets when a partial matching of entries
    between the two datasets is known.

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
    This method uses orthogonal procrustes on the data for which a 1-to-1 matching
    of elements is known, and then passes this as an initial guess to seedless procrustes.

    See also
    ---------
    For more on the implementation of Orthogonal Procrustes see :class:`~graspologic.align.OrthogonalProcrustes`
    For more on the implementation of Seedless Procrustes see :class:`~graspologic.align.SeedlessProcrustes`
    """

    @beartype
    def __init__(
        self,
        optimal_transport_lambda: float = 0.1,
        optimal_transport_eps: float = 0.01,
        optimal_transport_num_reps: int = 1000,
        iterative_num_reps: int = 100,
    ):
        check_scalar(
            optimal_transport_lambda,
            name="optimal_transport_lambda",
            target_type=(float, int),
            min_val=0,
        )
        self.optimal_transport_lambda = optimal_transport_lambda
        check_scalar(
            optimal_transport_eps,
            name="optimal_transport_eps",
            target_type=(float, int),
            min_val=0,
        )
        self.optimal_transport_eps = optimal_transport_eps
        check_scalar(
            optimal_transport_num_reps,
            name="optimal_transport_num_reps",
            target_type=(int),
            min_val=1,
        )
        self.optimal_transport_num_reps = optimal_transport_num_reps
        check_scalar(
            iterative_num_reps,
            name="iterative_num_reps",
            target_type=(int),
            min_val=0,
        )
        self.iterative_num_reps = iterative_num_reps
        super().__init__()

    @beartype
    def fit(
        self, X: np.ndarray, Y: np.ndarray, seeds: np.ndarray
    ) -> "SeededProcrustes":
        """
        Uses the two datasets to learn the matrix `self.Q_` that aligns the
        first dataset with the second using a `seeds` matrix.

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

        Returns
        -------
        self : returns an instance of self
        """
        init_Q = _find_Q_from_pairs(X, Y, seeds)
        procruster = SeedlessProcrustes(
            init="custom",
            initial_Q=init_Q,
            optimal_transport_lambda=self.optimal_transport_lambda,
            optimal_transport_eps=self.optimal_transport_eps,
            optimal_transport_num_reps=self.optimal_transport_num_reps,
            iterative_num_reps=self.iterative_num_reps,
        )
        procruster.fit(X, Y)
        self.Q_ = procruster.Q_
        return self

    @beartype
    def fit_transform(
        self, X: np.ndarray, Y: np.ndarray, seeds: np.ndarray
    ) -> np.ndarray:
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
        self.fit(X, Y, seeds)
        return self.transform(X)


def _find_Q_from_pairs(X: np.ndarray, Y: np.ndarray, seeds: np.ndarray) -> np.ndarray:
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
    Q : An estimate of the orthogonal alignment learned from the seeded elements
    """
    paired_inds1 = seeds[:, 0]
    paired_inds2 = seeds[:, 1]
    Xp = X[paired_inds1, :]
    Yp = Y[paired_inds2, :]
    op = OrthogonalProcrustes()
    op.fit(Xp, Yp)
    return op.Q_
