# Copyright 2020 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.utils import check_array
from scipy.linalg import orthogonal_procrustes

from .base import BaseAlign


class OrthogonalProcrustes(BaseAlign):
    """
    Compute the matrix solution of the orthogonal Procrustes problem, which is
    that given two matrices X and Y of equal shape, find an orthogonal matrix
    that most closely maps X to Y.

    Note that when used to match two datasets, this method unlike
    SeedlessProcrustes, not only requires that the datasets have the same
    number of entries, but also that there is some correspondence between the
    vertices. In statistical spectral graphs, this usually corresponds to the
    assumption that the vertex i in graph X has the same latent position as the
    vertex i in graph Y.

    Implementation-wise, this class is a wrapper of the
    `scipy.linalg.orthogonal_procrustes`, which itself uses an algorithm
    described in to find the optimal solution algorithm [1]_.

    Parameters
    ----------
        align_type : str, {'orthogonal' (default), 'scaling-orthogonal', 'diagonal-orthogonal'}
            Whether to perform only the regular orthogonal alignment, or a
            scaling followed by such alignment. This is commonly used in
            different test cases of the semiparametric latent position test.
            See :class:`~graspy.inference.LatentPositionTest` for more
            information.

            - 'orthogonal'
                Regular orthogonal Procrustes alignment.
            - 'scaling-orthogonal'
                First scales the two datasets to have the same Frobenius norm,
                then performs the regular orthogonal Procrustes alignment.
            - 'diagonal-orthogonal' # TODO
                First scales each entry of the two datasets to have the same
                norm as a respective entry of the other dataset.

        freeze_Y : boolean, optional (default=False)
            Irrelevant if the align_type is 'orthogonal', as the orthogonal
            transformation always modifies only the first dataset. In other
            cases:

            - True
                The second dataset will not be modified. The scaling, whether
                in Frobenius norm (if align_type='scaling-orthogonal'), or per
                each dimension (if align_type='diagonal-orthogonal'), will be
                applied to the first dataset in a way to match scale of the
                second.
            - True
                Both of the datasets will be scaled to have the Frobenius norm
                (if align_type='scaling-orthogonal') or the row-wise norms (if
                align_type='diagonal-orthogonal') be equal to value given by
                scale (unity by default).

        scale : int (default=1)
            Irrelevant if align_type='orthogonal', or if freeze_y=True, since
            in those cases only the first dataset is transformed. In other
            cases, this is the value to which the Frobenius norm (if
            align_type='scaling-orthogonal'), or the dimension-wise norms (if
            align_type='diagonal-orthogonal') will be equal to.

    Attributes
    ----------
        Q_X : array, size (d, d)
              final orthogonal matrix, used to modify X

        Q_Y : array, size (d, d)
              final orthogonal matrix, used to modify Y
              in OrthogonalProcrustes Q_Y is always equal to identity I

    References
    ----------
    .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.

    """

    def __init__(
        self,
        freeze_Y=False,
        align_type="orthogonal",
        scale=1.0,
    ):
        # check type
        if not isinstance(align_type, str):
            msg = "align_type must be a string, not {}".format(type(align_type))
            raise TypeError(msg)
        align_types_supported = [
            "orthogonal",
            "scaling-orthogonal",
            "diagonal-orthogonal",
        ]
        if align_type not in align_types_supported:
            msg = "supported types are {}".format(align_types_supported)
            raise ValueError(msg)
        if type(scale) is not float:
            msg = "scale must be a float, not {}".format(type(scale))
            raise TypeError(msg)
        # Value checking
        if scale <= 0:
            msg = "{} is an invalud value of the optimal transport eps, must be postitive".format(
                scale
            )
            raise ValueError(msg)

        super().__init__(freeze_Y=freeze_Y)
        self.align_type = align_type
        self.scale = scale

    def fit(self, X, Y):
        """
        Uses the two datasets to learn matrices Q_X and Q_Y.
        In regular orthogonal procrustes Q_X is a solution to the orthogonal
        procrustes problem, and Q_Y is the identity matrix.

        Parameters
        ----------
        X: np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions as ones in Y, but the number of vectors can differ.

        Y: np.ndarray, shape (m, d)
            Second dataset of vectors. These vectors need to have same number
            of dimensions as ones in X, but the number of vectors can differ.

        Returns
        -------
        self: returns an instance of self
        """

        # check for numpy-ness, 2d-ness and finite-ness
        if not isinstance(X, np.ndarray):
            msg = f"first dataset is a {type(X)}, not an np.ndarray! "
            raise TypeError(msg)
        if not isinstance(Y, np.ndarray):
            msg = f"first dataset is a {type(Y)}, not an np.ndarray! "
            raise TypeError(msg)
        X = check_array(X, accept_sparse=True, copy=True)
        Y = check_array(Y, accept_sparse=True, copy=True)

        # check for equal components and number of entries
        if X.shape[1] != Y.shape[1]:
            msg = "two datasets have different number of components!"
            raise ValueError(msg)
        _, d = X.shape
        if X.shape[0] != Y.shape[0]:
            msg = (
                "two datasets have different number of entries! "
                "OrthogonalProcrustes assumes that entries of the two "
                "datasets are matched. consider using SeedlessProcrustes "
                "instead."
            )
            raise ValueError(msg)

        _, d = X.shape
        if self.align_type == "orthogonal":
            D_X = np.eye(d)
            D_Y = np.eye(d)
        if self.align_type == "scaling-orthogonal":
            norm_X = np.linalg.norm(X, ord="fro")
            norm_Y = np.linalg.norm(Y, ord="fro")
            D_X = np.eye(d) / norm_X
            D_Y = np.eye(d) / norm_Y
        elif self.align_type == "diagonal-orthogonal":
            raise NotImplementedError("currently does not fit into this module")
            # normX = np.sum(X ** 2, axis=1)
            # normY = np.sum(Y ** 2, axis=1)
            # normX[normX <= 1e-15] = 1
            # normY[normY <= 1e-15] = 1
            # D_X = np.diag(1 / np.sqrt(normX[:, None]))
            # D_Y = np.diag(1 / np.sqrt(normY[:, None]))

        if self.freeze_Y is True:
            D_X = D_X @ np.linalg.inv(D_Y)
            D_Y = np.eye(D_Y.shape[0])

        X, Y = X @ D_X, Y @ D_Y
        R, _ = orthogonal_procrustes(X, Y)
        self.Q_X = R @ D_X
        self.Q_Y = D_Y
        return self
