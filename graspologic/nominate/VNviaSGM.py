import numpy as np
from ..match import GraphMatch as GMP
from sklearn.base import BaseEstimator


class VNviaSGM(BaseEstimator):
    """
    This class implements vertex nomination via seeded graph matching

    Parameters
    ----------
    h: int
        distance between voi used to create induced subgraph on `G_1`

    ell: int
        distance from seeds to other verticies to create induced subgraphs on `G_1`

    R: int
        Number of restarts for soft seeded graph matching algorithm

    a_inds: 1d-array
        Array of indices in the induced subgraph of A that will be matched to `b_inds`

    b_inds: 1d-array
        Array of indices in the induced subgraph of B that will be matched to `a_inds`

    n_seeds_used: int
        Number of seeds passed in `seedsA` that occured in the induced subgraph about `voi`

    nomination_list: list
        List of 2 tuples of the format nomination_list = [(j, p_val)] where p_val is the
        probability that the voi matches to node j in graph B (sorted by descending
        probability)

    Attributes
    ----------
    P : 2d-array, square
        The probability array that describes the probability `P[i, j]` that row `i` in `a_inds`
        matches `j` in `b_inds`

    corr : 1d-array
        The matrix that maps values from `a_inds` to `b_inds`


    References
    ----------
    .. [1] Patsolic, HG, Park, Y, Lyzinski, V, Priebe, CE. Vertex nomination via seeded graph matching. Stat Anal Data
    Min: The ASA Data Sci Journal. 2020; 13: 229– 244. https://doi.org/10.1002/sam.11454
    """

    def __init__(self, h=1, ell=1, R=100):
        self.h = h
        self.ell = ell
        self.R = R

    def fit(self, voi, A, B, seedsA=[], seedsB=[]):
        """
        Parameters
        ----------

        voi: int
            Vertices of interest (voi)

        A: 2d-array, square
            Adjacency matrix of `G_1`, the graph where voi is known

        B: 2d-array, square
            Adjacency matrix of `G_2`, the graph where voi is not known

        seedsA: list
            Seeds associated to adjacency matrix A

        seedsB: list
            Seeds associated to adjacency matrix B `len(seedsA)==len(seedsB)`
        """
        assert len(seedsA) == len(seedsB)
        if len(seedsA) == 0:
            print("Must include at least one seed to produce nomination list")
            return None

        voi = np.reshape(np.array(voi), (1,))

        # get reordering for A
        nsx1 = np.setdiff1d(np.arange(A.shape[0]), np.concatenate((seedsA, voi)))
        a_reord = np.concatenate((seedsA, voi, nsx1))

        # get reordering for B
        nsx2 = np.setdiff1d(np.arange(B.shape[0]), seedsB)
        b_reord = np.concatenate((seedsB, nsx2))

        AA = A[np.ix_(a_reord, a_reord)]
        BB = B[np.ix_(b_reord, b_reord)]

        seeds_reord = np.arange(len(seedsA))
        voi_reord = len(seedsA)

        subgraph_AA = np.array(_ego_list(AA, self.h, voi_reord, mindist=1))

        voi_reord = np.reshape(np.array(voi_reord), (1,))

        Sx1 = np.intersect1d(a_reord[subgraph_AA], seeds_reord)
        Sx2 = np.intersect1d(np.arange(BB.shape[0]), Sx1)

        if len(Sx2) <= 0:
            print("Voi was not a member of the induced subgraph A[seedsA]")
            return None

        Nx1 = np.array(_ego_list(AA, self.ell, list(Sx1), mindist=0))
        Nx2 = np.array(_ego_list(BB, self.ell, list(Sx2), mindist=0))

        foo = np.concatenate((Sx1, voi_reord))
        ind1 = np.concatenate((Sx1, voi_reord, np.setdiff1d(Nx1, foo)))
        ind2 = np.concatenate((Sx2, np.setdiff1d(Nx2, Sx2)))

        AA_fin = AA[np.ix_(ind1, ind1)]
        BB_fin = BB[np.ix_(ind2, ind2)]

        seeds_fin = list(range(len(Sx1)))
        sgm = GMP(
            n_init=self.R, shuffle_input=False, init_method="rand", padding="naive"
        )
        corr = sgm.fit_predict(AA_fin, BB_fin, seeds_A=seeds_fin, seeds_B=seeds_fin)
        P_outp = sgm.probability_matrix_

        self.P = P_outp
        self.corr = corr
        self.a_inds = a_reord[ind1]
        self.b_inds = b_reord[ind2]
        self.n_seeds_used = len(Sx1)

        nomination_list = list(zip(self.b_inds[self.n_seeds_used :], self.P[0]))
        nomination_list.sort(key=lambda x: x[1], reverse=True)
        self.nomination_list = nomination_list
        return self

    def fit_predict(self, voi, A, B, seedsA=[], seedsB=[]):
        """
        Fits the model with two assigned adjacency matrices, returning optimal
        permutation indices

        Parameters
        ----------
        A : 2d-array, square
            A square adjacency matrix

        B : 2d-array, square
            A square adjacency matrix

        seeds_A : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in `A`.

        seeds_B : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in `B` The elements of
            `seeds_A` and `seeds_B` are vertices which are known to be matched, that is,
            `seeds_A[i]` is matched to vertex `seeds_B[i]`.

        Returns
        -------
        perm_inds_ : 1-d array, some shuffling of [0, n_vert)
            The optimal permutation indices to minimize the objective function
        """
        retval = self.fit(voi, A, B, seedsA=seedsA, seedsB=seedsB)

        if retval is None:
            return None

        return self.nomination_list


def _ego(graph_adj_matrix, order, node, mindist=1):
    # Note all nodes are zero based in this implementation, i.e the first node is 0
    dists = [[node]]
    dists_conglom = [node]
    for ii in range(1, order + 1):
        clst = []
        for nn in dists[-1]:
            clst.extend(list(np.where(graph_adj_matrix[nn] >= 1)[0]))
        clst = np.array(list(set(clst)))

        cn_proc = np.setdiff1d(clst, dists_conglom)

        dists.append(cn_proc)

        dists_conglom.extend(cn_proc)
        dists_conglom = list(set(dists_conglom))

    ress = itertools.chain(*dists[mindist : order + 1])

    return np.array(list(set(ress)))


def _ego_list(graph_adj_matrix, order, node, mindist=1):
    if type(node) == list:
        total_res = []
        for nn in node:
            ego_res = _ego(graph_adj_matrix, order, nn, mindist=mindist)
            total_res.extend(ego_res)
        return list(set(total_res))
    else:
        return _ego(graph_adj_matrix, order, node)
