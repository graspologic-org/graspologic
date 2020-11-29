import numpy as np
from ..match import GraphMatch as GMP
from sklearn.base import BaseEstimator
import itertools


class VNviaSGM(BaseEstimator):
    """
    This class implements vertex nomination via seeded graph matching

    Parameters
    ----------
    G_1: 2d-array, square
        Adjacency matrix of `G_1`, the graph where voi is known

    G_2: 2d-array, square
        Adjacency matrix of `G_2`, the graph where voi is not known

    h: int
        distance between voi used to create induced subgraph on `G_1`

    ell: int
        distance from seeds to other verticies to create induced subgraphs on `G_1`

    R: int
        Number of restarts for soft seeded graph matching algorithm

    Attributes
    ----------
    n_seeds_used: int
        Number of seeds passed in `seedsA` that occured in the induced subgraph about `voi`

    nomination_list: 2d-array
        An array containing vertex nominations in the form nomination list = [[j, p_val],...]
        where p_val is the probability that the voi matches to node j in graph B (sorted by
        descending probability)


    References
    ----------
    .. [1] Patsolic, HG, Park, Y, Lyzinski, V, Priebe, CE. Vertex nomination via seeded graph matching. Stat Anal Data
        Min: The ASA Data Sci Journal. 2020; 13: 229– 244. https://doi.org/10.1002/sam.11454



    """

    def __init__(self, G_1, G_2, h=1, ell=1, R=100):
        if type(h) is int and h > 0:
            self.h = h
        else:
            msg = "h must be an integer > 0"
            raise ValueError(msg)
        if type(ell) is int and ell > 0:
            self.ell = ell
        else:
            msg = "ell must be an integer > 0"
            raise ValueError(msg)
        if type(R) is int and R > 0:
            self.R = R
        else:
            msg = "R must be an integer > 0"
            raise ValueError(msg)

        if G_1.shape[0] != G_1.shape[1] or G_2.shape[0] != G_2.shape[1]:
            msg = "Adjacency matrix entries must be square"
            raise ValueError(msg)
        else:
            self.G_1 = G_1
            self.G_2 = G_2

    def fit(self, X, y=[]):
        """
        Fits the model to two graphs.

        Parameters
        ----------
        X: int
            Vertex of interest (voi)

        y: list
            List of length two `[seedsA, seedsB]` where first element is
            the seeds associated with adjacency matrix G_1
            and the second element the adjacency matrix associated with G_2, note
            `len(seedsA)==len(seedsB)`. The elements of `seeds_A` and `seeds_B` are
            vertices which are known to be matched, that is, `seeds_A[i]` is matched
            to vertex `seeds_B[i]`.

        Returns
        -------
        self: A reference to self
        """
        voi = X

        if len(y) == 0:
            print("Must include at least one seed to produce nomination list")
            return None
        if len(y) != 2:
            msg = "List must be length two, with first element containing seeds \
                  of G_1 and the second containing seeds of G_2"
            raise ValueError(msg)

        seedsA = y[0]
        seedsB = y[1]

        if len(seedsA) != len(seedsB):
            msg = "Must have the same number of seeds for each adjacency matrix"
            raise ValueError(msg)
        if len(seedsA) == 0:
            print("Must include at least one seed to produce nomination list")
            return None

        voi = np.reshape(np.array(voi), (1,))

        # get reordering for A
        nsx1 = np.setdiff1d(np.arange(self.G_1.shape[0]), np.concatenate((seedsA, voi)))
        a_reord = np.concatenate((seedsA, voi, nsx1))

        # get reordering for B
        nsx2 = np.setdiff1d(np.arange(self.G_2.shape[0]), seedsB)
        b_reord = np.concatenate((seedsB, nsx2))

        AA = self.G_1[np.ix_(a_reord, a_reord)]
        BB = self.G_2[np.ix_(b_reord, b_reord)]

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
        sgm = GMP(n_init=self.R, shuffle_input=False, init="rand", padding="naive")
        corr = sgm.fit_predict(AA_fin, BB_fin, seeds_A=seeds_fin, seeds_B=seeds_fin)
        P_outp = sgm.probability_matrix_

        b_inds = b_reord[ind2]
        self.n_seeds_used = len(Sx1)

        nomination_list = list(zip(b_inds[self.n_seeds_used :], P_outp[0]))
        nomination_list.sort(key=lambda x: x[1], reverse=True)
        self.nomination_list = np.array(nomination_list)
        return self

    def fit_predict(self, X, y=[]):
        """
        Fits model to two adjacenty matricies and returns nomination list

        Parameters
        ----------
        X: int
            Vertex of interest (voi)

        y: list
            List of length two `[seedsA, seedsB]` where first element is
            the seeds associated with adjacency matrix G_1
            and the second element the adjacency matrix associated with G_2, note
            `len(seedsA)==len(seedsB)` The elements of `seeds_A` and `seeds_B` are
            vertices which are known to be matched, that is, `seeds_A[i]` is matched
            to vertex `seeds_B[i]`.

        Returns
        -------
        nomination_list : 2d-array
            The nomination array
        """
        retval = self.fit(X, y)

        if retval is None:
            return None

        return self.nomination_list


def _ego(graph_adj_matrix, order, node, mindist=1):
    """
    Generates a vertex list for the induced subgraph about a node with
    max and min distance parameters.

    Parameters
    ----------
    graph_adj_matrix: 2-d array
        Adjacency matrix of interest.

    order: int
        Distance to create the induce subgraph with. Max distance away from
        the node to include in subgraph.

    node: int
        The vertex to center the induced subgraph about.

    mindist: int
        The minimum distance away from the node to include in the subgraph.

    Returns
    -------
    induced_subgraph : list
        The list containing all the vertices in the induced subgraph.
    """
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
    """
    Generates a vertex list for the induced subgraph about a node with
    max and min distance parameters.

    Parameters
    ----------
    graph_adj_matrix: 2-d array
        Adjacency matrix of interest.

    order: int
        Distance to create the induce subgraph with. Max distance away from
        the node to include in subgraph.

    node: int or list
        The list of vertices to center the induced subgraph about.

    mindist: int
        The minimum distance away from the node to include in the subgraph.

    Returns
    -------
    induced_subgraph : list
        The list containing all the vertices in the induced subgraph.
    """
    if type(node) == list:
        total_res = []
        for nn in node:
            ego_res = _ego(graph_adj_matrix, order, nn, mindist=mindist)
            total_res.extend(ego_res)
        return list(set(total_res))
    else:
        return _ego(graph_adj_matrix, order, node)
