import numpy as np
from ..match import GraphMatch as GMP
from sklearn.base import BaseEstimator
import itertools


class VNviaSGM(BaseEstimator):
    """
    This class implements Vertex Nomination via Seeded Graph Matching
    (VNviaSGM) with the algorithm described in [1].

    VNviaSGM is a nomination algorithm, so instead of completely matching two
    graphs `A` and `B`, it proposes a nomination list of potential matches in
    graph `B` to a vertex of interest (voi) in graph `A`. VNviaSGM matches
    subgraphs about the given seeds and orders many times. All these results
    are then averaged to produce a nomination probability list for the voi.

    Parameters
    ----------
    order_voi_subgraph: int, positive (default = 1)
        distance between voi used to create induced subgraph on `A`

    order_seeds_subgraph: int, positive (default = 1)
        distance from seeds to other verticies to create induced subgraphs on `A`
        and `B`

    init: int, positive (default = 100)
        Number of restarts for soft seeded graph matching algorithm

    Attributes
    ----------
    n_seeds: int
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

    def __init__(self, order_voi_subgraph=1, order_seeds_subgraph=1, n_init=100):
        if type(order_voi_subgraph) is int and order_voi_subgraph > 0:
            self.order_voi_subgraph = order_voi_subgraph
        else:
            msg = "order_voi_subgraph must be an integer > 0"
            raise ValueError(msg)
        if type(order_seeds_subgraph) is int and order_seeds_subgraph > 0:
            self.order_seeds_subgraph = order_seeds_subgraph
        else:
            msg = "order_seeds_subgraph must be an integer > 0"
            raise ValueError(msg)
        if type(n_init) is int and n_init > 0:
            self.n_init = n_init
        else:
            msg = "R must be an integer > 0"
            raise ValueError(msg)

    def fit(self, voi, seeds=[], A=None, B=None):
        """
        Fits the model to two graphs.

        Parameters
        ----------
        voi: int
            Vertex of interest (voi)

        seeds: list
            List of length two `[seedsA, seedsB]` where first element is
            the seeds associated with adjacency matrix A
            and the second element the adjacency matrix associated with B, note
            `len(seedsA)==len(seedsB)`. The elements of `seedsA` and `seedsB` are
            vertices which are known to be matched, that is, `seedsA[i]` is matched
            to vertex `seedsB[i]`.

        A: 2d-array, square
            Adjacency matrix of `A`, the graph where voi is known

        B: 2d-array, square
            Adjacency matrix of `B`, the graph where voi is not known

        Returns
        -------
        self: A reference to self
        """
        if A is None or B is None:
            msg = "Adjacency matricies must be passed"
            raise ValueError(msg)
        elif len(A.shape) != 2 or len(B.shape) != 2:
            msg = "Adjacency matrix entries must be square"
            raise ValueError(msg)
        elif A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
            msg = "Adjacency matrix entries must be square"
            raise ValueError(msg)

        if len(seeds) == 0:
            print("Must include at least one seed to produce nomination list")
            return None
        if len(seeds) != 2:
            msg = "List must be length two, with first element containing seeds \
                  of A and the second containing seeds of B"
            raise ValueError(msg)

        seedsA = seeds[0]
        seedsB = seeds[1]

        if len(seedsA) != len(seedsB):
            msg = "Must have the same number of seeds for each adjacency matrix"
            raise ValueError(msg)
        if len(seedsA) == 0:
            print("Must include at least one seed to produce nomination list")
            return None

        voi = np.reshape(np.array(voi), (1,))

        # get reordering for Ax
        nsx1 = np.setdiff1d(np.arange(A.shape[0]), np.concatenate((seedsA, voi)))
        a_reord = np.concatenate((seedsA, voi, nsx1))

        # get reordering for B
        nsx2 = np.setdiff1d(np.arange(B.shape[0]), seedsB)
        b_reord = np.concatenate((seedsB, nsx2))

        AA = A[np.ix_(a_reord, a_reord)]
        BB = B[np.ix_(b_reord, b_reord)]

        seeds_reord = np.arange(len(seedsA))
        voi_reord = len(seedsA)

        subgraph_AA = np.array(
            _get_induced_subgraph_list(
                AA, self.order_voi_subgraph, voi_reord, mindist=1
            )
        )

        voi_reord = np.reshape(np.array(voi_reord), (1,))

        Sx1 = np.intersect1d(a_reord[subgraph_AA], seeds_reord)
        Sx2 = np.intersect1d(np.arange(BB.shape[0]), Sx1)

        if len(Sx2) <= 0:
            print(
                "Voi {} was not a member of the induced subgraph A[{}]".format(
                    voi, seedsA
                )
            )
            return None

        Nx1 = np.array(
            _get_induced_subgraph_list(
                AA, self.order_seeds_subgraph, list(Sx1), mindist=0
            )
        )
        Nx2 = np.array(
            _get_induced_subgraph_list(
                BB, self.order_seeds_subgraph, list(Sx2), mindist=0
            )
        )

        foo = np.concatenate((Sx1, voi_reord))
        ind1 = np.concatenate((Sx1, voi_reord, np.setdiff1d(Nx1, foo)))
        ind2 = np.concatenate((Sx2, np.setdiff1d(Nx2, Sx2)))

        AA_fin = AA[np.ix_(ind1, ind1)]
        BB_fin = BB[np.ix_(ind2, ind2)]

        seeds_fin = list(range(len(Sx1)))
        sgm = GMP(n_init=self.n_init, shuffle_input=False, init="rand", padding="naive")
        corr = sgm.fit_predict(AA_fin, BB_fin, seeds_A=seeds_fin, seeds_B=seeds_fin)
        P_outp = sgm.probability_matrix_

        b_inds = b_reord[ind2]
        self.n_seeds = len(Sx1)

        nomination_list = list(zip(b_inds[self.n_seeds :], P_outp[0]))
        nomination_list.sort(key=lambda x: x[1], reverse=True)
        self.nomination_list = np.array(nomination_list)
        return self

    def fit_predict(self, voi, seeds=[], A=None, B=None):
        """
        Fits model to two adjacenty matricies and returns nomination list

        Parameters
        ----------
        voi: int
            Vertex of interest (voi)

        seeds: list
            List of length two `[seedsA, seedsB]` where first element is
            the seeds associated with adjacency matrix A
            and the second element the adjacency matrix associated with B, note
            `len(seedsA)==len(seedsB)` The elements of `seedsA` and `seedsB` are
            vertices which are known to be matched, that is, `seedsA[i]` is matched
            to vertex `seedsB[i]`.

        A: 2d-array, square
            Adjacency matrix of `A`, the graph where voi is known

        B: 2d-array, square
            Adjacency matrix of `B`, the graph where voi is not known

        Returns
        -------
        nomination_list : 2d-array
            The nomination array
        """
        retval = self.fit(voi, seeds, A=A, B=B)

        if retval is None:
            return None

        return self.nomination_list


def _get_induced_subgraph(graph_adj_matrix, order, node, mindist=1):
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

    mindist: int (default = 1)
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


def _get_induced_subgraph_list(graph_adj_matrix, order, node, mindist=1):
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

    mindist: int (default = 1)
        The minimum distance away from the node to include in the subgraph.

    Returns
    -------
    induced_subgraph : list
        The list containing all the vertices in the induced subgraph.
    """
    if type(node) == list:
        total_res = []
        for nn in node:
            ego_res = _get_induced_subgraph(
                graph_adj_matrix, order, nn, mindist=mindist
            )
            total_res.extend(ego_res)
        return list(set(total_res))
    else:
        return _get_induced_subgraph(graph_adj_matrix, order, node)
