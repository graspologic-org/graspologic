import itertools
import warnings
from typing import Any, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator

from graspologic.types import Dict, List

from ..match import graph_match

# Type aliases
SeedsType = Union[np.ndarray, List[List[int]]]


class VNviaSGM(BaseEstimator):
    """
    This class implements Vertex Nomination via Seeded Graph Matching (VNviaSGM) with
    the algorithm described in [1].

    Rather than providing a 1-1 matching for the vertices of two graphs, as in
    :class:`~graspologic.match.GraphMatch`, VNviaSGM ranks the potential matches for a
    vertex of interst (VOI) in one to graph to the vertices in another graph, based on
    probability of matching.


    Parameters
    ----------
    order_voi_subgraph: int, positive (default = 1)
        Order used to create induced subgraph on ``A`` about VOI where the max distance
        between VOI and other nodes is ``order_voi_subgraph``. This induced subgraph
        will be used to determine what seeds are used when the SGM algorithm is called.
        If no seeds are in this subgraph about VOI, then a UserWarning is thrown, and
        ``nomination_list_`` is None.

    order_seeds_subgraph: int, positive (default = 1)
        Order used to  create induced subgraphs on ``A`` and ``B``. These subgraphs
        are centered about the seeds that were determined by the subgraph generated
        by ``order_voi_subgraph``. These two subgraphs will be passed into the SGM
        algorithm.

    n_init: int, positive (default = 100)
        Number of random initializations of the seeded graph matching algorithm (SGM).
        Increasing the number of restarts will make the probabilities returned more
        precise.

    max_nominations: int (default = None)
        Max number of nominations to include in the nomination list. If None is passed,
        then all nominations computed will be returned.

    graph_match_kws : dict (default = {})
        Gives users the option to pass custom arguments to the graph matching
        algorithm. Format should be {'arg_name': arg_value, ...}. See
        :class:`~graspologic.match.GraphMatch`


    Attributes
    ----------
    n_seeds_: int
        Number of seeds passed in `seedsA` that occured in the induced subgraph about
        VOI

    nomination_list_: 2d-array
        An array containing vertex nominations in the form nomination
        list = [[j, p_val],...] where p_val is the probability that the VOI matches
        to node j in graph B (sorted by descending probability)

    Notes
    -----
    VNviaSGM generates an initial induced subgraph about the VOI to determine which
    seeds are close enough to be used. If no seeds are close enough, then a warning
    is thrown and ``nomination_list_`` is set to None.

    All the seeds that are close enough are then used to generate subgraphs in both
    ``A`` and ``B``. These subgraphs are matched using several random initializations
    of the seeded graph matching algorithm (SGM), and a nomination list is returned.
    See :class:`~graspologic.match.GraphMatch` for SGM docs

    References
    ----------
    .. [1] Patsolic, HG, Park, Y, Lyzinski, V, Priebe, CE. Vertex nomination via seeded
        graph matching. Stat Anal Data Min: The ASA Data Sci Journal. 2020; 13: 229–
        244. https://doi.org/10.1002/sam.11454

    """

    nomination_list_: Optional[np.ndarray]

    def __init__(
        self,
        order_voi_subgraph: int = 1,
        order_seeds_subgraph: int = 1,
        n_init: int = 100,
        max_nominations: Optional[int] = None,
        graph_match_kws: Dict[str, Any] = {},
    ):
        if isinstance(order_voi_subgraph, int) and order_voi_subgraph > 0:
            self.order_voi_subgraph = order_voi_subgraph
        else:
            msg = '"order_voi_subgraph" must be an integer > 0'
            raise ValueError(msg)
        if isinstance(order_seeds_subgraph, int) and order_seeds_subgraph > 0:
            self.order_seeds_subgraph = order_seeds_subgraph
        else:
            msg = '"order_seeds_subgraph" must be an integer > 0'
            raise ValueError(msg)
        if isinstance(n_init, int) and n_init > 0:
            self.n_init = n_init
        else:
            msg = '"n_init" must be an integer > 0'
            raise ValueError(msg)

        if max_nominations is None:
            self.max_nominations = max_nominations
        elif isinstance(max_nominations, int) and max_nominations >= 1:
            self.max_nominations = max_nominations
        else:
            msg = '"max_nominations" must be an integer >= 1'
            raise ValueError(msg)

        # Error checking of these will be handled by GMP
        if isinstance(graph_match_kws, dict):
            self.graph_match_kws = graph_match_kws
        else:
            msg = '"graph_match_kws` must be type dict'
            raise ValueError(msg)

    def fit(
        self, A: np.ndarray, B: np.ndarray, voi: int, seeds: SeedsType
    ) -> "VNviaSGM":
        """
        Fits the model to two graphs.

        Parameters
        ----------
        A: 2d-array, square
            Adjacency matrix, the graph where ``voi`` is known

        B: 2d-array, square
            Adjacency matrix, the graph where ``voi`` is not known

        voi: int
            Vertex of interest

        seeds: list, 2d-array
            List of length two, of form `[seedsA, seedsB]`. The elements of `seedsA`
            and `seedsB` are vertex indices from ``A`` and ``B``, respectively, which
            are known to be matched; that is, vertex `seedsA[i]` is matched to vertex
            `seedsB[i]`. Note: `len(seedsA)==len(seedsB)`.

        Returns
        -------
        self: An instance of self
        """
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)

        if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
            msg = '"A" and "B" must be type np.ndarray'
            raise ValueError(msg)
        elif A.ndim != 2 or B.ndim != 2:
            msg = '"A" and "B" must be two-dimensional'
            raise ValueError(msg)
        elif A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
            msg = '"A" and "B" must be square'
            raise ValueError(msg)
        elif A.shape[0] > B.shape[0]:
            # NOTE: the new graph_match function can absolutely handle the reverse case.
            # However, it would require me to appropriately deal with the nodes of A
            # which are not matched, and I dont have time to figure out what this class
            # is doing right now. Further, I think with the old code using GraphMatch
            # this would have raised a silent bug in this case, so I think this is
            # at least an improvement.
            msg = '"A" is larger than "B"; please reverse the ordering of these inputs.'
            raise ValueError(msg)

        if not isinstance(voi, int):
            msg = '"voi" must be an integer'
            raise ValueError(msg)
        elif voi < 0 or voi >= A.shape[0]:
            msg = '"voi" must be in range[0, num_verts_A)'
            raise ValueError(msg)

        if not (isinstance(seeds, list) or isinstance(seeds, np.ndarray)):
            msg = '"seeds" must be a list'
            raise ValueError(msg)

        if isinstance(seeds, list):
            if len(seeds) != 2:
                msg = 'seeds must be length two, with first element containing seeds \
                      of "A" and the second containing seeds of "B"'
                raise ValueError(msg)

            if not (
                isinstance(seeds[0], list) or isinstance(seeds[0], np.ndarray)
            ) or not (isinstance(seeds[1], list) or isinstance(seeds[1], np.ndarray)):
                msg = '"seeds" elements must be lists or arrays'
                raise ValueError(msg)

            seedsA = np.array(seeds[0])
            seedsB = np.array(seeds[1])

        else:
            seeds = np.atleast_2d(seeds)

            if not isinstance(seeds, np.ndarray):
                msg = '"seeds" be a list or 2d-array'
                raise ValueError(msg)
            if seeds.shape[1] != 2:
                msg = '"seeds" must have a second dimension of two'
                raise ValueError(msg)

            seedsA = seeds[:, 0]
            seedsB = seeds[:, 1]

        if len(seedsA) != len(seedsB):
            msg = "Must have the same number of seeds for each adjacency matrix"
            raise ValueError(msg)
        elif len(seedsA) == 0:
            msg = 'len("seeds") must be at least one'
            raise ValueError(msg)
        elif not len(set(seedsA)) == len(seedsA) or not len(set(seedsB)) == len(seedsB):
            msg = '"seeds" column entries must be unique'
            raise ValueError(msg)
        elif len(seedsA) > A.shape[0]:
            msg = '"seeds" cant have more entries than its associated adjacency matrix'
            raise ValueError(msg)
        elif (seedsA >= A.shape[0]).any() or (seedsB >= B.shape[0]).any():
            msg = '"seeds" entries must be less than number of nodes in their graphs'
            raise ValueError(msg)

        # get vertex reordering for Ax
        # in the form (seedsA, voi, rest in order)
        nsx1 = np.setdiff1d(np.arange(A.shape[0]), np.append(seedsA, voi))
        a_reord = np.append(np.append(seedsA, voi), nsx1)

        # get reordering for B in the form (seedsB, rest in numerical order)
        nsx2 = np.setdiff1d(np.arange(B.shape[0]), seedsB)
        b_reord = np.concatenate((seedsB, nsx2))

        # Reorder the two graphs with our new vertices order
        A_perm = A[a_reord][:, a_reord]
        B_perm = B[b_reord][:, b_reord]

        # Record where the new seeds and voi locations are
        # in our re-ordered graphs
        seeds_reord = np.arange(len(seedsA))
        voi_reord = len(seedsA)

        # Determine what seeds are within a specified subgraph
        # given by `self.order_voi_subgraph`. If there are no
        # seeds in this subgraph, print a message and return None
        subgraph_A_perm = _get_induced_subgraph_list(
            A_perm, self.order_voi_subgraph, voi_reord, mindist=1
        )

        close_seeds = np.intersect1d(subgraph_A_perm, seeds_reord)

        if len(close_seeds) <= 0:
            warnings.warn(
                'Voi {} was not a member of the induced subgraph A[{}], \
                Try increasing "order_voi_subgraph"'.format(
                    voi, seedsA
                )
            )
            self.n_seeds_ = None
            self.nomination_list_ = None
            return self

        # Generate the two induced subgraphs that will be used by the matching
        # algorithm using the seeds that we identified in the previous step.
        verts_A = _get_induced_subgraph_list(
            A_perm, self.order_seeds_subgraph, close_seeds, mindist=0
        )

        verts_B = _get_induced_subgraph_list(
            B_perm, self.order_seeds_subgraph, close_seeds, mindist=0
        )

        # Determine the final reordering for the graphs that include only
        # the vertices found by the induced subgraphs in the previous step
        # For graph A, its of the form (close_seeds, voi, rest in verts_A
        # in num order). For graph B its of the form (close_seeds, rest in
        # verts_B in num order)
        permed_verts = np.append(close_seeds, voi_reord)
        ind1 = np.append(
            np.append(close_seeds, voi_reord), np.setdiff1d(verts_A, permed_verts)
        )
        ind2 = np.concatenate((close_seeds, np.setdiff1d(verts_B, close_seeds)))

        # Generate adjacency matrices for the ordering found in the prev step
        SG_1 = A_perm[ind1][:, ind1]
        SG_2 = B_perm[ind2][:, ind2]

        # Record the number of seeds used because this may differ from the number
        # of seeds passed. See the step where close_seeds was computed for an
        # explanation
        self.n_seeds_ = len(close_seeds)
        seeds_fin = np.arange(self.n_seeds_)
        partial_match = np.column_stack((seeds_fin, seeds_fin))

        # Call the SGM algorithm using user set parameters and generate a prob
        # vector for the voi.
        prob_vector = np.zeros((max(SG_1.shape[0], SG_2.shape[0]) - self.n_seeds_))

        for ii in range(self.n_init):
            _, perm_inds, _, _ = graph_match(
                SG_1, SG_2, partial_match=partial_match, **self.graph_match_kws
            )
            prob_vector[perm_inds[self.n_seeds_] - self.n_seeds_] += 1.0

        prob_vector /= self.n_init

        # Get the original vertices names in the B graph to make the nom list
        b_inds = b_reord[ind2]

        # Generate the nomination list. Note, the probability matrix does not
        # include the seeds, so we must remove them from b_inds. Return a list
        # sorted so it returns the vertex with the highest probability first.
        nomination_list_ = np.dstack((b_inds[self.n_seeds_ :], prob_vector))[0]
        nomination_list_ = nomination_list_[nomination_list_[:, 1].argsort()][::-1]

        if self.max_nominations is not None and self.max_nominations < len(
            nomination_list_
        ):
            nomination_list_ = nomination_list_[0 : self.max_nominations]

        self.nomination_list_ = nomination_list_

        return self

    def fit_predict(
        self, A: np.ndarray, B: np.ndarray, voi: int, seeds: SeedsType
    ) -> Optional[np.ndarray]:
        """
        Fits model to two adjacency matrices and returns nomination list

        Parameters
        ----------
        A: 2d-array, square
            Adjacency matrix, the graph where ``voi`` is known

        B: 2d-array, square
            Adjacency matrix, the graph where ``voi`` is not known

        voi: int
            Vertex of interest

        seeds: list, 2d-array
            List of length two, of form `[seedsA, seedsB]`. The elements of `seedsA`
            and `seedsB` are vertex indices from ``A`` and ``B``, respectively, which
            are known to be matched; that is, vertex `seedsA[i]` is matched to
            vertex `seedsB[i]`. Note: `len(seedsA)==len(seedsB)`.

        Returns
        -------
        nomination_list_ : 2d-array
            The nomination list.
        """
        self.fit(A, B, voi, seeds)

        return self.nomination_list_


def _get_induced_subgraph(
    graph_adj_matrix: np.ndarray, order: int, node: int, mindist: int = 1
) -> np.ndarray:
    """
    Generates a vertex list for the induced subgraph about a node with
    max and min distance parameters.

    Parameters
    ----------
    graph_adj_matrix: 2-d array
        Adjacency matrix of interest.

    order: int
        Distance to create the induced subgraph with. Max distance away from the node
        to include in subgraph.

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
    dists: List[Union[List[int], np.ndarray]] = [[node]]
    dists_conglom = [node]
    for ii in range(1, order + 1):
        clst = []
        for nn in dists[-1]:
            clst.extend(list(np.where(graph_adj_matrix[nn] >= 1)[0]))
        clst = np.unique(clst)

        cn_proc = np.setdiff1d(clst, dists_conglom)

        dists.append(cn_proc)

        dists_conglom = np.unique(np.append(dists_conglom, cn_proc))

    ress = list(itertools.chain(*dists[mindist : order + 1]))

    return np.unique(ress)


def _get_induced_subgraph_list(
    graph_adj_matrix: np.ndarray, order: int, node: int, mindist: int = 1
) -> np.ndarray:
    """
    Generates a vertex list for the induced subgraph about a node with
    max and min distance parameters.

    Parameters
    ----------
    graph_adj_matrix: 2-d array
        Adjacency matrix of interest.

    order: int
        Distance to create the induce subgraph with. Max distance away from the node
        to include in subgraph.

    node: int or list
        The list of vertices to center the induced subgraph about.

    mindist: int (default = 1)
        The minimum distance away from the node to include in the subgraph.

    Returns
    -------
    induced_subgraph : list
        The list containing all the vertices in the induced subgraph.
    """
    if isinstance(node, list) or isinstance(node, np.ndarray):
        total_res = []
        for nn in node:
            ego_res = _get_induced_subgraph(
                graph_adj_matrix, order, nn, mindist=mindist
            )
            total_res.extend(ego_res)
        return np.unique(total_res)
    else:
        return _get_induced_subgraph(graph_adj_matrix, order, node)
