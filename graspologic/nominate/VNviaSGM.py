import numpy as np
import random
from ..match import GraphMatch as GMP
from ..utils import pass_to_ranks
from scipy.stats import rankdata

class VNviaSGM(BaseEstimator):
    """
    
    """

    # def __init__(
    #     self,
    # ):
    #

    def fit(self, voi, A, B, seedsA, seedsB, h, ell, R, g):
        '''
        Parameters
        ----------

        voi: int
            Verticie of interest (voi)

        A: 2d-array, square
            Adjacency matrix of `G_1`, the graph where voi is known

        B: 2d-array, square
            Adjacency matrix of `G_2`, the graph where voi is not known

        seedsA: list
            Seeds associated to adjacenty matrix A

        seedsB: list
            Seeds associated to adjacency matrix B `len(seedsA)==len(seedsB)`

        h: int
            distance between voi used to create induced subgraph on `G_1`

        ell: int
            distance from seeds to other verticies to create induced subgraphs on `G_1`

        R: int
            Number of restarts for soft seeded graph matching algorithm

        g: float
            gamma to be used, max tol for alpha, tollerable dist from barycenter

        References
        ----------
        Patsolic, HG, Park, Y, Lyzinski, V, Priebe, CE. Vertex nomination via seeded graph matching. Stat Anal Data
        Min: The ASA Data Sci Journal. 2020; 13: 229– 244. https://doi.org/10.1002/sam.11454
        '''
        assert len(seedsA) == len(seedsB)

        voi = np.reshape(np.array(voi), (1,))

        # get reordering for A
        nsx1 = np.setdiff1d(list(range(A.shape[0])), np.concatenate((seedsA, voi)))
        a_reord = np.concatenate((seedsA, voi, nsx1))

        # get reordering for B
        nsx2 = np.setdiff1d(list(range(B.shape[0])), seedsB)
        b_reord = np.concatenate((seedsB, nsx2))

        AA = A[np.ix_(a_reord, a_reord)]
        BB = B[np.ix_(b_reord, b_reord)]

        seeds_reord = np.arange(len(seedsA))
        voi_reord = len(seedsA)
        #     print("seeds reord = ", seeds_reord, ", voi_reord = ", voi_reord)

        subgraph_AA = np.array(ego_list(AA, h, voi_reord, mindist=1))

        voi_reord = np.reshape(np.array(voi_reord), (1,))

        Sx1 = np.intersect1d(a_reord[subgraph_AA], seeds_reord)
        Sx2 = np.intersect1d(np.arange(BB.shape[0]), Sx1)
        #     print("Sx1 = ", Sx1, " Sx2 = ", Sx2)

        if len(Sx2) <= 0:
            print("Impossible")
            return None

        Nx1 = np.array(ego_list(AA, ell, list(Sx1), mindist=0))
        Nx2 = np.array(ego_list(BB, ell, list(Sx2), mindist=0))

        foo = np.concatenate((Sx1, voi_reord))
        ind1 = np.concatenate((Sx1, voi_reord, np.setdiff1d(Nx1, foo)))
        ind2 = np.concatenate((Sx2, np.setdiff1d(Nx2, Sx2)))

        AA_fin = AA[np.ix_(ind1, ind1)]
        BB_fin = BB[np.ix_(ind2, ind2)]


        seeds_fin = list(range(len(Sx1)))
        sgm = GMP(n_init=R, shuffle_input=False, init_method="rand", padding="naive")
        corr = sgm.fit_predict(AA_fin, BB_fin, seeds_A=seeds_fin, seeds_B=seeds_fin)
        P_outp = sgm.probability_matrix_

        self.P = P_outp
        self.corr = corr
        self.a_inds = a_reord[ind1]
        self.b_inds = b_reord[ind2]
        self.n_seeds_used = len(Sx1)

        nomination_list = list(zip(self.b_inds[self.n_seeds_used:], self.P[0]))
        nomination_list.sort(key=lambda x: x[1], reverse=True)
        self.nomination_list = nomination_list

    def fit_predict(self, A, B, seeds_A=[], seeds_B=[]):
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
        self.fit(A, B, seeds_A, seeds_B)
        return self.nomination_list


def ego(graph_adj_matrix, order, node, mindist=1):
    # Note all nodes are zero based in this implementation, i.e the first node is 0
    dists = [[node]]
    for ii in range(1, order + 1):
        clst = []
        for nn in dists[-1]:
            clst.extend(list(np.where(graph_adj_matrix[nn] == 1)[0]))
        clst = list(set(clst))

        # Remove all the ones that are closer (i.e thtat have already been included)
        dists_conglom = []
        for dd in dists:
            dists_conglom.extend(dd)
        dists_conglom = list(set(dists_conglom))

        cn_proc = []
        for cn in clst:
            if cn not in dists_conglom:
                cn_proc.append(cn)

        dists.append(cn_proc)
    ress = []

    for ii in range(mindist, order + 1):
        ress.extend(dists[ii])

    return np.array(list(set(ress)))


def ego_list(graph_adj_matrix, order, node, mindist=1):
    #     print(type(node))
    if type(node) == list:
        total_res = []
        for nn in node:
            ego_res = ego(graph_adj_matrix, order, nn, mindist=mindist)
            total_res.extend(ego_res)
        return list(set(total_res))
    else:
        return ego(graph_adj_matrix, order, node)