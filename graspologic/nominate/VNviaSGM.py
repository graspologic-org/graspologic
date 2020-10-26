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



    def vnsgm_ordered(self,x,S,A,B,h,ell,R,g,pad=0,sim=True,verb=False,plotF=False):
        '''
        Parameters
        ----------

        x: list
            Verticies of intrerest

        S: list
            Vector of seeds

        A: ndarray
            `G_1` where voi are known

        B: ndarray
            `G_2` where voi unknown

        rest same as before
        '''
        s = len(S)
        Nh = ego_list(A, h, x, mindist=1) # TODO: make able to take a list in <- unlist(ego(g1,h,nodes=x,mindist=1))

        Sx1 = None; Sx2 = None

        Sx1 = list(set(Nh).intersection(set(S))); Sx1.sort(); sx = len(Sx1)

        Sx2 = list(set(list(range(B.shape[0]))).intersection(set(Sx1))); Sx2.sort(); sx2 = len(Sx2)
        #print("SX1, Sx2, ", Sx1, Sx2)

        case = 'possible' if sx2>0 else 'impossible'

        if case == 'possible':
            Cx2 = ego_list(B, ell, Sx2, mindist=1)
            Cx2 = set(Cx2).difference(set(Sx2))
            print(Cx2)
            if sim:
                case = np.array(['possible' if _x in Cx2 else 'impossible' for _x in x])


            if len(np.where(case == 'possible')[0])>0:
                Nx1 = ego_list(A, ell, Sx1, mindist=0)
                Nx2 = ego_list(B, ell, Sx2, mindist=0)

                #print("NX1, Nx2, ",len(Nx1), len(Nx2), Nx1, Nx2)

                if sim:
                    #print("in opt 1")
                    wxp = np.where(case == 'possible')[0]
                    print(wxp)
                    xp = x#x[wxp]

                    print(Sx1, x, Nx1, np.setdiff1d(Nx1, np.concatenate((Sx1, x))))
                    ind1 = np.concatenate((Sx1, x, np.setdiff1d(Nx1, np.concatenate((Sx1, x)))))
                    print(Sx2, xp, Nx2)#, np.setdiff1d(Nx2, np.concatenate((Sx2, xp))))


                    ind2 = np.concatenate((Sx2, xp, np.setdiff1d(Nx2, np.concatenate((Sx2, xp)))))
                else:
                    #print("in opt 2")
                    ind1 = Sx1 + x + list(set(Nx1).difference(set(Sx1+x)))

                    ind2 = Sx2 + list(set(Nx2).difference(set(Sx2)))

                if verb:
                    print("seed = ", Sx1, ", matching ", ind1, " and ", ind2)

                #print("inside 2 w/ params Sx1 = ", Sx1, ", x = ", x, ", Nx1 = ", Nx1)
                #print("inside 2 w/ indlens ", len(ind1), len(ind2), ind1, ind2)
                n_iters = 20
                As = shuffle_adj_matrix(A, ind1)
                Bs = shuffle_adj_matrix(B, ind2)

                sgm = GMP(n_init=n_iters)

                # Note we must padd because the inputs are note guarenteed to be the same
                totv1 = As.shape[1]
                totv2 = Bs.shape[1]

    #             if (totv1>totv2):
    #                 #print("in pad 2")
    #                 diff = totv1-totv2
    #                 Bs = np.concatenate((Bs, np.full((Bs.shape[0], diff), pad)), axis=1)
    #                 Bs = np.concatenate((Bs, np.full((diff, Bs.shape[1]), pad)), axis=0)
    #             elif (totv1<totv2):
    #                 #print("in pad 2")
    #                 diff = totv2-totv1
    #                 As = np.concatenate((As, np.full((As.shape[0], diff), pad)), axis=1)
    #                 As = np.concatenate((As, np.full((diff, As.shape[1]), pad)), axis=0)

                self.corr = sgm.fit_predict(As, Bs, seeds_A=np.arange(len(Sx1)), seeds_B=np.arange(len(Sx1))) #multistart_sgm(A, B, R, len(Sx1), g, pad=pad, n_iters=n_iters)
                self.P = sgm.P_final
                self.Cx2 = list(Cx2)
                self.labelsGx = ind1
                self.labelsGxp = ind2
            else:
                self.ind1 = self.ind2 = self.P = self.corr = None
        else:
            self.ind1 = self.ind2 = self.P = self.Cx2= self.corr = None

        return self

    def fit(self, x, seeds, A, B, h, ell, R, g, pad=0, sim=False, verb=False, plotF=False):
        """
        Fits the model with two assigned adjacency matrices

        Parameters
        ----------
        x: int or ndarray
        verticies of interest (voi)

        seeds: ndarray
            seeds vector

        A: ndarray
            Adjacency matrix of `G_1`, the graph where voi is known

        B: ndarray
            Adjacency matrix of `G_2`, the graph where voi is not known

        h: int
            distance between voi used to create induced subgraph on `G_1`

        ell: int
            distance from seeds to other verts to create induced subgraphs on `G_1`

        R: int
            number of restarts

        g: float
            gamma to be used, max tol for alpha, tollerable dist from barycenter

        Returns
        -------
        self : returns an instance of self
        """

        nv1 = A.shape[0]
        nv2 = A.shape[0]

        nv = max(nv1, nv2)

        nsx1 = set(range(nv1)).difference(set([seeds[0], x]))

        vec = [seeds[0], x]; vec.extend(nsx1)

        AA = shuffle_adj_matrix(A, vec)

        ns2 = set(range(nv2)).difference(set([seeds[1]])) # <- setdiff(1:nv2,seeds[,2])
        vec2 = [seeds[1]]; vec2.extend(ns2) # <- c(seeds[,2],ns2)

        BB = shuffle_adj_matrix(B, vec2)

        nrow_seeds = seeds.shape[0]

        S = list(range(nrow_seeds))

        voi = list(range(nrow_seeds, nrow_seeds+2))

        P = self.vnsgm_ordered(voi,S,AA,BB,h,ell,R,g,pad=pad,sim=sim,verb=verb,plotF=plotF)

        self.x = x
        self.S = seeds

        if self.Cxp is not None:
            foo = []
            for kk in self.Cxp:
                foo.append(vec2[kk])
            self.Cxp = foo

        if self.labelsGx is not None:
            foo = []
            for kk in self.labelsGx:
                foo.append(vec[kk])
            self.labelsGx = foo

        if self.labelsGxp is not None:
            foo = []
            for kk in self.labelsGxp:
                foo.append(vec2[kk])
            self.labelsGxp = foo

        return self

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
        return self.perm_inds_

def ego(graph_adj_matrix, order, node, mindist=1):
    #Note all nodes are zero based in this implementation, i.e the first node is 0
    dists = [[node]]
    for ii in range(1, order+1):
        clst = []
        for nn in dists[-1]:
            clst.extend(list(np.where(graph_adj_matrix[nn]==1)[0]))
        clst = list(set(clst))

        #Remove all the ones that are closer (i.e thtat have already been included)
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

    for ii in range(mindist, order+1):
        ress.extend(dists[ii])

    return list(set(ress))

def ego_list(graph_adj_matrix, order, node, mindist=1):
    print(type(node))
    if type(node) == list:
        total_res = []
        for nn in node:
            ego_res = ego(graph_adj_matrix, order, nn, mindist=1)
            total_res.extend(ego_res)
        return list(set(total_res))
    else:
        return ego(graph_adj_matrix, order, node)
