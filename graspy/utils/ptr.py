import numpy as np
from .utils import import_graph, is_unweighted, is_symmetric, is_loopless, symmetrize
from scipy.stats import rankdata


def pass_to_ranks(graph, method='zero-boost'):
    """
    Rescales edge weights of an adjacency matrix based on their relative rank in 
    the graph. 

    Parameters
    ----------
        graph: Adjacency matrix 
        
        method: string, optional
            'zero-boost': preserves the edge weight for all 0s, but ranks the other
                edges as if the ranks of all 0 edges has been assigned. If there are 
                10 0-valued edges, the lowest non-zero edge gets weight 11 / (number
                of possible edges). Ties settled by the average of the weight that those
                edges would have received. Number of possible edges is determined 
                by the type of graph (loopless or looped, directed or undirected)
            'simple-all': assigns ranks to all non-zero edges, settling ties using 
                the average. Ranks are then scaled by 
                    .. math:: \frac{2 rank(non-zero edges)}{n^2 + 1}
                where n is the number of nodes
            'simple-nonzero':
                same as 'simple-all' but ranks are scaled by
                    .. math:: \frac{2 rank(non-zero edges)}{num_nonzero + 1}

    See also
    --------
        scipy.stats.rankdata

    Returns
    ------- 
        graph: numpy.ndarray, shape(n_vertices, n_vertices)
            Adjacency matrix of graph after being passed to ranks
    """ 
    
    graph = import_graph(graph) # just for typechecking

    if is_unweighted(graph):
        return graph

    if method == 'zero-boost':
        if is_symmetric(graph):
            # start by working with half of the graph, since symmetric
            triu = np.triu(graph)
            non_zeros = triu[triu != 0]
            rank = rankdata(non_zeros)
            
            num_zeros = 0
            possible_edges = 0
            if is_loopless(graph):
                num_zeros = (len(graph[graph == 0]) - graph.shape[0]) / 2
                possible_edges = graph.shape[0] * (graph.shape[0] - 1) / 2 
            else: 
                num_zeros = (len(triu[triu == 0]) - graph.shape[0] * (graph.shape[0] - 1) / 2) 
                possible_edges = graph.shape[0] * (graph.shape[0] + 1) / 2
            
            # shift up by the number of zeros 
            rank = rank + num_zeros

            # normalize by the number of possible edges for this kind of graph
            rank = rank / possible_edges

            # put back into matrix form and reflect over the diagonal
            triu[triu != 0] = rank 
            graph = symmetrize(triu, method='triu')
            
            return graph
        else: 
            raise NotImplementedError()

    elif method in ['simple-all', 'simple-nonzero']:
        if is_symmetric(graph): 
            non_zeros = graph[graph != 0]
            rank = rankdata(non_zeros)

            normalizer = 1
            if method == 'simple-all':
                normalizer = graph.shape[0] ** 2
            elif method =='simple-nonzero':
                normalizer = rank.shape[0]

            rank = rank * 2/ (normalizer + 1)

            graph[graph != 0] = rank
            return graph    
        else: 
            raise NotImplementedError()       
    
    else: 
        raise ValueError('Unsuported pass-to-ranks method')
    