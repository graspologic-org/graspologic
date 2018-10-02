import unittest
import graphstats as gs
import numpy as np
import networkx as nx
from graphstats.embed.ase import AdjacencySpectralEmbed
from graphstats.embed.lse import LaplacianSpectralEmbed  
from graphstats.simulations.simulations import er_np, er_nm, weighted_sbm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def _kmeans_comparison(data, labels, n_clusters):
    """
    Function for comparing the ARIs of kmeans clustering for arbitrary number of data/labels

    Parameters
    ----------
        data: list-like 
            each element in the list is a dataset to perform k-means on 
        labels: list-like
            each element in the list is a set of lables with the same number of points as 
            the corresponding data
        n_clusters: int
            the number of clusters to use for k-means 
    
    Returns
    -------
        aris: list, length the same as data/labels
            the i-th element in the list is an ARI (Adjusted Rand Index) corresponding to the result 
            of k-means clustering on the i-th data/labels

    """
    
    if len(data) != len(labels): 
        raise ValueError('Must have same number of labels and data')

    aris = []
    for i in range(0, len(data)):
        kmeans_prediction = KMeans(n_clusters=n_clusters).fit_predict(data[i])
        aris.append(adjusted_rand_score(labels[i], kmeans_prediction))
    
    return aris

def _test_output_dim(self, method, *args, **kwargs):
    k = 4
    embed = method(k=k)
    n = 10
    M = 20
    A = er_nm(n, M) + 5
    embed._reduce_dim(A)
    self.assertEqual(embed.lpm.X.shape, (n, 4))
    self.assertTrue(embed.lpm.is_symmetric())

def _test_sbm_er_binary_undirected(self, method, P, *args, **kwargs):
    np.random.seed(8888)
        
    num_sims = 50
    verts = 200
    communities = 2 

    verts_per_community = [100, 100]
    
    sbm_wins = 0
    er_wins = 0
    for sim in range(0, num_sims):
        sbm = weighted_sbm(verts_per_community, P)
        er = er_np(verts, 0.5)
        embed_sbm = method(k=2)
        embed_er = method(k=2)

        labels_sbm = np.zeros((verts), dtype=np.int8)
        labels_er = np.zeros((verts), dtype=np.int8)
        labels_sbm[100:] = 1
        labels_er[100:] = 1

        embed_sbm.fit(sbm)
        embed_er.fit(er)

        X_sbm = embed_sbm.lpm.X
        X_er = embed_er.lpm.X

        self.assertEqual(X_sbm.shape, (verts, communities))
        self.assertEqual(X_er.shape, (verts, communities))

        aris = _kmeans_comparison((X_sbm, X_er), (labels_sbm, labels_er), communities)
        sbm_wins = sbm_wins + (aris[0] > aris[1])
        er_wins = er_wins + (aris[0] < aris[1])
    #print(sbm_wins)
    #print(er_wins)
    self.assertTrue(sbm_wins > er_wins)    

class TestAdjacencySpectralEmbed(unittest.TestCase):
    
    def test_output_dim(self):
        _test_output_dim(self, AdjacencySpectralEmbed)

    def test_sbm_er_binary_undirected(self):
        P = np.array([[0.8, 0.2], [0.2, 0.8]])
        _test_sbm_er_binary_undirected(self, AdjacencySpectralEmbed, P)

class TestLaplacianSpectralEmbed(unittest.TestCase):

    def test_output_dim(self):
        _test_output_dim(self, LaplacianSpectralEmbed)
    
    def test_sbm_er_binary_undirected(self):
        P = np.array([[0.8, 0.2], [0.2, 0.3]])
        _test_sbm_er_binary_undirected(self, LaplacianSpectralEmbed, P)

if __name__ == '__main__': 
    unittest.main()

        