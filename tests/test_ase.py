import unittest
import graphstats as gs
import numpy as np
import networkx as nx
from graphstats.embed.ase import ASEEmbed
from graphstats.simulations.simulations import er_np, er_nm, weighted_sbm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def kmeans_comparison(data, labels, n_clusters):
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


class TestASEEmbed(unittest.TestCase):
    
    def test_ase_er(self):
        k = 3
        embed = ASEEmbed(k=k)
        n = 10
        M = 20
        A = er_nm(n, M) + 5
        embed._reduce_dim(A)
        self.assertEqual(embed.lpm.X.shape, (n, k))
        self.assertTrue(embed.lpm.is_symmetric())

    def test_sbm_er_binary_undirected(self):
        num_sims = 50
        verts = 200
        communities = 2 

        verts_per_community = [100, 100]
        P = np.array([[0.8, 0.2], [0.2, 0.8]])

        for sim in range(0, num_sims):
            sbm = weighted_sbm(verts_per_community, P)
            er = er_np(verts, 0.5)

            embed_sbm = ASEEmbed(k=2)
            embed_er = ASEEmbed(k=2)

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

            aris = kmeans_comparison((X_sbm, X_er), (labels_sbm, labels_er), communities)
            
            self.assertTrue(aris[0] > aris[1])
        

if __name__ == '__main__': 
    unittest.main()

        