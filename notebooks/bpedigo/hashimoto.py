#%%

import numpy as np
import graspy


def to_hashimoto(graph):
    # for every pair of EDGES
    # i to j, k to l
    # m is number of edges without self loops n(n-1)
    # directed network with 2, edges
    """
    for e1, edge1 = every edge not on the diag:
        for e2, edge2 = every edge not on the diag:
            i = edge1 row
            j = edge1 column
            k = edge2 row
            l = edge2 column
            if j == k and i != l: 
                B[e1, e2] = 1
    """
    row, col = np.where(graph != 0)
    edges = graph[row, col]
    m = len(row)
    B = np.zeros((m, m))
    for e1, edge1 in enumerate(edges):
        for e2, edge2 in enumerate(edges):
            i = row[e1]
            j = col[e1]
            k = row[e2]
            l = col[e2]
            if j == k and i != l:
                B[e1, e2] = 1

    return B, col


n = [200, 200]
p = [[0.1, 0.01], [0.01, 0.07]]
A = graspy.simulations.sbm(n, p, directed=False)
graspy.plot.heatmap(A)
len(A[A != 0])

B, col = to_hashimoto(A)

X, Y = ase.fit_transform(A)
graspy.plot.pairplot(np.concatenate((X, Y), axis=1), labels=200 * ["1"] + 200 * ["2"])

#%%
ase = graspy.embed.AdjacencySpectralEmbed(n_components=4)

X, Y = ase.fit_transform(B)
X = np.concatenate((X, Y), axis=1)
nodes_vec = np.zeros((A.shape[0], X.shape[1]))
for dim in range(X.shape[1]):
    s = X[:, dim]
    for c, x in zip(col, s):
        nodes_vec[c, dim] = x
        # if x > 0:

        # if x < 0:
        #     nodes_vec[dim,c] -=


graspy.plot.pairplot(nodes_vec)
#%%

labels = np.array(200 * ["1"] + 200 * ["2"])
l = labels[col]

graspy.plot.pairplot(X, labels=l)
