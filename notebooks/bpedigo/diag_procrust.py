#%%
import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

def diag_procrust(X1, X2):
    normX1 = np.sum(np.square(X1), axis=1) 
    normX2 =  np.sum(np.square(X1), axis=1)
    normX1[normX1 <= 1e-15] = 1
    normX2[normX2 <= 1e-15] = 1
    X1 = np.divide(X1, np.sqrt(normX1[:, None]))
    X2 = np.divide(X1, np.sqrt(normX2[:, None]))
    R, s = orthogonal_procrustes(X1, X2)
    X2 = np.dot(X2, R)
    X2 = s / np.sum(X2**2) * X2
    return np.linalg.norm(X1 - X2)
#%%
points1 = np.array([[0, 0], 
                    [3, 0], 
                    [3, -2]], dtype=np.float64)
rotation = np.array([[0, 1],
                        [-1, 0]])
# rotated 90 degrees
points2 = np.dot(points1, rotation)
# diagonally scaled
diagonal = np.array([[2, 0, 0], 
                    [0, 3, 0],
                    [0, 0, 2]])
points2 = np.dot(diagonal, points2)
X1 = points1
X2 = points2
normX1 = np.sum(X1**2, axis=1) 
normX2 =  np.sum(X2**2, axis=1)
normX1[normX1 <= 1e-15] = 1
normX2[normX2 <= 1e-15] = 1
X1 = X1 / np.sqrt(normX1[:, None])
X2 = X2 / np.sqrt(normX2[:, None])
R, s = orthogonal_procrustes(X1, X2)
X1 = X1 @ R
# X2 = s / np.sum(X2**2) * X2
print(np.linalg.norm(X1 - X2))
#%%
# triangle in 2d
points1 = np.array([[0, 0], 
                    [3, 0], 
                    [3, -2]], dtype=np.float64)
rotation = np.array([[0, 1],
                        [-1, 0]])
# rotated 90 degrees
points2 = np.dot(points1, rotation)
# diagonally scaled
diagonal = np.array([[2, 0, 0], 
                    [0, 3, 0],
                    [0, 0, 2]])
points2 = np.dot(diagonal, points2)

#%%
print(points1)
print(points2)

#%%
n = diag_procrust(points1, points2)
print(n)
print(np.isclose(n, 0))

n = diag_procrust(points1, points2)
print(n)
print(np.isclose(n, 0))

#%%
