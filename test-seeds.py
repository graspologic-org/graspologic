# %%
import numpy as np 
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes, SeededProcrustes
import time
import matplotlib.pyplot as plt
from graspologic.plot import heatmap
from scipy.stats import special_ortho_group, ortho_group

# %%
np.random.seed(314) #creating a seed (what does input change?)
X = np.random.uniform(0, 1, (10, 2)) # creating an X to test
Q = special_ortho_group.rvs(2) # Q is the transform matrix (random)
Y = X @ Q # setting Y to X times Q 
inds = [1,0,3,4,2]
Y[inds,:]

# %%
seeds = np.array([[0,1,2,3,4],[1,0,3,4,2]]) # creating a matrix of seeds
aligner_SP = SeededProcrustes() # creating new instance of seeded procrustes object
X_prime_SP = aligner_SP.fit_transform(X, Y, seeds) #aligning X with Y and transforming X using Q_
aligner_SP.fit(X,Y,seeds,verbose=False)
# 10 points - 5 seeds

# %%
heatmap(aligner_SP.Q_, figsize=(4,4), vmin=-1, vmax=1)
aligner_SP.Q_
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_title("SeededProcrustes, Simple rotation example")
ax.scatter(X[:,0], X[:,1], label=r"$X$", marker='x')
ax.scatter(Y[:,0], Y[:,1], label=r"$Y$", s=100, alpha=0.5)
ax.scatter(X_prime_SP[:,0],
           X_prime_SP[:,1],
           label=r"$X_{SP}$",
           marker='$S$')
ax.set(xlim=(-1.20, 1.20), ylim=(-1.20, 1.20))
ax.set_xticks([0])
ax.set_yticks([0])
ax.legend()
ax.grid()

