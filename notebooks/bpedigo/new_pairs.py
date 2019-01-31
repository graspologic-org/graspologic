#%%
# %matplotlib inline
from graspy.plot import *
from graspy.simulations import sbm
from graspy.embed import AdjacencySpectralEmbed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
g = sbm([50, 50], [[0.8, 0.2], [0.2, 0.8]])
ase = AdjacencySpectralEmbed()
X = ase.fit_transform(g)
labels = 25 * [0] + 25 * [1] + 25 * [2] + 24 * [-1] + [-2]
# pairplot(X, size=50, alpha=0.6)

plt.show()


#%%

edgeplot(g,labels=labels)
#%%
%matplotlib inline

#%%
# %matplotlib inline
labels.shape