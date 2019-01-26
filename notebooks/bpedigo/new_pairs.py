#%%

from graspy.plot import pairplot
from graspy.simulations import sbm
from graspy.embed import AdjacencySpectralEmbed
import numpy
import matplotlib.pyplot as plt
g = sbm([50, 50], [[0.8, 0.2], [0.2, 0.8]])
ase = AdjacencySpectralEmbed()
X = ase.fit_transform(g)
labels = 50 * [0] + 50 * [1]
pairplot(X, size=50, alpha=0.6)

plt.show()
#%%
