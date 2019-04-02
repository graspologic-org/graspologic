#%%

from graspy.models import EREstimator, SBEstimator, RDPGEstimator
from graspy.datasets import load_drosophila_left
from graspy.plot import heatmap

left_adj, cell_labels = load_drosophila_left(return_labels=True)
left_adj_uw = left_adj.copy()
left_adj_uw[left_adj_uw > 0] = 1

heatmap(left_adj_uw, inner_hier_labels=cell_labels)

er = EREstimator(fit_degrees=False)
er.fit(left_adj)
print(er.degree_corrections_)
heatmap(er.sample())


#%%
