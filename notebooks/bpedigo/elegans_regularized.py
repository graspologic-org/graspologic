#%%
from graspy.embed import LaplacianSpectralEmbed
from graspy.plot import pairplot
from graspy.utils import pass_to_ranks
import pandas as pd
import numpy as np

filepath = "~/JHU_code/elegans/nice_data/herm_chem_A_self_undirected.csv"
metadata_path = "~/JHU_code/elegans/nice_data/master_cells.csv"
cell_path = "~/JHU_code/elegans/nice_data/herm_chem_self_cells.csv"
metadata_df = pd.read_csv(metadata_path)
metadata_df = metadata_df.set_index("name")
cells = np.squeeze(pd.read_csv(cell_path, header=None).values)

#%%
p_labels = metadata_df.loc[cells, "pharynx"].fillna("nonpharynx")
p_labels = p_labels.values
adj = pd.read_csv(filepath, header=None)
adj = adj.values
adj = pass_to_ranks(adj)
for l in np.logspace(0, 1, 5):
    lse = LaplacianSpectralEmbed(form="R-DAD", regularizer=l)
    latent = lse.fit_transform(adj)
    pairplot(latent, labels=p_labels)  #%%
#%%

from graspy.plot import heatmap

heatmap(adj, inner_hier_labels=p_labels)
#%%
im
adj_p = adj + 0.1
ase = AdjacencySpectralEmbed
