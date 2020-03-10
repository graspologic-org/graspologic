import numpy as np
import matplotlib.pyplot as plt
import random

from .gmp import GraphMatch as GMP
from graspy.simulations import sbm_corr


def match_ratio(v, w):
    return 1 - (np.count_nonzero(abs(v - w)) / n)


# code for constructing the matrix to recreate figure 2 from SGM
# rows are seeds, columns are rhos

n = 300
m = range(21)
rhos = 0.1 * np.arange(11)
ratios = np.zeros((21, 11))
n_per_block = 100
n_blocks = 3
block_members = np.array(n_blocks * [n_per_block])
block_probs = np.array([[0.7, 0.3, 0.4], [0.3, 0.7, 0.3], [0.4, 0.3, 0.7]])
directed = False
loops = False

for k in range(11):
    rho = rhos[k]
    for i in m:
        sums = 0
        for j in range(2):
            A1, A2 = sbm_corr(
                block_members, block_probs, rho, directed=directed, loops=loops
            )
            node_shuffle_input = np.random.permutation(n)
            A2_shuffle = A2[np.ix_(node_shuffle_input, node_shuffle_input)]
            node_unshuffle_input = np.array(range(n))
            node_unshuffle_input[node_shuffle_input] = np.array(range(n))

            W1 = np.sort(random.sample(list(range(n)), i))
            W1 = W1.astype(int)
            W2 = np.array(node_unshuffle_input[W1])

            faq = GMP(gmp=True)
            faq = faq.fit(A1, A2_shuffle, W1, W2)
            sums += match_ratio(faq.perm_inds_, node_unshuffle_input)

        ratios[i, k] = sums / 2

for i in range(11):
    plt.plot(m, ratios[:, i], "-o", label=str(rhos[i]))

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
plt.xlabel("seeds (m)")
plt.ylabel("match ratio")
plt.savefig(
    "./graspy/match/fig2.png",
    fmt="png",
    dpi=150,
    facecolor="w",
    bbox_inches="tight",
    pad_inches=0.3,
)
