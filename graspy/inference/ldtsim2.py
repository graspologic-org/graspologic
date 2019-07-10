import numpy as np 
import matplotlib.pyplot as plt  
np.random.seed(8888)
import graspy

from graspy.inference import LatentDistributionTest
from graspy.embed import AdjacencySpectralEmbed
from graspy.simulations import sbm, rdpg
from graspy.utils import symmetrize
from graspy.plot import heatmap
import seaborn as sns 
from pandas import DataFrame

B = [[0.5, 0.2],
     [0, 0.05]]
B = symmetrize(B)
k = 2
tests = 1
start = 50
stop = 451
diff = 100
ns = []
ms = []
newms = []
error_list = []
temp = []

for n in range(start, stop, diff):
    ns.append(n)
    for m in range(n, n+stop-start, diff):
        cn = [n//k] * k
        cm = [m//k] * k
        A1 = sbm(cn, B)
        A2 = sbm(cm, B)
        valid = 0
        ldt = LatentDistributionTest(n_components=2)
        for p in range(tests):
            p = ldt.fit(A1, A2)
            if(p < 0.05):
                valid += 1
        error = valid/tests
        temp.append(error)
        ms.append(m-n)
    error_list.append(temp)
    temp = []

for num in ms:
    if num not in newms:
        newms.append(num)

df = DataFrame(error_list, index=ns, columns=newms)
sns.heatmap(df, annot=True, linewidths=0.5)
plt.title("Variation of Type 1 Error with Different Cases of LDT")
plt.xlabel("m - n")
plt.ylabel("n")
plt.show()
