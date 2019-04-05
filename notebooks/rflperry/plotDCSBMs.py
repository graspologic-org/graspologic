import numpy as np
import graspy
import matplotlib.pyplot as plt

n = [5,5]
p = [[0.7,0.3],[0.3,0.7]]

dc = np.random.power#np.ones(10)/10

def checkBlockError(i,j,n,p,A):
    iidx = range(sum(n[:i]),sum(n[:i+1]))
    jidx = range(sum(n[:j]),sum(n[:j+1]))
    trueP = p[i][j]
    ones = sum([A[k][l] for k in iidx for l in jidx])
    simP =  ones / (len(iidx)*len(jidx))
    return (trueP-simP)**2


errors = np.empty((0,4), float)
ns = [i for i in range(1,51)]
for k in ns:
    n = [k,k]
    dcProbs = np.array([np.random.power(a=3) for i in range(sum(n))])
    for k in range(len(n)):
        dcProbs[range(sum(n[:k]),sum(n[:k+1]))] /= sum(dcProbs[range(sum(n[:k]),sum(n[:k+1]))])
    A = graspy.simulations.sbm(n, p, directed=False, loops=False, wt=1, wtargs=None, dc=dcProbs, dcargs={'a':3})
    errors = np.append(errors, np.array([[checkBlockError(i,j,n,p,A) for i in range(len(n)) for j in range(len(n))]]), axis=0)

for err, label in zip(np.transpose(errors),['Block 0,0','Block 0,1','Block 1,0','Block 1,1']):
    plt.plot(ns,err, label=label)
plt.legend()
plt.show()

