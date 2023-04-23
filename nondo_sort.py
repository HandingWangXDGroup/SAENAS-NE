import numpy as np

def NonDominatedSorting(pop):
    """Perform non-dominated sorting.

    :param pop: the current population
    :type pop: array
    """
    _,npop  = pop.shape
    rank = np.zeros(npop)
    dominatedCount = np.zeros(npop)
    dominatedSet = [[] for i in range(npop)]
    F = [[]]
    for i in range(npop):
        for j in range(i + 1, npop):
            p = pop[:,i]
            q = pop[:,j]
            if Dominates(p, q):
                dominatedSet[i].append(j)
                dominatedCount[j] += 1
            if Dominates(q, p):
                dominatedSet[j].append(i)
                dominatedCount[i] += 1
        if dominatedCount[i] == 0:
            rank[i] = 0
            F[0].append(i)
    k = 0
    while (True):
        Q = []
        for i in F[k]:
            p = pop[:,i]
            for j in dominatedSet[i]:
                dominatedCount[j] -= 1
                if dominatedCount[j] == 0:
                    Q.append(j)
                    rank[j] = k + 1
        if len(Q) == 0:
            break
        F.append(Q)
        k += 1
    return F,rank

def Dominates(x, y):
    """Check if x dominates y.

    :param x: a sample
    :type x: array
    :param y: a sample
    :type y: array
    """
    return np.all(x >= y) & np.any(x > y)