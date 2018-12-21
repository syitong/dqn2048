import numpy as np

def _greedy(Q,s):
    qmax = np.max(Q(s))
    actions = []
    for i,q in enumerate(Q(s)):
        if q == qmax:
            actions.append(i)
    return actions

def greedy(Q,s):
    return np.random.choice(_greedy(Q,s))

def ep_greedy(Q,s,ep):
    # input(Q(s))
    if np.random.rand() < ep:
        return np.random.choice(len(Q(s)))
    else:
        return greedy(Q,s)
