import numpy as np
import random

lbda = 0.9
alpha = 0.1
eps = 0.1
gamma = 0.9

L = 6
E = np.zeros((L, 2))
Q = np.zeros((L, 2))

for ep in range(0, 10):
    s = 0
    a = random.randint(0, 1)
    while s != L - 1:
        # print("s: ", s, "a:", a)
        ## Update state
        s_new = s+1 if a == 1 else s
        r = 1 if s_new == L-1 else 0
        
        ## Select action from episode-greedy policy
        a_new = 0 if Q[s_new][0] >= Q[s_new][1] else 1
        if random.random() < eps:
            a_new = random.randint(0, 1)

        ## Calculate trace
        delta = r + gamma * Q[s_new][a_new] - Q[s][a]
        ## Accumulating trace
        # E[s][a] += 1
        ## Replacing trace
        E[s][a] = 1

        Q = Q + alpha * delta * E
        E = gamma * lbda * E
        s, a = s_new, a_new

    print(Q)
    error = 0
    for i in range(5):
        if Q[i][0] >= Q[i][1]:
            error += 1
    print("Episode", ep, "error", error)

