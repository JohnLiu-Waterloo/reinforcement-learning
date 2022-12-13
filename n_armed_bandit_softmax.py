# Section 2.3

import matplotlib.pyplot as plt 
import numpy as np
import random

n = 10
N = 100

def run_tasks(T, temp):
    n_best = 0              # number of tasks where we picked optimal solution
    total_avg_reward = 0    # sum of average reward of each task
    pct_best = []
    for ncase in range(0, N):
        # Set up reward
        reward = [np.random.normal(0, 1) for i in range(n)]    # mean reward for action i
        best_action = reward.index(max(reward))
        # print(reward)

        nt = [0 for i in range(n)]      # number of trials for action i
        Qt = np.ones(n)

        for t in range(T):
            #### Softmax policy with temperature temp ####
            prob = np.exp(Qt / temp)
            sum_exp = sum(prob)
            prob = prob / sum_exp
            a = random.choices(range(n), prob)[0]
            # print(Qt)
            # print(prob)
            # print(a)

            r = np.random.normal(reward[a], 1)
            nt[a] = nt[a] + 1
            Qt[a] = Qt[a] +  1/nt[a] * (r - Qt[a])

            # print("Action {}, reward {}".format(a, r))
            # print(Qt)

        avg_reward = sum([Qt[i] * nt[i] for i in range(n)]) / T
        # print("Best action: {}, average reward: {}".format(a, avg_reward))
        if a == best_action:
            n_best += 1
        total_avg_reward += avg_reward
        pct_best.append(n_best / (ncase + 1))

    return pct_best[-1], total_avg_reward / N 

for T in [10, 50, 100, 200, 1000, 2000]:
    print("Time ", T)
    for temp in [0.5, 0.2, 0.15, 0.1, 0.08, 0.05]:
        p, r = run_tasks(T, temp)
        print("Temperature ", temp, " best ", p, " reward ", r)

