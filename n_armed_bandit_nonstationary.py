# Section 2.7

import numpy as np
import random

n = 10
N = 100
eps = 0.1
alpha = 0.1


def run_tasks(T, sample_average=False):
    n_best = 0              # number of tasks where we picked optimal solution
    total_avg_reward = 0    # sum of average reward of each task
    pct_best = []
    for ncase in range(0, N):
        # Set up reward
        reward = [0 for i in range(n)]    # mean reward for action i
        # print(reward)

        nt = [0 for i in range(n)]      # number of trials for action i
        Qt = np.ones(n)

        for t in range(T):
            ##### episolon-greedy policy #####
            if random.random() < eps:
                a = random.randrange(0, n)
                # print("explore {}".format(a))
            else:
                a = np.argmax(Qt)

            r = np.random.normal(reward[a], 1)
            nt[a] = nt[a] + 1

            if sample_average:
                Qt[a] = Qt[a] +  1/nt[a] * (r - Qt[a])
            else:
                Qt[a] = Qt[a] +  alpha * (r - Qt[a])

            # random walk for all rewards
            reward = [r + 0.1 if random.random() < 0.5 else r - 0.1 for r in reward]

            # print("Action {}, reward {}".format(a, r))
            # print(Qt)

        best_choice = np.argmax(Qt)
        if best_choice == reward.index(max(reward)):
            n_best += 1
        pct_best.append(n_best / (ncase + 1))

        avg_reward = sum([Qt[i] * nt[i] for i in range(n)]) / T
        total_avg_reward += avg_reward

        # print("Best choice: {}, average reward: {}".format(best_choice, avg_reward))
        # print(np.round(Qt, 2))
        # print(np.round(reward, 2))

    return pct_best[-1], total_avg_reward / N 

for T in [10, 50, 100, 200, 1000, 2000]:
    print("Time: ", T)
    p, r = run_tasks(T, True)
    print("Sample average: best ", p, " reward ", r)

    p, r = run_tasks(T, False)
    print("Constant step size: best ", p, " reward ", r)

