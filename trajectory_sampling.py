import argparse
import numpy as np
import random

N_STATES = 10000
N_TIME = 50000
INTERVAL = 2000
eps = 0.1
alpha = 0.1


# transition matrix
T = [[random.randint(0, N_STATES-1) for j in range(2)] for i in range(N_STATES)]
# reward matrix
R = np.random.normal(0, 1, (N_STATES, 2))
# Q function
Q = np.zeros((N_STATES,2))


def compute_reward():
    s = 0
    reward = 0
    while True:
        a = 0 if Q[s][0] > Q[s][1] else 1
        s_new = T[s][a]
        reward += R[s][a]
        if random.random() < 0.1:
            return reward
        s = s_new


# on-policy sampling
def run_policy():
    s = 0
    for t in range(N_TIME):
        a = 0 if Q[s][0] > Q[s][1] else 1
        if random.random() < eps:
            a = random.randint(0, 1)

        s_new = T[s][a]
        Q[s][a] += alpha * (R[s][a] + np.max(Q[s_new]) - Q[s][a])

        # terminate with 0.1 probability 
        if random.random() < 0.1:
            s = 0
        else:
            s = s_new

        if t % INTERVAL == INTERVAL-1:
            print("time", t, "reward", compute_reward())


# uniform sampling
def run_uniform():
    t = 0
    while t < N_TIME:
        for s in range(N_STATES):
            a = 0 if Q[s][0] > Q[s][1] else 1
            if random.random() < eps:
                a = random.randint(0, 1)

            s_new = T[s][a]
            Q[s][a] += alpha * (R[s][a] + np.max(Q[s_new]) - Q[s][a])

            t += 1
            if t % INTERVAL == INTERVAL-1:
                print("time", t, "reward", compute_reward())


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--mode", help="uniform or policy")

    args = argParser.parse_args()
    if args.mode == "uniform":
        run_uniform()
    else:
        run_policy()

if __name__ == "__main__":
    main()