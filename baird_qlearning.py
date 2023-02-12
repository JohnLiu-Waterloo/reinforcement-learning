# ex 11.3

import numpy as np
import random

gamma = 0.99
alpha = 0.01
eps = 0.01

V = np.array([
    [2, 0, 0, 0, 0, 0, 0, 1],
    [0, 2, 0, 0, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, 0, 0, 1],
    [0, 0, 0, 2, 0, 0, 0, 1],
    [0, 0, 0, 0, 2, 0, 0, 1],
    [0, 0, 0, 0, 0, 2, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 2]
])
# Q = np.zeros((7, 2))

### Different initial values ###
theta = np.array([1, 1, 1, 1, 1, 1, 20, 1])
# theta = np.array([1, 1, 1, 1, 1, 1, 1, 1])
# this one converges
# theta = np.array([ 3.23, 3.23, 3.23, 3.23, 3.23, 3.23, 12.92, -6.46])

n_ep = 10000
DEBUG = False
s = random.randint(0, 6)
a = 0
q = np.sum(V[s] * theta)
ep = 0
while ep < n_ep:
    ep += 1
    s_new = random.randint(0, 6)
    q_new = np.sum(V[s_new] * theta)

    a = 1 if s_new == 6 else 0
    rho = 1 / (1/7) if s_new == 6 else 0 
    theta_new = theta + alpha * rho * (gamma * np.sum(V[s_new] * theta) - np.sum(V[s] * theta)) * V[s]
    # theta_new = theta + alpha * rho * (gamma * max(Q[s_new][0], Q[s_new][1]) - q) * V[s]
    if DEBUG and rho > 0:
        print(np.round(theta, 2))
        print(s, ",", V[s])
        print(s_new, ",", V[s_new])
        print("Diff,", (gamma * np.sum(V[s_new] * theta) - np.sum(V[s] * theta)))
        print("rho", rho)
        print(np.round(theta_new, 4))
        print("==================")
    s = s_new
    theta = theta_new
    q = q_new
    if ep % 500 == 0:
    # if True:
        print(np.round(theta, 2))