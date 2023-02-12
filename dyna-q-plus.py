from collections import defaultdict
import math
import numpy as np
import random

DEBUG = False

n = 25
eps = 0.1
gamma = 0.95
alpha = 0.1
PLUS_PLANNING = False
K = 1e-4

grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0]
]

# actions: 0 right, 1 left, 2 down, 3 up

def get_plan(Q):
    actions = ['r', 'l', 'd', 'u']
    for y in range(6):
        res = ""
        for x in range(9):
            a_best = 0
            for a in range(1, 4):
                if Q[y][x][a] > Q[y][x][a_best]:
                    a_best = a
            res += actions[a_best]
        print(res)


def move(S, A):
    y, x = S
    if A == 0:
        x += 1
    elif A == 1:
        x -= 1
    elif A == 2:
        y += 1
    elif A == 3:
        y -= 1

    if x < 0 or x >= 9 or y < 0 or y >= 6 or grid[y][x] == 3:
        y, x = S

    return y, x


Q = np.zeros((6, 9, 4))
L = np.zeros((6, 9, 4))
M = [[[(0, (i, j)) for k in range(4)] for j in range(9)] for i in range(6)]

start = (5, 3)
end = (0, 8)

S = start
nsteps = 6000

run = 0
seen = defaultdict(set)
reward = 0
for step in range(nsteps):
    if step == 3000:
        grid[3][0] = 3
        grid[3][8] = 0
    if PLUS_PLANNING:
        max_value = 0
        for a in range(4):
            val = Q[S[0]][S[1]][a] + K * math.sqrt(step - L[S[0]][S[1]][a])
            if val > max_value:
                A = a
                max_value = val

    else:
        # select A using epsilon greedy
        A = random.randint(0,3)
        if random.random() > eps:
            # if multiple actions yield same Q-value, then choose one at random
            actions = []
            max_value = np.max(Q[S[0]][S[1]])
            for a in range(4):
                if Q[S[0]][S[1]][a] == max_value:
                    actions.append(a)
            A = actions[random.randint(0,len(actions)-1)]

    L[S[0]][S[1]][A] = step
    seen[S].add(A)

    y, x = move(S, A)
    R = 0
    if (y, x) == end:
        R = 1

    M[S[0]][S[1]][A] = (R, (y, x))
    Q[S[0]][S[1]][A] += alpha * (R + gamma * np.max(Q[y][x]) - Q[S[0]][S[1]][A])

    if DEBUG:
        print("S", S, "A", A)
        print("Q(S)", Q[S[0]][S[1]])
        print("R", R, "S'", y, x)
        a_new = 0
        for a in range(1, 4):
            if Q[y][x][a] > Q[y][x][a_new]:
                a_new = a
        print("Q(S')", Q[y][x], "max a", a_new)

    if (y, x) == end:
        S = start
        reward += 1
        run = 0
    else:
        S = (y, x)
        run += 1

    for i in range(n):
        SS = list(seen.keys())[random.randint(0, len(seen)-1)]
        AA = random.randint(0, 3)       # consider all possible actions
        RR, SS_new = M[SS[0]][SS[1]][AA]
        if not PLUS_PLANNING:
            RR += K * math.sqrt(step-L[SS[0]][SS[1]][AA])

        if DEBUG:
            aa_new = 0
            for aa in range(1, 4):
                if Q[SS_new[0]][SS_new[1]][aa] > Q[SS_new[0]][SS_new[1]][aa_new]:
                    aa_new = aa
        Q[SS[0]][SS[1]][AA] += alpha * (RR + gamma * np.max(Q[SS_new[0]][SS_new[1]]) - Q[SS[0]][SS[1]][AA])

    if step % 500 == 499:
        print(step, ":", reward)

print(Q.shape)
print(Q.transpose(2, 0, 1).round(2))
print(L.transpose(2, 0, 1).round(2))
get_plan(Q)
if DEBUG:
    for y in range(6):
        for x in range(9):
            print(y, x, M[y][x])
    print(seen)