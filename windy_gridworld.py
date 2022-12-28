import numpy as np
import random

DEBUG = False

wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
X = len(wind)
Y = 7

Q = np.zeros((Y, X, 9), dtype=int)
P = np.full((Y, X), 2, dtype=int)      # initial policy is to move right

start = (0, 3)
end = (7, 3)

eps = 0.1
alpha = 0.1

def print_action(a):
    return str(a%3-1) + "/" +str(int(a/3)-1)

for ep in range(2000):
    if ep % 50 == 0 and ep > 0:
        print("Episode", ep)
    px, py = start
    # nmoves = 0
    while (px != end[0] or py != end[1]):
    # while (px != end[0] or py != end[1]) and nmoves < X * Y * 5:
        # nmoves += 1
        ## derive action from policy
        a = P[py][px]
        if random.random() < eps:
            # king's move with stationary option
            a = random.randrange(0, 8)

        if DEBUG: 
            print(px, ",", py, ", policy:", print_action(P[py][px]), ", action:", print_action(a))
        
        ## take action and apply wind
        dx = int(a % 3) - 1
        dy = int(a / 3) - 1
        wind_applied = wind[px] + int(random.random() * 3.0) - 1
        # bound in grid
        px_next = max(min(X-1, px + dx), 0)
        py_next = max(min(Y-1, py + dy + wind_applied), 0)
        if DEBUG:
            print(px_next, py_next)

        # update Q
        a_next = P[py_next][px_next]  # derive next action from policy 
        Q[py][px][a] = Q[py][px][a] + alpha * (-1 + Q[py_next][px_next][a_next] - Q[py][px][a])

        ## update policy
        best_move = a
        for i in range(9):
            if Q[py][px][i] > Q[py][px][best_move]:
                best_move = i
        P[py][px] = best_move

        py, px = py_next, px_next

## print optimal policy
px, py = start
while px != end[0] or py != end[1]:
    a = P[py][px]
        
    ## take action and apply wind
    dx = int(a % 3) - 1
    dy = int(a / 3) - 1
    print(px, ",", py, ", policy:", print_action(P[py][px]))

    wind_applied = wind[px] # + int(random.random() * 3.0) - 1
    px_next = max(min(X-1, px + dx), 0)
    py_next = max(min(Y-1, py + dy + wind_applied), 0)

    px, py = px_next, py_next

"""
0 , 3 , policy: 1/-1
1 , 2 , policy: 1/-1
2 , 1 , policy: 1/-1
3 , 0 , policy: 1/-1
4 , 0 , policy: 1/0
5 , 1 , policy: 1/-1
6 , 1 , policy: 1/0
"""