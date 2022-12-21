import numpy as np
import random

# 0: off track
# 1: track
# 2: start
# 3: finish
track = [
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
]
track.reverse()

def print_track(x, y):
    for j in range(Y):
        t = ""
        for i in range(X):
            if i == x and Y - j - 1 == y:
                t += "c"
            else:
                t += str(track[Y-j-1][i])
        print(t)

X, Y = len(track[0]), len(track)
V = 5
eps = 1
DEBUG = False

vx, vy = 0, 0

# states: px, py, vx, vy
# actions: 3, 3
# returns: S * A
Q = np.full((Y, X, V + 1, V + 1, 3, 3), -1000)
N = np.zeros((Y, X, V + 1, V + 1, 3, 3))

# policy: return the best dx, dy for a state
# default policy accelerates y by 1
P = np.full((Y, X, V + 1, V + 1), 5)

num_ep = 10000
for ep in range(num_ep):
    # decay epsilon to encourage exploration at beginning and exploitation at end
    if ep > num_ep / 20:
        eps = 0.2
    elif ep > num_ep / 10:
        eps = 0.05
    elif eps > num_ep / 2:
        eps = 0.01

    # initialize world
    px, py = random.randint(1, 6) + 2, 0
    vx, vy = 0, 0
    states = []
    reward = 0
    finished = False

    while not finished:
        ## take action to modify velocity
        while True:
            tmp = P[py][px][vy][vx]
            # explore random option with probability eps
            if random.random() < eps:
                tmp = random.randint(0, 8)
            
            vx_old, vy_old = vx, vy
            dx, dy = int(tmp/3) - 1, tmp % 3 - 1
            # velocity has to be between 0 and 5
            vx = min(max(0, vx + dx), 5)
            vy = min(max(0, vy + dy), 5)

            if vx == 0 and vy == 0:
                vx, vy = vx_old, vy_old
            else:
                if DEBUG:
                    print_track(px, py)
                    print("x: ", px, ", y: ", py, ", vx: ", vx, ", vy: ", vy)
                break
            
        ## drive car to new location and add random displacement
        px_old, py_old = px, py
        px, py = vx + px, vy + py
        if random.random() < 0.5:
            if random.random() < 0.5:
                px += 1
            else:
                py += 1
        
        ## calculate reward
        r = 0
        if py >= Y-5 and px >= X-1:
            r = -5 if py >= Y else -1
            finished = True
        elif px < 0 or py < 0 or px >= X or py >= Y:
            px = min(max(0, px), X-1)
            py = min(max(0, py), Y-1)
            r = -5
        else:
            if track[py][px] == 0:
                r = -5
            else:
                r = -1

        states.append((px_old, py_old, vx_old, vy_old, dx, dy, r))
        reward += r
        if DEBUG:
            print("Reward ", r)
        
        ## move car back on track if needed
        if not finished and track[py][px] == 0:
            if DEBUG:
                print("Move ", px, ", ", py, " back on track")
            if px >= 8:
                while track[py][px] == 0:
                    px -= 1
            else:
                while track[py][px] == 0:
                    px += 1
            if DEBUG:
                print("Moved to ", px, ", ", py)
        
        if finished:
            # update reward for all states
            update_reward = 0
            states.reverse()
            for state in states:
                pxx, pyy, vxx, vyy, dxx, dyy, rr = state
                update_reward += rr
                # incremental update for Q(s, a)
                N[pyy][pxx][vyy][vxx][dyy+1][dxx+1] += 1
                if N[pyy][pxx][vyy][vxx][dyy+1][dxx+1] == 1:
                    Q[pyy][pxx][vyy][vxx][dyy+1][dxx+1] = update_reward
                else:
                    Q[pyy][pxx][vyy][vxx][dyy+1][dxx+1] += 1 / N[pyy][pxx][vyy][vxx][dyy+1][dxx+1] * (update_reward - Q[pyy][pxx][vyy][vxx][dyy+1][dxx+1])
            
                # update policy
                best_move = 0
                best_value = Q[pyy][pxx][vyy][vxx][0][0]
                for i in range(1, 9):
                    cur_value = Q[pyy][pxx][vyy][vxx][i%3][int(i/3)]
                    if cur_value > best_value:
                        best_value = cur_value
                        best_move = i
                P[pyy][pxx][vyy][vxx] = best_move
                if DEBUG:
                    print("x: ", pxx, ", y: ", pyy, ", vx: ", vxx, ", vy: ", vyy,
                        " - dx: ", int(best_move/3)-1, ", dy: ", best_move % 3 - 1)

# Run sim
nsims = 2
for sim in range(nsims):
    finished = False
    px, py = random.randint(1, 6) + 2, 0
    vx, vy = 0, 0
    while not finished:
        ## take action to modify velocity
        tmp = P[py][px][vy][vx]
        dx, dy = int(tmp/3) - 1, tmp % 3 - 1
        # velocity has to be between 0 and 5
        vx = min(max(0, vx + dx), 5)
        vy = min(max(0, vy + dy), 5)
        print("x: ", px, ", y: ", py, ", vx: ", vx, ", vy: ", vy)
        print("Best action - dx: ", dx, ", dy: ", dy)
        
        ## drive car to new location, including random displacement
        px_old, py_old = px, py
        px, py = vx + px, vy + py
        if random.random() < 0.5:
            if random.random() < 0.5:
                px += 1
            else:
                py += 1

        if py >= Y-5 and px >= X-1:
            finished = True
        
        px = min(max(0, px), X-1)
        py = min(max(0, py), Y-1)
    
        ## move car back on track if needed
        if not finished and track[py][px] == 0:
            if DEBUG:
                print("Move ", px, ", ", py, " back on track")
            if px >= 8:
                while track[py][px] == 0:
                    px -= 1
            else:
                while track[py][px] == 0:
                    px += 1
            if DEBUG:
                print("Moved to ", px, ", ", py)
        
        print_track(px, py) 
    print()
    print("#################################################")
    print()
