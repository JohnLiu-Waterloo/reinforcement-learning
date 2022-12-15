# Section 4.3, policy iteration

import math
import numpy as np

def poisson(mean, n):
    return math.pow(mean, n) / math.factorial(n) * math.exp(-mean)

def pprint(V):
    # print in reverse order
    n = len(V)
    for i in range(n):
        print([round(v, 2) for v in V[n-i-1]])
    print()

eps = 1      # for ending policy evaluation
gamma = 0.9
max_cars = 20
max_dist = 14

# initial value function
V = [[0.0 for i in range(max_cars+1)] for j in range(max_cars+1)]
A = [[0 for i in range(max_cars+1)] for j in range(max_cars+1)]

for iter in range(10):
    # policy improvement iterations
    print("Iteration ", iter)
    diff = eps + 1
    while diff > eps:
        diff = 0
        for cars1 in range(max_cars+1):
            for cars2 in range(max_cars+1):
                # begin evaluation for state
                val = 0
                for rent1 in range(max_dist):
                    for rent2 in range(max_dist):
                        if cars1 - A[cars1][cars2] - rent1 < 0 or cars2 + A[cars1][cars2] - rent2 < 0:
                            continue    # bankrupt, no more reward
                        
                        for return1 in range(max_dist):
                            for return2 in range(max_dist):
                                new1 = min(cars1 + return1 - rent1 - A[cars1][cars2], max_cars)
                                new2 = min(cars2 + return2 - rent2 + A[cars1][cars2], max_cars)
                                if new1 < 0 or new2 < 0:
                                    continue        # moved too many cars, infeasible
                                
                                prob = poisson(3, rent1) * poisson(4, rent2) * poisson(3, return1) * poisson(2, return2)
                                val += prob * (10.0*(rent1+rent2) - 2.0*abs(A[cars1][cars2]) + gamma * V[new1][new2])
                
                diff = max(diff, abs(val - V[cars1][cars2]))
                V[cars1][cars2] = val
        print("Delta: ", diff)
        # pprint(V)

    pprint(V)       # converged V

    # policy improvement
    changed = False
    for cars1 in range(max_cars+1):
        for cars2 in range(max_cars+1):
            best_move = 0
            best_val = V[cars1][cars2]

            for nmove in range(-5, 6):
                # make sure we can move this many cars, and don't move more than necessary
                if 0 <= cars1 - nmove <= max_cars and 0 <= cars2 + nmove <= max_cars:
                    val = 0
                    for rent1 in range(max_dist):
                        for rent2 in range(max_dist):
                            if cars1 - nmove - rent1 < 0 or cars2 + nmove - rent2 < 0:
                                continue    # bankrupt, no more reward
                            
                            for return1 in range(max_dist):
                                for return2 in range(max_dist):
                                    new1 = min(cars1 + return1 - rent1 - A[cars1][cars2], max_cars)
                                    new2 = min(cars2 + return2 - rent2 + A[cars1][cars2], max_cars)
                                    if new1 < 0 or new2 < 0:
                                        continue        # moved too many cars, infeasible

                                    prob = poisson(3, rent1) * poisson(4, rent2) * poisson(3, return1) * poisson(2, return2)
                                    val += prob * (10.0*(rent1+rent2) - 2.0*abs(nmove) + gamma * V[new1][new2])

                    # print(cars1, cars2, nmove, val)
                    if best_val < val:
                        best_val = val
                        best_move = nmove
            
            if A[cars1][cars2] != best_move:
                changed = True
            
            A[cars1][cars2] = best_move

    pprint(A)
    if not changed:
        print("Stable policy")
        break
# np.ramdom.poisson()