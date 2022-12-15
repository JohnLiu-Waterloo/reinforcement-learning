# section 4.4, value iteration
import matplotlib.pyplot as plt

eps = 1e-8
N = 128
p = 0.55

V = [0.0 for i in range(N+1)]
V[N] = 1.0
A = [0 for i in range(N)]

diff = eps + 1
iter = 0
while diff > eps:
    iter += 1

    diff = 0
    stable = True
    for s in range(1, N):
        best_val = 0
        best_move = 1
        for a in range(1, min(s, N-s)+1):
            val = (1-p) * V[s-a] + p * V[s+a]
            
            # only update if the increase in value is numerically significant
            if val > best_val + 1e-16:
                best_move = a
                best_val = val

        diff = max(diff, abs(V[s] - best_val))
        stable = stable if A[s] == best_move else False
        V[s] = best_val
        A[s] = best_move
    
    if iter % 5 == 1:
        print("Iteration ", iter)
        plt.plot(V)
        print("Delta: ", diff, "Stable: ", stable)

plt.figure()
plt.plot(A, 'o')
for i in range(100):
    print(i, " ", A[i], " ", V[i])