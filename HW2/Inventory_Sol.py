import numpy as np
T = 10
M = 10
M1 = M + 1
x = 2
c = 10.5
r = 12
w = 0.3  # 0.1500001/.9

def p(d):
    return 1/M

def solveInventoryProblem():
    V = np.zeros((T+1,M1))
    for s in range(M1):
        V[T][s] = x*s - w*s
    pi = np.zeros((T,M1), dtype='int')
    Q = np.zeros(M1)

    for tp1 in range(T,0,-1):
        t = tp1 - 1
        for s in range(M1):
            for a in range(M1-s):
                b = s + a
                Esales = 0
                Q[a] = 0
                for d in range(1,M1):
                    Esales += min(b, d) * p(d)
                    remain = max(0, b-d)
                    Q[a] += V[t+1][remain] * p(d)
                Ereward = r*Esales - c*a - w*s
                Q[a] += Ereward
            a = Q.argmax()   # a = np.argmax(Q)
            pi[t][s] = a
            V[t][s] = Q[a]
            if t == T-1 and s == 0: print(Q)

    print("Optimal value\n", V)
    print("Optimal policy\n", pi)

if __name__ == "__main__":
    solveInventoryProblem()