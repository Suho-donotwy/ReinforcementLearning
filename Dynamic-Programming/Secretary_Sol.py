import numpy as np
import matplotlib.pyplot as plt

# state b : 0, n : 1
b = 0
n = 1
T = 100

def solveSecretaryProblem ():
    V = np.zeros((T+1,2))
    V[T][b] = 1
    V[T][n] = 0
    pi = np.zeros(T+1, dtype='int')

    for tp1 in range(T, 0, -1):
        t = tp1 - 1
        go = 0.0 + 1/tp1*V[t+1][b] + t/tp1*V[t+1][n]
        stop = t / T
        if (go > stop):  # V[t][b] = max(go, stop)
            V[t][b] = go
            pi[t] = 1
        else:
            V[t][b] = stop
            pi[t] = 0
        V[t][n] = max(0, go)


    print("V function\n", V)
    print("Optimal policy\n", pi)
    print ('Watching ratio = ', sum(pi)/T)

def evalPassmPolicy (m):
    V = np.zeros((T+1,2))
    V[T][b] = 1
    V[T][n] = 0

    for tp1 in range(T, 0, -1):
        t = tp1 - 1
        go = 0.0 + 1/tp1*V[t+1][b] + t/tp1*V[t+1][n]
        stop = t / T
        if (t < m):
            V[t][b] = go
        else:
            V[t][b] = stop
        V[t][n] = go
    return V[0][b]

def plotPassmPolicyPerformance():
    Vpi = np.zeros(T)
    for m in range(T):
        Vpi[m] = evalPassmPolicy(m)
    m = np.argmax(Vpi)
    print ("V[", m, "] = ", Vpi[m])
    plt.plot(Vpi)
    plt.show()


if __name__ == "__main__":
    solveSecretaryProblem()
    plotPassmPolicyPerformance()
