import numpy as np
T = 10
M = 10
M1 = M + 1
x = 2
c = 10.5
r = 12
w = 0.3

def p(d):
    return 1/M

def solveInventoryProblem():
    V = np.zeros((T+1,M1))
    for s in range(M1):
        V[T][s] = x * s - w * s
    pi = np.zeros((T,M1), dtype = 'int')
    Q = np.zeros(M1)

    for itr in range(T-1,-1,-1):
        for inventory in range(0,M1): # inventory = state
            for action in range(0,M1-inventory): # Action which makes inventory + action > 10 is non-meaningful.
                Expectation_of_Reward = r * sum([min(inventory+action, d) * p(d) for d in range(1,M1)]) - c * action - w * inventory
                Expectation_of_V = sum(p(d) * V[itr+1][max(0,inventory+action-d)] for d in range(1,M1))
                Q[action] = Expectation_of_Reward + Expectation_of_V

            V[itr][inventory] = np.max(Q)
            pi[itr][inventory] = np.argmax(Q)

    print(pi)

solveInventoryProblem()
