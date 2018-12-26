import numpy as np

# state = best : 0, not best :1
b = 0
n = 1
# action = stop : 0, go : 1
stop = 0
go = 1

def solveSecretaryProblem(T):
    V = np.zeros((T+1,2))
    V[T][b] = 1
    V[T][n] = 0
    pi = np.zeros(T+1, dtype = 'int') # policy

    tem_action = np.zeros(2)

    for itr in range(T-1,-1, -1):
        tem_action[stop] = itr/T
        tem_action[go] = 0 + (1/(itr+1))*V[itr+1][b] + (itr/(itr+1))*V[itr+1][n]
        V[itr][b] = np.max(tem_action)
        pi[itr] = np.argmax(tem_action)

        tem_action[stop] = 0
        V[itr][n] = np.max(tem_action)

    print(V)
    pi = np.where(pi > 0, 'Go', 'Stop')
    print(pi)

solveSecretaryProblem(20)