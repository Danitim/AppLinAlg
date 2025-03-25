import numpy as np

def ldm_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    M = np.eye(n)
    D = np.zeros((n, n))

    for i in range(n):
        # Вычисляем D[i][i]
        sum_LDM = sum(L[i][k] * D[k][k] * M[i][k] for k in range(i))
        D[i][i] = A[i][i] - sum_LDM

        # Строим строки L
        for j in range(i+1, n):
            sum_LDM = sum(L[j][k] * D[k][k] * M[i][k] for k in range(i))
            L[j][i] = (A[j][i] - sum_LDM) / D[i][i]

        # Строим строки M
        for j in range(i+1, n):
            sum_LDM = sum(L[i][k] * D[k][k] * M[j][k] for k in range(i))
            M[j][i] = (A[i][j] - sum_LDM) / D[i][i]

    return L, D, M