import numpy as np

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                val = A[i][i] - sum_k
                if val <= 0:
                    raise ValueError("Матрица не является положительно определённой.")
                L[i][j] = np.sqrt(val)
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]

    return L

def solve_cholesky(A, b):
    if not is_positive_definite(A):
        raise ValueError("Матрица не является положительно определённой. Алгоритм остановлен.")

    L = cholesky_decomposition(A)

    # Решаем Ly = b (прямой ход)
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # Решаем Lᵗx = y (обратный ход)
    Lt = L.T
    x = np.zeros_like(b)
    for i in reversed(range(len(b))):
        x[i] = (y[i] - np.dot(Lt[i, i+1:], x[i+1:])) / Lt[i, i]

    return x