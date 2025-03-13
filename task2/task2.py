import numpy as np

def lu_decomposition(A):
    """
    Выполняет LU-разложение с частичным поворотом.
    Возвращает матрицы P, L и U.
    """
    n = len(A)
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()
    
    for i in range(n):
        # Выбор ведущего элемента (частичный поворот)
        pivot = np.argmax(abs(U[i:, i])) + i
        if pivot != i:
            U[[i, pivot]] = U[[pivot, i]]
            P[[i, pivot]] = P[[pivot, i]]
            if i > 0:
                L[[i, pivot], :i] = L[[pivot, i], :i]
        
        L[i, i] = 1
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j] -= L[j, i] * U[i]
    
    return P, L, U

def forward_substitution(L, b):
    """
    Решает систему Ly = b методом прямой подстановки.
    """
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def backward_substitution(U, y):
    """
    Решает систему Ux = y методом обратной подстановки.
    """
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def inverse_matrix(A):
    """
    Находит обратную матрицу A^(-1) с использованием LU-разложения.
    """
    n = A.shape[0]
    P, L, U = lu_decomposition(A)
    inv_A = np.zeros_like(A)
    
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        y = forward_substitution(L, np.dot(P, e))
        inv_A[:, i] = backward_substitution(U, y)
    
    return inv_A