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

def solve_linear_system(A, b):
    """
    Решает систему линейных уравнений Ax = b с использованием LU-разложения с частичным поворотом.
    Возвращает найденное решение x и проверку Ax - b.
    """
    P, L, U = lu_decomposition(A)
    Pb = np.dot(P, b)
    y = forward_substitution(L, Pb)
    x = backward_substitution(U, y)
    check = np.dot(A, x) - b
    return x, check