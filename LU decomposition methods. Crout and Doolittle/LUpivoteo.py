import numpy as np
from LU import generar_sistema
from pivoteos import pivoteo_completo


def descomposicion_doolittle(A, b):
    # n = len(A)
    # L = np.zeros((n, n))
    # U = np.zeros((n, n))
    # P = np.eye(n)
    A = A.copy()
    b = b.copy()
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    P_col = np.arange(n)

    for i in range(n):
        A, b, L, U, P_col = pivoteo_completo(A, b, L, U, P_col, i)

        L[i][i] = 1
        for j in range(i, n):
            suma = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - suma
        for j in range(i + 1, n):
            if U[i][i] == 0:
                print(A, b)
                print(L)
                print(U)
                raise ZeroDivisionError(f"División por cero detectada en la descomposición Doolittle U[{i},{i}]")
            suma = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - suma) / U[i][i]

    return L, U, b, P_col

def descomposicion_crout(A, b):
    # n = len(A)
    # L = np.zeros((n, n))
    # U = np.zeros((n, n))
    A = A.copy()
    b = b.copy()
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    P_col = np.arange(n)
    # P = np.eye(n)

    for j in range(n):
        A, b, L, U, P_col = pivoteo_completo(A, b, L, U, P_col, j)

        for i in range(j, n):
            suma = sum(L[i][k] * U[k][j] for k in range(j))
            L[i][j] = A[i][j] - suma

        U[j][j] = 1

        for i in range(j + 1, n):
            if L[j][j] == 0:
                print(A, b, L)
                raise ZeroDivisionError(f"L[{j},{j}] es cero en Crout")
                
            suma = sum(L[j][k] * U[k][i] for k in range(j))
            U[j][i] = (A[j][i] - suma) / L[j][j]

    return L, U, b, P_col

def resolver_LU_doolittle(L, U, b, P_col):
    n = len(b)
    d = np.zeros(n)
    for i in range(n):
        d[i] = b[i] - sum(L[i][j] * d[j] for j in range(i))

    x = np.zeros(n)
    for i in reversed(range(n)):
        if U[i][i] == 0:
            raise ZeroDivisionError(f"División por cero detectada en la resolución de LU: U[{i},{i}]")
        x[i] = (d[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    x_final = np.zeros_like(x)
    for i in range(n):
        x_final[P_col[i]] = x[i]

    return x_final

def resolver_LU_crout(L, U, b, P_col):
    n = len(b)
    d = np.zeros(n)
    for i in range(n):
        suma = sum(L[i][j] * d[j] for j in range(i))
        if L[i][i] == 0:
            raise ZeroDivisionError(f"L[{i},{i}] es cero")
        d[i] = (b[i] - suma) / L[i][i]

    x = np.zeros(n)
    for i in reversed(range(n)):
        suma = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (d[i] - suma)


    x_final = np.zeros_like(x)
    for i in range(n):
        x_final[P_col[i]] = x[i]

    return x_final

# -------- EJEMPLO DE USO --------
'''
A = np.array([[-2, 1, 6],
              [-7, 0, 8],
              [7, 1, -5]], dtype=float)
b = np.array([-10, -24, 23], dtype=float)
# Resuelta con pivoteo completo

# Falla por cero en diagonal
A = np.array([[0, 8, -3],
              [5, 3, 7],
              [7, 0, -4]], dtype=float)
b = np.array([-90, -15, -66], dtype=float)
# Resuelta con pivoteo completo

A = np.array([[-2, -6, 1],
              [6, -4, 9],
              [4, 3, 0]], dtype=float)
b = np.array([34, -110, -48], dtype=float)
# Resuelta con pivoteo completo

A = np.array([[-4, -5, 0],
              [-8, -10, -10],
              [2, 2, -8]], dtype=float)
b = np.array([65, 40, -102], dtype=float)
# Resuelta con pivoteo completo

--- 

# No se puede realizar pivoteo completo porque hay submatriz singular
A = np.array([[4, 7, -4],
              [1, -2, 0],
              [4, -10, -2]], dtype=float)
b = np.array([-37, -3, 4], dtype=float)

# Solución real tiene valores muy pequeños, submatriz singular
A = np.array([[7, 0, 9],
              [9, 0, -5],
              [0, -2, -10]], dtype=float)
b = np.array([-28, -36, 10], dtype=float)

# No se puede resolver por pc: submatriz singular
A = np.array([[6, 5, 0],
              [0, 6, 1],
              [-9, -2, -6]], dtype=float)
b = np.array([-22, -47, -17], dtype=float)
'''
# print(A.shape)
A = np.array([[2, -1, 4, 1, -1],
              [-1, 3, -2, -1, 2],
              [5, 1, 3, -4, 1],
              [3, -2, -2, -2, 3],
              [-4, -1, -5, 3, -4]], dtype=float)
b = np.array([7, 1, 33, 24, -49], dtype=float)

x_r = np.linalg.solve(A, b)
print(f"Solición Real: {x_r}")
# A, b, x = generar_sistema(3, -100, 100)

# print(np.linalg.det(A))
# condicion = np.linalg.cond(A)
# print("Número de condición:", condicion)


try:
    # Doolittle
    L_d, U_d, b_d, P_col_d = descomposicion_doolittle(A, b)
    x_d = resolver_LU_doolittle(L_d, U_d, b_d, P_col_d)
    print("Solución con Doolittle:", x_d)

    # Crout
    L_c, U_c, b_c, P_col_c = descomposicion_crout(A, b)
    x_c = resolver_LU_crout(L_c, U_c, b_c, P_col_c)
    print("Solución con Crout:", x_c)

except ValueError as ve:
    print("Error de validación:", ve)
except ZeroDivisionError as zde:
    print("Error de división por cero:", zde)
except Exception as e:
    print("Error inesperado:", e)
