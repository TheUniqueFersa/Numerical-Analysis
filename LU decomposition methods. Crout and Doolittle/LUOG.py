import numpy as np

def validar_entrada(A, b):
    # A debe ser cuadrada
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError("A y b deben ser arreglos de NumPy")
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimensiones de A y b no coinciden")

def doolittle_decomposition(A):
    validar_entrada(A, np.zeros(A.shape[0]))
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            suma = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - suma
        for j in range(i+1, n):
            if U[i][i] == 0:
                raise ZeroDivisionError(f"División por cero detectada en U[{i},{i}]")
            suma = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - suma) / U[i][i]
    return L, U
def resolver_LU(L, U, b):
    # validar_entrada(L, b)
    n = len(b)
    d = np.zeros(n)
    for i in range(n):
        d[i] = b[i] - sum(L[i][j] * d[j] for j in range(i))

    x = np.zeros(n)
    for i in reversed(range(n)):
        if U[i][i] == 0:
            raise ZeroDivisionError(f"División por cero detectada en U[{i},{i}]")
        x[i] = (d[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


import numpy as np

def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        for i in range(j, n):
            suma = sum(L[i][k] * U[k][j] for k in range(j))
            L[i][j] = A[i][j] - suma

        for i in range(j, n):
            if L[j][j] == 0:
                raise ZeroDivisionError(f"L[{j},{j}] es cero")
            suma = sum(L[j][k] * U[k][i] for k in range(j))
            U[j][i] = (A[j][i] - suma) / L[j][j]

    return L, U

def resolver_LU_crout(L, U, b):
    n = len(b)
    d = np.zeros(n)
    for i in range(n):
        suma = sum(L[i][j] * d[j] for j in range(i))
        if L[i][i] == 0:
            raise ZeroDivisionError(f"L[{i},{i}] es cero")
        d[i] = (b[i] - suma) / L[i][i]

    # Sustitución regresiva: resolver U·x = y
    x = np.zeros(n)
    for i in reversed(range(n)):
        suma = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (d[i] - suma)

    return x

"""
# --- Prueba con un sistema conocido ---
A = np.array([
    [3, 2, -1],
    [2, -3, 4],
    [1, 1, 1]
], dtype=float)

b = np.array([4, 8, 6], dtype=float)


A = np.array([
    [-1, -4, 9],
    [-1, 4, -4],
    [0, 3, 0]
], dtype=float)

b = np.array([107, -64, -30], dtype=float)
"""

def generar_sistema(dim, li, ls): #dimensión, limite inferior, limite superior
    while True:
        A = np.random.randint(li, ls, size=(dim, dim)).astype(float)  # Convertimos a float directamente
        if np.linalg.det(A) != 0:
            break  # Aseguramos que sea invertible (det ≠ 0)

    x = np.random.randint(li, ls, size=(dim,)).astype(float)  # Vector columna tipo float
    b = A @ x  # Generamos b como vector columna también
    return A, b, x


A, b, x = generar_sistema(3, -10, 10)
#b = b.flatten()
#x= x.flatten()

print(x.shape)

print("Matriz A:")
print(A)
print("\nVector b:")
print(b)
print("\nSolución real x:")
print(x)

# Definir A y b
"""
A = np.array([[3, 2, -1],
              [2, -3, 4],
              [1, 1, 1]], dtype=float)
b = np.array([4, 8, 6], dtype=float)
"""
try:
    # Doolittle
    L_d, U_d = doolittle_decomposition(A)
    x_d = resolver_LU(L_d, U_d, b)
    print("Solución con Doolittle:", x_d)

    # Crout
    L_c, U_c = crout_decomposition(A)
    x_c = resolver_LU_crout(L_c, U_c, b)
    print("Solución con Crout:", x_c)

except ValueError as ve:
    print("Error de validación:", ve)
except ZeroDivisionError as zde:
    print("Error de división por cero:", zde)
except Exception as e:
    print("Error inesperado:", e)
