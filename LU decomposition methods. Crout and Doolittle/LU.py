import numpy as np

def validar_entrada(A, b):
    # A debe ser instancia de np, cuadrada y b debe ser de la dimensión adecuada
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError("A y b deben ser arreglos de NumPy")
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimensiones de A y b no coinciden")

def descomposicion_doolittle(A):
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
                raise ZeroDivisionError(f"División por cero detectada en la descomposición U[{i},{i}]")
            suma = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - suma) / U[i][i]
    return L, U
def resolver_LU_doolittle(L, U, b):
    # validar_entrada(L, b)
    n = len(b)
    d = np.zeros(n)
    # Sustitución prograsiva
    for i in range(n):
        d[i] = b[i] - sum(L[i][j] * d[j] for j in range(i))

    # Sustitución regresiva
    x = np.zeros(n)
    for i in reversed(range(n)):
        if U[i][i] == 0:
            raise ZeroDivisionError(f"División por cero detectada en la resolución de LU: U[{i},{i}]")
        x[i] = (d[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def descomposicion_crout(A):
    validar_entrada(A, np.zeros(A.shape[0]))
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
    # Sustitución prograsiva
    for i in range(n):
        suma = sum(L[i][j] * d[j] for j in range(i))
        if L[i][i] == 0:
            raise ZeroDivisionError(f"L[{i},{i}] es cero")
        d[i] = (b[i] - suma) / L[i][i]

    # Sustitución regresiva
    x = np.zeros(n)
    for i in reversed(range(n)):
        suma = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (d[i] - suma)

    return x

def generar_sistema(dim, li, ls, prnt=1): #dimensión, limite inferior, limite superior
    while True:
        A = np.random.randint(li, ls, size=(dim, dim)).astype(float)  # Se convierte aa float directamente
        if np.linalg.det(A) != 0:
            break  # Aseguramos que sea invertible (det != 0)

    x = np.random.randint(li, ls, size=(dim,)).astype(float)  # Vector columna tipo float
    b = A @ x  # Se genera b
    if prnt:
      print("Matriz A:")
      print(A)
      print("\nVector b:")
      print(b)
      print("\nSolución real x:")
      print(x)
    return A, b, x

"""
# Definir A y b
A = np.array([[3, 2, -1],
              [2, -3, 4],
              [1, 1, 1]], dtype=float)
b = np.array([4, 8, 6], dtype=float)

#Requiere pivoteo
A = np.array([[7, 0, 9],
              [9, 0, -5],
              [0, -2, 10]], dtype=float)
b = np.array([-28, -36, 10], dtype=float)

# Definir A y b
A = np.array([[-10, 8, 2],
              [4, 2, 0],
              [-2, -8, -6]], dtype=float)
b = np.array([70, 16, -82], dtype=float)
"""

#b = b.flatten()
#x= x.flatten()

#print(x.shape)
#print(np.linalg.det(A))

def main():
    A, b, x = generar_sistema(3, -10, 10)
    try:
        # Doolittle
        L_d, U_d = descomposicion_doolittle(A)
        x_d = resolver_LU_doolittle(L_d, U_d, b)
        print("Solución con Doolittle:", x_d)

        # Crout
        L_c, U_c = descomposicion_crout(A)
        x_c = resolver_LU_crout(L_c, U_c, b)
        print("Solución con Crout:", x_c)

    except ValueError as ve:
        print("Error de validación:", ve)
    except ZeroDivisionError as zde:
        print("Error de división por cero:", zde)
    except Exception as e:
        print("Error inesperado:", e)

# main()