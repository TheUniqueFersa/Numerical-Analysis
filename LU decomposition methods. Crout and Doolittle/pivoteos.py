import numpy as np
import numpy as np

def pivoteo_completo(A, b, L, U, P_col, paso):
    """
    Realiza pivoteo completo en la columna y fila correspondiente al paso actual.

    Parámetros:
    - A: matriz de coeficientes (modificada en el lugar)
    - b: vector de constantes (modificado en el lugar)
    - L: matriz L parcialmente construida
    - U: matriz U parcialmente construida
    - P_col: vector que guarda el orden de las columnas
    - paso: índice actual del ciclo principal (columna actual)

    Retorna:
    - A, b, L, U, P_col actualizados con las filas y columnas intercambiadas según el pivote máximo.
    """
    n = A.shape[0]

    # Buscar el índice del valor máximo absoluto en la submatriz A[paso:, paso:]
    submatriz = np.abs(A[paso:, paso:])
    i_max, j_max = np.unravel_index(np.argmax(submatriz), submatriz.shape)
    i_max += paso
    j_max += paso

    # Si el máximo es cero, la submatriz es singular
    if A[i_max, j_max] == 0:
        raise ZeroDivisionError("No se puede realizar pivoteo completo: submatriz singular")

    # Intercambio de filas en A, b, y en L hasta la columna actual
    if i_max != paso:
        A[[paso, i_max], :] = A[[i_max, paso], :]
        b[[paso, i_max]] = b[[i_max, paso]]
        L[[paso, i_max], :paso] = L[[i_max, paso], :paso]

    # Intercambio de columnas en A, U, y actualización del vector de permutación de columnas
    if j_max != paso:
        A[:, [paso, j_max]] = A[:, [j_max, paso]]
        U[:, [paso, j_max]] = U[:, [j_max, paso]]
        P_col[[paso, j_max]] = P_col[[j_max, paso]]

    return A, b, L, U, P_col
