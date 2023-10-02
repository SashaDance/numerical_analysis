from matrix import Matrix
from copy import deepcopy
import math


def find_max_nondiag_elem(matrix: Matrix) -> tuple[float, int, int]:
    max_elem = -1
    n = matrix.n
    p = q = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if abs(matrix[i][j]) > max_elem:
                max_elem = abs(matrix[i][j])
                p = i
                q = j
    return max_elem, p, q


def jacobi(matrix: Matrix, tolerance: float = 1e-10) -> (list, Matrix):
    matrix = deepcopy(matrix)

    n = matrix.n
    max_elem, p, q = find_max_nondiag_elem(matrix)

    eigenvectors = Matrix.identity_matrix(n)

    # rotation loop
    while max_elem > tolerance:
        if matrix[p][p] != matrix[q][q]:
            phi = math.atan(2 * matrix[p][q] / (matrix[p][p] - matrix[q][q])) / 2
        else:
            phi = math.pi / 4

        # building rotation matrix
        U = Matrix.identity_matrix(n)
        U[p][p] = math.cos(phi)
        U[p][q] = -1 * math.sin(phi)
        U[q][p] = math.sin(phi)
        U[q][q] = math.cos(phi)

        # making rotation
        matrix = Matrix.transpose(U) * matrix * U

        max_elem, p, q = find_max_nondiag_elem(matrix)

        eigenvectors = eigenvectors * U

    eigenvalues = [matrix[i][i] for i in range(n)]

    return eigenvalues, eigenvectors

def cond(matrix: Matrix) -> float:
    eigenvalues = jacobi(matrix)[0]

    max_eigenvalue = -1
    min_eigenvalue = 10e10

    for eigenvalue in eigenvalues:
        max_eigenvalue = max(max_eigenvalue, abs(eigenvalue))
        min_eigenvalue = min(min_eigenvalue, abs(eigenvalue))

    cond = max_eigenvalue / min_eigenvalue

    return cond

matrix = [
    [-1, 3, 2],
    [3, -3, 3],
    [2, 3, -3]
]

m = Matrix(matrix=matrix)

print(jacobi(m)[0])
print(jacobi(m)[1])
print(cond(m))
