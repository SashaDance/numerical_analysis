import numpy as np
from matrix import Matrix

def ldl_decom(matrix: Matrix) -> list:
    n = matrix.n
    S = Matrix(n, n)
    D = Matrix(n, n)

    for i in range(n):
        # calculating element D[i][i] and S[i][i]
        cum_sum = 0
        for k in range(i):
            cum_sum += S[k][i] * S[k][i] * D[k][k]
        flag = (matrix[i][i] - cum_sum) >= 0
        D[i][i] = 1 if flag else -1
        S[i][i] = abs(matrix[i][i] - cum_sum) ** (1 / 2)
        for j in range(i + 1, n):
            # calculating element of S matrix
            cum_sum = 0
            for k in range(i):
                cum_sum += S[k][i] * S[k][j] * D[k][k]
            S[i][j] = (matrix[i][j] - cum_sum) / (S[i][i] * D[i][i])

    # finding the solution

    S_tranposed = Matrix.transpose(S)

    y = [0 for i in range(n)]
    y[0] = matrix[0][n] / S_tranposed[0][0]

    for i in range(1, n):
        cum_sum = 0
        for k in range(i):
            cum_sum += Matrix.transpose(S)[i][k] * y[k]
        y[i] = (matrix[i][n] - cum_sum) / S_tranposed[i][i]

    y = [D[i][i] * y[i] for i in range(n)]

    solution = [0 for i in range(n)]
    solution[n - 1] = y[n - 1] / S[n - 1][n - 1]
    for i in range(n - 1):
        k = n - i - 2
        cum_sum = 0
        for j in range(k + 1, n):
            cum_sum = cum_sum + solution[j] * S[k][j]
        solution[k] = (y[k] - cum_sum) / S[k][k]

    return solution


matrix = [
    [-1, 3, 2, 2],
    [3, -3, 3, 9],
    [2, 3, -3, 6]
]

matrix = Matrix(matrix=matrix)
print(ldl_decom(matrix))
