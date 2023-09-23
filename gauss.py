import numpy as np
from matrix import Matrix


def gauss(n: int, matrix: list[list]) -> list:
    solution = [0 for i in range(n)]
    # straight Gauss
    # finding the greatest element in i'th column
    for i in range(n):
        max_elem = matrix[i][i]
        max_row = i
        for j in range(i + 1, n):
            if abs(matrix[j][i]) > abs(max_elem):
                max_elem = matrix[j][i]
                max_row = j

        # exchanging max_row and i'th row
        for j in range(i, n + 1):
            '''
            n + 1 because we need to change
            the right sight of the equation as well
            '''
            h = matrix[i][j]
            matrix[i][j] = matrix[max_row][j]
            matrix[max_row][j] = h

        # dividing i'th row by max_elem
        h = max_elem
        for j in range(i, n + 1):
            matrix[i][j] = matrix[i][j] / h

        '''
        making zeros below the i'th diagonal element,
        which is 1 by that time
        '''
        for k in range(i + 1, n):
            h = matrix[k][i]
            for j in range(i, n + 1):
                matrix[k][j] = matrix[k][j] - matrix[i][j] * h

    # backwards Gauss
    solution[n - 1] = matrix[n - 1][n]
    for i in range(n - 1):
        k = n - i - 2
        cum_sum = 0
        for j in range(k + 1, n):
            cum_sum = cum_sum + solution[j] * matrix[k][j]
        solution[k] = matrix[k][n] - cum_sum

    return solution


def check_huge_system(n: int = 1024) -> None:
    matrix = Matrix.create_random_matrix(n, n)

    column = Matrix(matrix=[[i + 1] for i in range(n)])
    right_sight = matrix * column

    right_sight = [elem[0] for elem in right_sight.matrix]
    matrix.add_column(right_sight)

    print(gauss(1024, matrix.matrix))


# check_huge_system()

matrix = [
    [-1, 3, 2],
    [3, -3, 3],
    [2, 3, -3]
]

print(np.linalg.cond(matrix))
