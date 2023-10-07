from matrix import Matrix
from gauss import gauss


def function(vector: Matrix) -> Matrix:
    x, y = vector[0][0], vector[0][1]

    return  Matrix(matrix=[
        [x * x - y * y + 3 * y],
        [x * x + 3 * x * y + 2 * y * y + 2 * x + 4 * y]
    ])


def jacobian(vector: Matrix) -> Matrix:
    x, y = vector[0][0], vector[0][1]

    return Matrix(matrix=[
        [2 * x, 2 * x + 3 * y + 2],
        [-2 * y + 3, 3 * x + 4 * y + 4]
    ])


def calculate_diff(x_1: Matrix, x_2: Matrix) -> float:
    if x_1.n == x_2.n:
        n = x_1.n
        result = 0
        for i in range(n):
            result += (x_1[i] - x_2[i]) ** 2
        result = result ** 1 / 2
    else:
        raise ValueError('lengths of vectors must be equal')

    return result


def newton_method(x_0: Matrix, tolerance: float = 1e-6) -> list:
    x_prev = x_0
    x_next = Matrix(0, x_0.n)
    iteration = 0
    while True:
        iteration += 1
        F = function(x_prev)
        DF = jacobian(x_prev)
        right_sight = [x[0] * (-1) for x in F()]

        DF.add_column(right_sight)
        z = gauss(2, DF)

        x_next = x_prev + z

        if calculate_diff(x_next, x_prev) < tolerance:
            break

        print(f'Iteration: {iteration}, current solution: {x_next}')
        x_prev = x_next

    print('Solution')
    return x_next

print(newton_method(Matrix(matrix=[[1.5, -0.5]])))