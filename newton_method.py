from matrix import Matrix
from gauss import gauss


def function(vector: list) -> list:
    x, y = vector

    return [
        x * x - y * y + 3 * y,
        x * x + 3 * x * y + 2 * y * y + 2 * x + 4 * y
    ]


def jacobian(vector: list) -> list[list]:
    x, y = vector

    return [
        [2 * x, 2 * x + 3 * y + 2],
        [-2 * y + 3, 3 * x + 4 * y + 4]
    ]


def calculate_diff(x_1: list, x_2: list) -> float:
    if len(x_1) == len(x_2):
        n = len(x_1)
        result = 0
        for i in range(n):
            result += (x_1[i] - x_2[i]) ** 2
        result = result ** 1 / 2
    else:
        raise ValueError('lengths of vectors must be equal')

    return result


def newton_method(x_0: list, tolerance: float = 1e-10) -> list:
    x_prev = x_0
    x_next = [0 for i in range(len(x_0))]
    while True:

        F = function(x_prev)
        DF = Matrix(matrix=jacobian(x_prev))

        right_sight = [x * (-1) for x in F]

        DF.add_column(right_sight)
        z = gauss(2, DF)

        x_next = [x_1 + x_2 for x_1, x_2 in zip(x_prev, z)]

        if calculate_diff(x_next, x_prev) < tolerance:
            break

        x_prev = x_next

    return x_next


print(newton_method([1, 0]))
