import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from matrix import Matrix
from gauss import gauss

import pandas as pd


def predict(coefficients: Matrix, x: float) -> float:
    res = 0
    for i in range(coefficients.m):
        res += coefficients[0][i] * x ** i

    return res


def least_squares_pol(x: list[float], y: list[float], m: int,
                      visualize: bool = True) -> Matrix:
    """
    :param x: list of x coordinates of points that you need to interpolate
    :param y: list of y coordinates of points that you need to interpolate
    :param m: the degree of polynomial
    :param visualize: visualize solution or not
    :return: optimal coefficients of the polynomial
    """
    # initializing arrays
    b = [[0 for i in range(m + 1)] for j in range(m + 1)]
    c = [0 for i in range(m + 1)]
    # calculating b coefficients of system that is need to be solved
    for p in range(m + 1):
        for q in range(m + 1):
            b[p][q] = sum([x[i] ** (p + q) for i in range(len(x))])

    # calculating c coefficients of system that is need to be solved
    for p in range(m + 1):
        c[p] = sum([y[i] * x[i] ** p for i in range(len(x))])

    matrix = Matrix(matrix=b)
    right_sight = c
    matrix.add_column(right_sight)

    solution = gauss(m + 1, matrix)

    if visualize:
        plt.scatter(x, y, color='blue', label='Points')
        pol_x = np.linspace(min(x), max(x), 40)
        pol_y = [predict(solution, x_i) for x_i in pol_x]
        plt.plot(pol_x, pol_y, color='red', label='interpolation polynomial')
        plt.grid()
        plt.legend()
        plt.show()

    return solution


def function(x: float) -> float:
    return np.arcsin(2 * x - 1)


def tab(func: Callable = function, l: float = 0, r: float = 1,
        n: int = 10) -> tuple[list[float], list[float]]:
    step = (r - l) / n
    x_arr = list(np.arange(l, r, step))
    y_arr = list(map(func, x_arr))

    return x_arr, y_arr


def print_results(x: list[float], y: list[float]) -> None:
    lin_least_squares = least_squares_pol(x, y, 1, visualize=True)
    quad_least_squares = least_squares_pol(x, y, 2, visualize=True)
    cub_least_squares = least_squares_pol(x, y, 3, visualize=True)

    lin_predictions = [predict(lin_least_squares, x_i) for x_i in x]
    quad_predictions = [predict(quad_least_squares, x_i) for x_i in x]
    cub_predictions = [predict(cub_least_squares, x_i) for x_i in x]

    lin_resid = [y_pred - y_i for y_pred, y_i in zip(lin_predictions, y)]
    quad_resid = [y_pred - y_i for y_pred, y_i in zip(quad_predictions, y)]
    cub_resid = [y_pred - y_i for y_pred, y_i in zip(cub_predictions, y)]

    data = {
        'x values': x,
        'given function values': y,
        'linear function values': lin_predictions,
        'quadratic function values': quad_predictions,
        'cubic function values': cub_predictions,
        'linear residuals': lin_resid,
        'quadratic residuals': quad_resid,
        'cubic residuals': cub_resid
    }

    # create a DataFrame using the data
    df = pd.DataFrame(data)

    # calculate sum of residuals
    sum_resid = df[
        ['linear residuals', 'quadratic residuals', 'cubic residuals']].sum()
    df = df.append(sum_resid, ignore_index=True)
    df = df.fillna('')

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print(df.to_string(index=False))


if __name__ == '__main__':
    data = tab()
    x = data[0]
    y = data[1]

    # print(least_squares_pol(x, y, 3))
    print_results(x, y)
