import numpy as np

from matrix import Matrix
from thomas_algorithm import thomas_algorithm
import matplotlib.pyplot as plt
from typing import Callable

def spline_interpolation(x: list[float], y: list[float]) -> None:
    # initializing
    end = max(x)
    start = min(x)
    n = len(x) - 1
    h = (end - start) / n
    f = [0 for i in range(n + 1)]

    for i in range(1, n):
        f[i] = (y[i - 1] + y[i + 1] - 2 * y[i]) / h

    # calculating coefficients to use thomas algorithm
    c_diag = [h / 3 for i in range(n)]
    c_diag[-1] = 0

    b_diag = [4 * h / 3 for i in range(n + 1)]
    b_diag[-1] = 1
    b_diag[0] = 1

    a_diag = [h / 3 for i in range(n)]
    a_diag[0] = 0

    kappa_1 = c_diag[0]
    kappa_2 = a_diag[-1]

    # getting c coefficients
    c = thomas_algorithm(kappa_1, kappa_2, a_diag[:-1],
                         b_diag[1:-1], c_diag[1:], f)

    # caclculating a, b and d coefficients
    a = y.copy()
    b = [0 for i in range(n)]
    d = [0 for i in range(n)]
    for i in range(n):
        b[i] = (
            (y[i + 1] - y[i]) / h
            - h * (2 * c[i] + c[i + 1]) / 3
        )
        d[i] = (c[i + 1] - c[i]) / (3 * h)

    k = 100  # size of sample from interpolation
    x_sample = np.linspace(start, end, k)
    y_sample = np.zeros(100, dtype=np.float64)
    for i in range(k):
        if x_sample[i] == end:
            ind = n - 1
        else:
            ind = int(x_sample[i] / h)

        residual = x_sample[i] - ind * h
        y_sample[i] = (a[ind] + b[ind] * residual
                       + c[ind] * residual ** 2 + d[ind] * residual ** 3
        )

    plt.scatter(x, y, color='red')
    plt.plot(x_sample, y_sample)

    plt.show()

def function(x: float) -> float:
    return np.arcsin(2 * x - 1)


def tab(func: Callable = function, l: float = 0, r: float = 1,
        n: int = 10) -> tuple[list[float], list[float]]:
    step = (r - l) / n
    x_arr = list(np.arange(l, r, step))
    y_arr = list(map(func, x_arr))

    return x_arr, y_arr

data = tab()
x = data[0]
y = data[1]

spline_interpolation(x, y)