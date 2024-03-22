import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


class GridMethod:
    def __init__(self, a: float, m: int, n: int, v: float,
                 alpha_t: Callable[[float], float],
                 y_x: Callable[[float], float],
                 f_x_t: Callable[[float, float], float]):
        self.m = m
        self.n = n
        self.v = v
        self.h = 1 / m
        self.tau = (v * self.h) / a
        self.time_b = n * self.tau

        self.x_interval = np.arange(0, 1 + self.h, self.h)
        self.t_interval = np.arange(0, self.time_b + self.tau, self.tau)

        self.alpha_t = alpha_t
        self.y_x = y_x
        self.f_x_t = f_x_t

    def solve(self):
        y_sol = np.zeros((self.n + 1, self.m + 1))

        # borders
        for i in range(self.m + 1):
            x = self.x_interval[i]
            y_sol[0][i] = self.y_x(x)

        for i in range(self.n + 1):
            t = self.t_interval[i]
            y_sol[i][0] = self.alpha_t(t)

        # calculating solution
        for i in range(self.n):
            for j in range(1, self.m + 1):
                t = self.t_interval[i]
                x = self.x_interval[j]
                y_sol[i + 1][j] = (
                        self.v * y_sol[i][j - 1]
                        + (1 - self.v) * y_sol[i][j]
                        + self.tau * self.f_x_t(x, t)
                )

        return y_sol

