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


if __name__ == '__main__':
    def alpha_t(t: float):
        return 1 - 2 * t


    def y_x(x: float):
        return x + np.exp(x)


    def f_x_t(x: float, t: float):
        return 2 * np.exp(x)


    def actual_solution(x: float, t: float):
        return x - 2 * t + np.exp(x)


    a = 2
    m_arr = [100, 200, 400, 100, 100]
    n_arr = [40, 80, 160, 80, 20]
    v_arr = [0.8, 0.8, 0.8, 0.4, 1.6]
    points = [10, 20, 40, 10, 10]

    fig, ax = plt.subplots(2)

    for i in range(3):
        solver = GridMethod(
            a, m_arr[i], n_arr[i], v_arr[i], alpha_t, y_x, f_x_t
        )

        y_grid_method = solver.solve()

        ax[0].plot(
            solver.x_interval, y_grid_method[-1, :],
            label=f'M={m_arr[i]}, N={n_arr[i]}, mu={v_arr[i]}'
        )

    y_actual = np.zeros((solver.n + 1, solver.m + 1))
    for j in range(solver.n + 1):
        for k in range(solver.m + 1):
            x = solver.x_interval[k]
            t = solver.t_interval[j]
            y_actual[j][k] = actual_solution(x, t)

    ax[0].plot(
        solver.x_interval, y_actual[-1, :],
        label='Actual solution'
    )

    ax[0].legend()
    plt.show()

# TODO: 1, 2, 3 on same graph + 1, 4, 5 on same graph; legends