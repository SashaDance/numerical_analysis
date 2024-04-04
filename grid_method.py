import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import pandas as pd
import sys


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
    m_arr = [100, 200, 400]
    n_arr = [40, 80, 160]
    v_arr = [0.8, 0.8, 0.8]
    points = [10, 20, 40]

    fig, ax = plt.subplots(2)

    # overwriting the files
    sys.stdout = open('data/grid_method_1.txt', 'w')
    print()
    sys.stdout = open('data/grid_method_2.txt', 'w')
    print()
    # first plot
    for i in range(len(m_arr)):
        solver = GridMethod(
            a, m_arr[i], n_arr[i], v_arr[i], alpha_t, y_x, f_x_t
        )

        y_grid_method = solver.solve()

        y_actual = np.zeros((solver.n + 1, solver.m + 1))
        x_arr = np.arange(0, 1 + 0.1, 0.1)
        for j in range(solver.n + 1):
            for k in range(solver.m + 1):
                x = solver.x_interval[k]
                t = solver.t_interval[j]
                y_actual[j][k] = actual_solution(x, t)

        y_grid = y_grid_method[-1:, ::solver.m // 10][0]
        y_act = y_actual[-1:, ::solver.m // 10][0]

        # printing results
        df = pd.DataFrame({
            'x': x_arr,
            'y_numerical': y_grid,
            'y_actual': y_act,
            'diff': [a - b for a, b in zip(y_act, y_grid)]
        })
        sys.stdout = open('data/grid_method_1.txt', 'a')
        print(f'M={m_arr[i]}, N={n_arr[i]}, mu={v_arr[i]}')
        print(df, '\n')

        ax[0].plot(
            solver.x_interval, y_grid_method[-1, :],
            label=f'M={m_arr[i]}, N={n_arr[i]}, mu={v_arr[i]}'
        )

    ax[0].plot(
        solver.x_interval, y_actual[-1, :],
        label='Actual solution'
    )

    # second plot
    m_arr = [100, 100, 100]
    n_arr = [40, 80, 20]
    v_arr = [0.8, 0.4, 1.6]

    for i in range(len(m_arr)):
        solver = GridMethod(
            a, m_arr[i], n_arr[i], v_arr[i], alpha_t, y_x, f_x_t
        )

        y_grid_method = solver.solve()

        y_actual = np.zeros((solver.n + 1, solver.m + 1))
        x_arr = np.arange(0, 1 + 0.1, 0.1)
        for j in range(solver.n + 1):
            for k in range(solver.m + 1):
                x = solver.x_interval[k]
                t = solver.t_interval[j]
                y_actual[j][k] = actual_solution(x, t)

        y_grid = y_grid_method[-1:, ::solver.m // 10][0]
        y_act = y_actual[-1:, ::solver.m // 10][0]

        # printing results
        df = pd.DataFrame({
            'x': x_arr,
            'y_numerical': y_grid,
            'y_actual': y_act,
            'diff': [a - b for a, b in zip(y_act, y_grid)]
        })
        sys.stdout = open('data/grid_method_2.txt', 'a')
        print(f'M={m_arr[i]}, N={n_arr[i]}, mu={v_arr[i]}')
        print(df, '\n')

        ax[1].plot(
            solver.x_interval, y_grid_method[-1, :],
            label=f'M={m_arr[i]}, N={n_arr[i]}, mu={v_arr[i]}'
        )

    ax[1].plot(
        solver.x_interval, y_actual[-1, :],
        label='Actual solution'
    )

    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()

    plt.show()

