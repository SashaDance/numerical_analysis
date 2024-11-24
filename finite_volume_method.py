import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from thomas_algorithm import ThomasAlg
from typing import Callable

class FVMSolver:
    def __init__(self, right_edge: float, time: float,
                 p: Callable[[float], float],
                 lamb: Callable[[float], float],
                 s: Callable[[float, float], float],
                 phi: Callable[[float], float],
                 alpha: Callable[[float], float],
                 beta: Callable[[float], float],
                 m: int):
        self.L = right_edge
        self.T = time
        self.p = p
        self.lamb = lamb
        self.s = s
        self.phi = phi
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.dx = self.L / self.m
        self.x_edges = [self.dx * i for i in range(self.m + 1)]
        self.x = [
            (self.x_edges[i] + self.x_edges[i + 1]) / 2 for i in range(self.m)
        ]
        self.dt = self.dx
        self.t = [
            self.dt * i for i in range(round(self.T / self.L) * self.m + 1)
        ]
        self.current_u = []

    def __get_coefs(self, i: int, t: float) -> tuple[float, float, float, float]:
        if i == 0:
            a = 2 * self.lamb(self.x[0]) / self.dx
        else:
            a = (
                2 * self.lamb(self.x[i - 1]) * self.lamb(self.x[i])
                / (
                    self.dx
                    * (self.lamb(self.x[i - 1]) + self.lamb(self.x[i]))
                )
            )
        if i == self.m - 1:
            c = 2 * self.lamb(self.x[self.m - 1]) / self.dx
        else:
            c = (
                2 * self.lamb(self.x[i]) * self.lamb(self.x[i + 1])
                / (self.dx * (self.lamb(self.x[i]) + self.lamb(self.x[i + 1])))
            )
        a_0 = self.p(self.x[i]) * self.dx / self.dt
        b = a + c + a_0
        s_c = self.s(self.x[i], t)
        u_ = self.current_u[i] if i < (self.m - 1) else self.beta(t)
        d = s_c * self.dx + a_0 * u_

        return a, b, c, d

    def solve(self) -> list[float]:
        # initializing with start condition
        self.current_u = [self.phi(x) for x in self.x]
        # edge conditions
        for cur_t in self.t[1:]:
            a, b, c, d = self.__get_coefs(0, cur_t)
            matrix = [[1, -c / b] + [0] * (self.m - 2)]
            right_sight = [d / b + a * self.alpha(cur_t) / b]
            for i in range(1, self.m - 1):
                a, b, c, d = self.__get_coefs(i, cur_t)
                row = [0] * self.m
                row[i - 1] = -a
                row[i] = b
                row[i + 1] = -c
                right_sight.append(d)
                matrix.append(row)

            a, b, c, d = self.__get_coefs(self.m - 1, cur_t)
            matrix.append([0] * (self.m - 2) + [-a / b, 1])
            right_sight.append(d / b + c * self.beta(cur_t) / b)
            # flattening
            matrix_flat = [elem for row in matrix for elem in row]
            solver_system = ThomasAlg(matrix_flat, right_sight)
            # updating u on current time step
            self.current_u = solver_system.solve()

        return self.current_u


if __name__ == '__main__':
    params = {
        'right_edge': 1,
        'time': 3,
        'p': lambda x: 1,
        'lamb': lambda x: 1 / 2,
        's': lambda x, t: np.exp(x + t) / 2,
        'phi': lambda x: np.exp(x),
        'alpha': lambda t: np.exp(t),
        'beta': lambda t: np.exp(1 + t)
    }
    m_set = [25, 125, 625]
    print_sets = [
        [0 + i for i in range(25)],
        [2 + 5 * i for i in range(25)],
        [12 + 25 * i for i in range(25)]
    ]
    fig, ax = plt.subplots()
    actual_solution = lambda x, t: np.exp(x + t)
    for m, print_set in zip(m_set, print_sets):
        solver = FVMSolver(**params, m=m)
        solution = solver.solve()
        data = pd.DataFrame(columns=['x', 'u numerical', 'u actual', 'residue'])
        x_arr = solver.x
        x_arr = [0] + [x_arr[i] for i in range(len(x_arr)) if i in print_set] + [1]
        data['x'] = x_arr
        u_numerical = solution
        u_numerical = (
            [solver.alpha(3)]
            + [u_numerical[i] for i in range(len(u_numerical)) if i in print_set]
            + [solver.beta(3)]
        )
        data['u numerical'] = u_numerical
        data['u actual'] = [actual_solution(x, 3) for x in data['x']]
        data['residue'] = data['u numerical'] - data['u actual']
        sns.lineplot(x=data['x'], y=data['u numerical'], label=f'Numerical solution m={m}', ax=ax)
        print(f'm={m}')
        print(data)
    sns.lineplot(x=data['x'], y=data['u actual'], label=f'Actual solution', ax=ax)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.show()
print()