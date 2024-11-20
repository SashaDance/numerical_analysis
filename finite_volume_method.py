import scipy.integrate as integrate
import scipy
import numpy as np
from tqdm import tqdm
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
        print(self.x, self.t, sep='\n')

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
        s_c = integrate.quad(
            lambda x: self.s(x, t),
            self.x_edges[i], self.x_edges[i + 1]
        )[0] / self.dx
        u_ = self.current_u[i + 1] if i < (self.m - 1) else self.beta(t)
        d = s_c * self.dx + a_0 * u_

        return a, b, c, d

    def solve(self) -> list[float]:
        # initializing with start condition
        self.current_u = [self.phi(x) for x in self.x]
        # edge conditions
        for cur_t in tqdm(self.t[1:]):
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

        print(self.current_u)
        return self.current_u


if __name__ == '__main__':
    solver = FVMSolver(
        right_edge=1, time=3,
        p=lambda x: 1,
        lamb=lambda x: 1 / 6,
        s=lambda x, t: 4 * x * (x ** 2 - 1) * np.exp(t),
        phi=lambda x: 4 * (x ** 3),
        alpha=lambda t: 0,
        beta=lambda t: 4 * np.exp(t),
        m=1000
    )
    # solver = FVMSolver(
    #     right_edge=1, time=3,
    #     p=lambda x: 1,
    #     lamb=lambda x: 1,
    #     s=lambda x, t: np.sin(x) - 2 * np.exp(x - t),
    #     phi=lambda x: np.sin(x) + np.exp(x),
    #     alpha=lambda t: np.exp(-t),
    #     beta=lambda t: np.sin(1) + np.exp(1 - t),
    #     m=125
    # )
    solver.solve()
    # print([np.sin(x) + np.exp(x - 3) for x in solver.x])
    print([4 * (x ** 3) * np.exp(3) for x in solver.x])