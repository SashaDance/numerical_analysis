import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def thomas_algorithm(a: list[float], b: list[float], c: list[float],
                     d: list[float], m: int) -> list[float]:
    alphas = [0] * m
    betas = [0] * m
    solution = [0] * m

    alphas[0] = -c[0] / b[0]
    betas[0] = d[0] / b[0]
    for i in range(1, m):
        r = alphas[i - 1] * a[i] + b[i]
        alphas[i] = -c[i] / r
        betas[i] = (d[i] - betas[i - 1] * a[i]) / r

    solution[m - 1] = betas[m - 1]
    for i in range(m - 2, -1, -1):
        solution[i] = alphas[i] * solution[i + 1] + betas[i]

    return solution


def phi(x: float) -> float:
    return np.exp(x)


def alpha(t: float) -> float:
    return np.exp(t)


def beta(t: float) -> float:
    return np.exp(t + 1)


def f(x: float, t: float) -> float:
    return np.exp(x + t) / 2


def u(x: float, t: float) -> float:
    return np.exp(x + t)


def method(N: int, M: int) -> np.ndarray:
    m = M - 1
    h = 1 / M
    tau = 3 / N
    k = 1 / 2
    r = k * tau / (h ** 2)
    y = np.zeros((M + 1, N + 1))
    t1 = [i * tau for i in range(N + 1)]
    t2 = [round((i+0.5)*tau,4) for i in range(N+1)]

    x = [i * h for i in range(M + 1)]

    # Initial conditions
    for i in range(M + 1):
        y[i, 0] = phi(x[i])

    for i in range(1, N + 1):
        y[0, i] = alpha(t1[i])
        y[M, i] = beta(t1[i])

    a = [-r / 2] * m
    b = [1 + r] * m
    c = [-r / 2] * m
    d = [0] * m

    for i in range(N):
        for j in range(m):
            q = j + 1
            d[j] = (r / 2) * y[q - 1, i] + (1 - r) * y[q, i] + (r / 2) * y[
                q + 1, i] + tau * f(x[q], t2[i])

        d[0] -= a[0] * y[0, i + 1]
        d[m - 1] -= c[m - 1] * y[M, i + 1]
        solution = thomas_algorithm(a, b, c, d, m)
        for k in range(m):
            y[k + 1, i + 1] = solution[k]

    return y


def main():
    N = 1000
    M_values = [10, 100, 1000]
    names = ['u1', 'u2', 'u3']
    x = [round(i * 1 / 10, 4) for i in range(11)]
    real_solution = [u(xi, 3) for xi in x]

    resulting_table = {
        'x': pd.Series(x),
        'actual_solution': pd.Series(real_solution)
    }

    approx_solutions = []

    for idx, M in enumerate(M_values):
        approx_solution = method(N, M)
        approx_solutions.append([approx_solution[i, N] for i in range(M)])
        approx_solution_11 = [approx_solution[(M // 10) * j, N] for j in
                              range(11)]
        residuals = [real_solution[k] - approx_solution_11[k] for k in
                     range(11)]

        resulting_table[names[idx]] = pd.Series(approx_solution_11)
        resulting_table[f'resid_{names[idx]}'] = pd.Series(residuals)

    df = pd.DataFrame(resulting_table)
    pd.set_option('display.max_columns', 64)
    pd.set_option('display.expand_frame_repr', False)
    print(df)

    plt.plot(x, real_solution, label='actual solution')
    for i, M in enumerate(M_values):
        plt.plot([round(i * 1 / M, 4) for i in range(M)], approx_solutions[i],
                 label=f'M={M}')
    plt.legend()
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()


if __name__ == "__main__":
    main()
