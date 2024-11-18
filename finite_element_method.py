import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def k(x: float) -> float:
    return np.tanh(x) + 1


def q(x: float) -> float:
    return 2 * np.tanh(x)


def f(x: float) -> float:
    return 2 * (x ** 2 * np.tanh(x) - x / np.cosh(x) ** 2 - 1)


def u(x: float) -> float:
    return x ** 2 + 1


def I1(x: list[float], h: list[float], j: int) -> float:
    return h[j] * k(x[j] - h[j] / 2)


def I2(x: list[float], h: list[float], j: int) -> float:
    return h[j] / 3 * q(x[j] - h[j] / 2) + h[j + 1] / 3 * q(x[j] + h[j] / 2)


def I3(x: list[float], h: list[float], j: int) -> float:
    return h[j] / 2 * f(x[j] - h[j] / 2) + h[j + 1] / 2 * f(x[j] + h[j] / 2)


def I4(x: list[float], h: list[float], j: int) -> float:
    return h[j] / 6 * q(x[j] - h[j] / 2)


def thomas_algorithm(a: list[float], b: list[float], c: list[float], d: list[float], m: int) -> list[float]:
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


def method(n: int) -> list[float]:
    y = np.zeros(n + 1)
    y[0] = 1
    y[n] = 5
    a = 0
    b = 2
    h1 = (b - a) / n

    h = [h1] * (n + 1)
    x = [i * h1 for i in range(n + 1)]
    a = [0] * (n - 1)
    b = [0] * (n - 1)
    c = [0] * (n - 1)
    d = [0] * (n - 1)

    for i in range(1, n):
        a[i - 1] = -1 / (h[i] ** 2) * I1(x, h, i) + I4(x, h, i)
        b[i - 1] = (
            1 / (h[i] ** 2) * I1(x, h, i)
            + 1 / (h[i + 1] ** 2) * I1(x, h, i + 1) + I2(x, h, i)
        )
        c[i - 1] = -1 / (h[i + 1] ** 2) * I1(x, h, i + 1) + I4(x, h, i + 1)
        d[i - 1] = I3(x, h, i)

    d[0] -= a[0] * y[0]
    d[n - 2] -= c[n - 2] * y[n]
    solution = thomas_algorithm(a, b, c, d, n - 1)
    for k in range(n - 1):
        y[k + 1] = solution[k]

    return y


x = [i * 2 / 10 for i in range(11)]
real_solution = [u(x[i]) for i in range(11)]
approx_solutions = []
results = {
    'x': pd.Series(x),
    'actual_solution': pd.Series(real_solution)
}
for n, name in zip([10, 100, 1000], ['u1', 'u2', 'u3']):
    x = [i * 2 / n for i in range(n + 1)]
    y = method(n)
    approx_solutions.append(y)
    approx_solution = [y[n // 10 * i] for i in range(11)]
    residual = [real_solution[i] - approx_solution[i] for i in range(11)]

    results[name] = pd.Series(approx_solution)
    results[f'resid_{name}'] = pd.Series(residual)


df = pd.DataFrame(results)
pd.set_option('display.max_columns', 64)
pd.set_option('display.expand_frame_repr', False)
print(df)


plt.plot(
    [i * 2 / 10 for i in range(11)],
    approx_solutions[0],
    label='n=10'
)
plt.plot(
    [i * 2 / 100 for i in range(101)],
    approx_solutions[1],
    label='n=100'
)
plt.plot(
    [i * 2 / 1000 for i in range(1001)],
    approx_solutions[2],
    label='n=1000'
)
plt.plot(
    results['x'],
    results['actual_solution'],
    label='actual solution'
)

plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.grid()
plt.show()
