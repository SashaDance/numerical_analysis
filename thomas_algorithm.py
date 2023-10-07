
def thomas_algorithm(kappa_1: int, kappa_2: int, a: list,
                     b: list, c: list, f: list) -> list:

    mu_1 = f[0]
    alphas = [kappa_1]
    betas = [mu_1]
    n = len(f)
    solution = [0 for i in range(n)]

    # finding alphas and betas
    for i in range(1, n - 1):
        alpha = -c[i - 1] / (a[i - 1] * alphas[i - 1] + b[i - 1])
        beta = (
            (f[i] - a[i - 1] * betas[i - 1]) /
            (a[i - 1] * alphas[i - 1] + b[i - 1])
        )
        alphas.append(alpha)
        betas.append(beta)

    mu_2 = f[n - 1]

    # backwards
    solution[n - 1] = (
            (kappa_2 * betas[n - 2] + mu_2) /
            (1 - kappa_2 * alphas[n - 2])
    )

    for i in range(n - 2, -1, -1):
        solution[i] = alphas[i] * solution[i + 1] + betas[i]

    return solution


a = [-5, 4]
b = [-7, 4]
c = [2, -9]
f = [37, 17, 71, 51]

kappa_1 = -9
kappa_2 = -9

print('Solution')
print(thomas_algorithm(kappa_1, kappa_2, a, b, c, f))
