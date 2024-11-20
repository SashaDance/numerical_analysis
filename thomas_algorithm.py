import numpy as np


class ThomasAlg():
    def __init__(self, matrix: list[float], f: list[float]):
        self.matrix = matrix
        self.f = f
        self.n = len(f)
        assert self.n == np.sqrt(len(matrix)), f'Incorrect input'
        self.__get_coefs()
        self.alphas = [-self.c[0]]
        self.betas = [f[0]]

    def __get_coefs(self) -> None:
        self.b = [self.matrix[i * self.n + i] for i in range(self.n)]
        self.c = [self.matrix[i * self.n + i + 1] for i in range(self.n - 1)]
        self.a = [self.matrix[(i + 1) * self.n + i] for i in range(self.n - 1)]

    def solve(self):
        for i in range(1, self.n - 1):
            self.alphas.append(
                -self.c[i] / (self.alphas[i - 1] * self.a[i - 1] + self.b[i])
            )
            self.betas.append(
                (self.f[i] - self.a[i - 1] * self.betas[i - 1])
                / (self.a[i - 1] * self.alphas[i - 1] + self.b[i])
            )

        ans = [0] * self.n
        ans[-1] = (
                (-self.a[-1] * self.betas[-1] + self.f[-1])
                / (1 + self.a[-1] * self.alphas[-1])
        )
        for i in range(self.n - 2, -1, -1):
            ans[i] = self.alphas[i] * ans[i + 1] + self.betas[i]

        return ans

if __name__ == '__main__':
    matrix = [
        2, -1, 0, 0,
        1, 15, -2, 0,
        0, -1, 3, 1,
        0, 0, 1, 1
    ]
    f = [-2, 38, 11, 6]

    inst = ThomasAlg(matrix, f)
    solution = inst.solve()
    print('Solution:')
    for i in range(len(solution)):
        print(f'x_{i} = {solution[i]}')
