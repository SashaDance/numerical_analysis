from gauss import gauss
import numpy as np


def ldl_decom(n: int, matrix: list[list]) -> list:
    S = np.zeros(n, n)
    D = np.zeros(n, n)

    for i in range(n):
        # calculating element of D matrix
        cum_sum = 0
        for k in range(i):
            cum_sum = S[k][i] * S[k][i]
        for j in range(i, n):

