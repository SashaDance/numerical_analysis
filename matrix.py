import random


class Matrix:
    def __init__(self, n=None, m=None, matrix=None):
        if matrix is not None:
            self.matrix = matrix
            # checking if matrix shape is (m, 1), i. e. it is a list
            self.check_matrix()
            self.m = len(matrix[0])
            self.n = len(matrix)
        if n is not None and m is not None:
            self.n = n
            self.m = m
            self.matrix = self.fill_zeros(n, m)

    def __getitem__(self, item):
        return self.matrix[item]

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.m == other.n:
                result = Matrix(self.n, other.m)
                for i in range(self.n):
                    for j in range(other.m):
                        for k in range(self.m):
                            result[i][j] += self.matrix[i][k] * other[k][j]
            else:
                raise ValueError(
                    'Unsupported matrices shapes for multiplication')
        else:
            raise TypeError('Unsupported operand type: Matrix expected')

        return result

    def __str__(self):
        return '\n'.join(" ".join(map(str, row)) for row in self.matrix)

    def check_matrix(self) -> None:
        m = len(self.matrix[0])
        for row in self.matrix:
            if len(row) == m:
                continue
            else:
                raise ValueError(
                    'Invalid matrix: Rows have inconsistent lengths')

    @staticmethod
    def fill_zeros(n: int, m: int):

        matrix = [[0 for j in range(m)] for i in range(n)]
        return matrix

    @staticmethod
    def create_random_matrix(n: int, m: int, segment: tuple = (0, 1)):
        matrix = Matrix(n, m)
        for i in range(n):
            for j in range(m):
                matrix[i][j] = random.uniform(segment[0], segment[1])

        return matrix


# matr_1 = Matrix(matrix=[[1, 2, 2], [5, 3, 1]])
# matr_2 = Matrix(matrix=[[1, 2], [1, 1], [1, 1]])
matr_1 = Matrix(matrix=[[1, 2, 3]])
matr_2 = Matrix(matrix=[[1], [1], [1]])
print(matr_1 * matr_2)
