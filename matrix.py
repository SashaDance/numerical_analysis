
class Matrix:
    def __init__(self, n=None, m=None, matrix=None):
        if matrix is not None:
            self.matrix = matrix
            self.n = len(matrix)
            self.m = len(matrix[0])
