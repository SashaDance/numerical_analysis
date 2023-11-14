from matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

PI = np.pi

def fourier_transform(x: np.ndarray[np.float64],
                      freq: int = 1000, n: int = 9,
                      use_fft: bool = True) -> np.ndarray[np.complex128]:
    """
    :param x: discrete values of function
    :param n: 2^n is the length of the signal
    :param freq: frequency
    :param use_fft: use Fast Fourier Transform algorithm or not
    :return: list of coefficients of Fourier representation
    """
    N = 2 ** n
    if not use_fft:
        # initializing the X_dft
        X_dft = np.zeros(N, dtype=np.complex128)
        # calculating DFT
        for i in range(N):
            X_dft[i] = np.sum(x * np.exp(-2j * PI * i * np.arange(N) / N))
        return X_dft

    if use_fft:
        # doing the bit reverse order
        permutation = np.zeros(N, dtype=np.int64)
        for i in range(N):
            p = i
            for j in range(1, n + 1):
                permutation[i] += 2 ** (n - j) * (p - 2 * (p // 2))
                p = p // 2
        X_fft = x[permutation]  # reorder input
        X_fft = X_fft.astype(np.complex128)  # changing the type of np.ndarray

        # calculating DFT (using FFT)
        for k in range(1, n + 1):
            for i in range(0, 2 ** (n - k)):
                for l in range(0, 2 ** (k - 1)):
                    index1 = i * 2 ** k + l
                    index2 = index1 + 2 ** (k - 1)
                    x1 = (X_fft[index1] +
                          np.exp(-2j * np.pi * l / (2 ** k)) * X_fft[index2]
                    )
                    x2 = (X_fft[index1] -
                          np.exp(-2j * np.pi * l / (2 ** k)) * X_fft[index2]
                    )
                    X_fft[index1] = x1
                    X_fft[index2] = x2

        return X_fft

def tab(t: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    np.random.seed(42)
    return (
            np.sin(2 * PI * 100 * t) +
            np.sin(2 * PI * 240 * t) +
            np.sin(2 * PI * 60 * t) +
            2 * np.random.randn(t.shape[0])
    )

def spectral_density(X: np.ndarray[np.complex128]) -> np.ndarray[np.float64]:
    N = X.shape[0]
    return np.real(X * np.conj(X)) / N

def plot_results(x: np.ndarray[np.float64], freq: float = 1000,
                 n: int = 9) -> None:
    N = 2 ** n

    X_dft = fourier_transform(x, use_fft=True)
    X_fft = fourier_transform(x, use_fft=True)

    X_fft_scipy = fft(x)

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(
        freq * np.arange(0, N // 2) / N,
        spectral_density(X_dft)[: N // 2]
    )
    axs[0].set_title('Implemented DFT')

    axs[1].plot(
        freq * np.arange(0, N // 2) / N,
        spectral_density(X_fft)[: N // 2]
    )
    axs[1].set_title('Implemented FFT')

    axs[2].plot(
        freq * np.arange(0, N // 2) / N,
        spectral_density(X_fft_scipy)[: N // 2]
    )
    axs[2].set_title('Scipy FFT')

    plt.show()

if __name__ == '__main__':
    n = 9
    N = 2 ** n
    freq = 1000

    t = np.arange(0, N, 1) / 1000
    x = tab(t)

    plot_results(x)
