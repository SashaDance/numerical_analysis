from matrix import Matrix
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

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
            for i in range(2 ** (n - k)):
                for l in range(2 ** (k - 1)):
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

    fig, axs = plt.subplots(3, 3)

    # plotting the function, real and imaginary part of DFT
    axs[0][0].plot(
        freq * np.arange(0, N // 2) / N,
        np.real(X_fft_scipy)[: N // 2]
    )
    axs[0][0].set_title('Real part of DFT')

    axs[0][1].plot(
        x[: N]
    )
    axs[0][1].set_title('Given Signal')

    axs[0][2].plot(
        freq * np.arange(0, N // 2) / N,
        np.imag(X_fft_scipy)[: N // 2]
    )
    axs[0][2].set_title('Imaginary part of DFT')

    # plotting DFT in 3 cases: implemented DFT and FFT, function from scipy lib
    axs[1][0].plot(
        freq * np.arange(0, N // 2) / N,
        spectral_density(X_dft)[: N // 2]
    )
    axs[1][0].set_title('Implemented DFT')

    axs[1][1].plot(
        freq * np.arange(0, N // 2) / N,
        spectral_density(X_fft)[: N // 2]
    )
    axs[1][1].set_title('Implemented FFT')

    axs[1][2].plot(
        freq * np.arange(0, N // 2) / N,
        spectral_density(X_fft_scipy)[: N // 2]
    )
    axs[1][2].set_title('Scipy FFT')

    # plotting the inverse FFT in 3 cases

    axs[2][0].plot(
        np.real(ifft(X_dft))
    )
    axs[2][0].set_title('IFFT of implemented DFT')

    axs[2][1].plot(
        np.real(ifft(X_fft))
    )
    axs[2][1].set_title('IFFT of implemented FFT')

    axs[2][2].plot(
        np.real(ifft(X_fft_scipy))
    )
    axs[2][2].set_title('IFFT of scipy FFT')

    plt.show()


if __name__ == '__main__':
    n = 9
    N = 2 ** n
    freq = 1000

    t = np.arange(0, N, 1) / freq
    x = tab(t)

    plot_results(x)
