from core.kernels import Kernels, Kernel
import numpy as np


class KPCA:
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0,
                 degree: int = 3, sigma: float = None, n_components: int = None):
        self._kernel = kernel
        self._alpha = alpha
        self._coefficient = coefficient
        self._degree = degree
        self._sigma = sigma
        self._n_components = n_components

    def check_n_components(self, n_features: int):
        if self._n_components is None:
            self._n_components = n_features - 1

    def fit(self, x: np.ndarray):
        self.check_n_components(x.shape[1])
        pass

    def transform(self, x: np.ndarray):
        pass

    def fit_transform(self, x: np.ndarray):
        kernel = Kernel(self._kernel, self._alpha, self._coefficient, self._degree, self._sigma, self._n_components)
        kernel.calc_array(x)

        # Centering the symmetric NxN kernel matrix.
        N = self._kernel.shape[0]
        one_n = np.ones((N, N)) / N
        K = self._kernel - one_n.dot(self._kernel) - self._kernel.dot(one_n) + one_n.dot(self._kernel).dot(one_n)

        # Obtaining eigenvalues in descending order with corresponding eigenvectors from the symmetric matrix.
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        alphas = np.column_stack((eigenvectors[:, -i] for i in range(1, self._n_components + 1)))
        lambdas = [eigenvalues[-i] for i in range(1, self._n_components + 1)]

        return alphas, lambdas

    def test(self):
        print(self._kernel)
