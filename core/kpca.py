from core.kernels import Kernels, Kernel
import numpy as np


class KPCA:
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0,
                 degree: int = 3, sigma: float = None, n_components: int = None):
        self.kernel = Kernel(kernel, alpha, coefficient, degree, sigma)
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None

    def _check_n_components(self, n_features: int) -> None:
        if self.n_components is None:
            self.n_components = n_features - 1
        else:
            self.n_components = min(n_features, self.n_components)

    def fit(self, x: np.ndarray):
        self._check_n_components(x.shape[1])
        pass

    def transform(self, x: np.ndarray):
        pass

    def fit_transform(self, x: np.ndarray):
        kernel_matrix = self.kernel.calc_array(x)
        self._check_n_components(kernel_matrix.shape[0])

        # Centering the symmetric NxN kernel matrix.
        N = kernel_matrix.shape[0]
        one_n = np.ones((N, N)) / N
        K = kernel_matrix - one_n.dot(kernel_matrix) - kernel_matrix.dot(one_n) + one_n.dot(kernel_matrix).dot(one_n)

        # Obtaining eigenvalues in descending order with corresponding eigenvectors from the symmetric matrix.
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(K)

        alphas = np.column_stack((self.eigenvectors[:, -i] for i in range(1, self.n_components + 1)))
        lambdas = [self.eigenvalues[-i] for i in range(1, self.n_components + 1)]

        return alphas, lambdas
