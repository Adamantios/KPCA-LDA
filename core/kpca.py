from core.kernels import Kernels, Kernel
import numpy as np


class NotFittedException(Exception):
    pass


class KPCA:
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0,
                 degree: int = 3, sigma: float = None, n_components: int = None):
        self._x_fit = None
        self.kernel = Kernel(kernel, alpha, coefficient, degree, sigma)
        self._kernel_matrix = None
        self.n_components = n_components
        self.alphas = None
        self.lambdas = None

    def _check_n_components(self, n_features: int) -> None:
        if self.n_components is None:
            self.n_components = n_features
        else:
            self.n_components = min(n_features, self.n_components)

    def _is_fitted(self):
        return False if self._x_fit is None else True

    def fit(self, x: np.ndarray):
        self._x_fit = x

        # Get the kernel matrix.
        self._kernel_matrix = self.kernel.calc_array(x)
        # Get the kernel's dimension. Kernel is symmetric, i.e K x K, so get the K.
        kernel_dimension = self._kernel_matrix.shape[0]

        # Correct the number of components if needed.
        self._check_n_components(kernel_dimension)

        # Create an array with all its values equal to 1 divided by the K of the kernel.
        one_n = np.ones((kernel_dimension, kernel_dimension)) / kernel_dimension

        # Center the kernel matrix.
        kernel_centered = self._kernel_matrix - one_n.dot(self._kernel_matrix) - self._kernel_matrix.dot(one_n) \
                          + np.linalg.multi_dot([one_n, self._kernel_matrix, one_n])

        # Get the eigenvalues and eigenvectors of the kernel, in ascending order.
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_centered)

        # Get as many alphas and lambdas (eigenvectors and eigenvalues) as the new number of components,
        # in descending order.
        self.alphas = np.delete(np.flip(eigenvectors, axis=1), np.s_[self.n_components:], axis=1)
        self.lambdas = np.delete(np.flip(eigenvalues), np.s_[self.n_components:])

        return self.alphas, self.lambdas

    def transform(self, x: np.ndarray):
        if not self._is_fitted():
            raise NotFittedException('KPCA has not been fitted yet!')

        kernel_matrix = self.kernel.calc_array(self._x_fit, x)
        return kernel_matrix.dot(self.alphas / self.lambdas)

    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self._kernel_matrix.dot(self.alphas / self.lambdas)
