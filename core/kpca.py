from core.kernels import Kernels, Kernel
import numpy as np


class NotFittedException(Exception):
    pass


class KPCA:
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0,
                 degree: int = 3, sigma: float = None, n_components: int = None):
        self._is_fitted = False
        self.kernel = Kernel(kernel, alpha, coefficient, degree, sigma)
        self.n_components = n_components
        self.alphas = None
        self.lambdas = None

    def _check_n_components(self, n_features: int) -> None:
        if self.n_components is None:
            self.n_components = n_features
        else:
            self.n_components = min(n_features, self.n_components)

    @staticmethod
    def _get_one_ns(shape: int) -> np.ndarray:
        """
        Creates a (shape, shape) symmetric matrix of all 1's divided by shape.

        :param shape: the shape of the one n to be created.
        :return: the one n matrix.
        """
        return np.ones((shape, shape)) / shape

    @staticmethod
    def _center_matrix(kernel_matrix: np.ndarray) -> np.ndarray:
        """
        Centers a matrix.

        :param kernel_matrix: the matrix to be centered.
        :return: the centered matrix.
        """

        # Get the kernel's shape.
        m, n = kernel_matrix.shape

        # Create one n matrices.
        one_m, one_n = KPCA._get_one_ns(m), KPCA._get_one_ns(n)

        # Center the kernel matrix.
        return kernel_matrix - one_m.dot(kernel_matrix) - kernel_matrix.dot(one_n) \
               + np.linalg.multi_dot([one_m, kernel_matrix, one_n])

    @staticmethod
    def _center_symmetric_kernel(kernel_matrix: np.ndarray) -> np.ndarray:
        """
        Centers a matrix. Slightly more efficient than the _center_matrix function.

        :param kernel_matrix: the matrix to be centered.
        :return: the centered matrix.
        """
        # Get the kernel's shape.
        n = kernel_matrix.shape[0]
        # Create one n matrix.
        one_n = KPCA._get_one_ns(n)

        # Center the kernel matrix.
        return kernel_matrix - one_n.dot(kernel_matrix) - kernel_matrix.dot(one_n) \
               + np.linalg.multi_dot([one_n, kernel_matrix, one_n])

    def fit(self, x: np.ndarray):
        # Get the kernel matrix.
        kernel_matrix = self.kernel.calc_array(x)

        # Correct the number of components if needed.
        self._check_n_components(kernel_matrix.shape[0])

        # Center the kernel matrix.
        kernel_matrix = KPCA._center_symmetric_kernel(kernel_matrix)

        # Get the eigenvalues and eigenvectors of the kernel, in ascending order.
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)

        # Get as many alphas and lambdas (eigenvectors and eigenvalues) as the new number of components,
        # in descending order.
        self.alphas = np.delete(np.flip(eigenvectors, axis=1), np.s_[self.n_components:], axis=1)
        self.lambdas = np.delete(np.flip(eigenvalues), np.s_[self.n_components:])

        # Set is fitted flag to true.
        self._is_fitted = True

        return self.alphas, self.lambdas

    def transform(self, x: np.ndarray):
        if not self._is_fitted:
            raise NotFittedException('KPCA has not been fitted yet!')

        # Calculate the kernel matrix.
        kernel_matrix = self.kernel.calc_array(x)

        # Center the kernel matrix.
        kernel_matrix = KPCA._center_matrix(kernel_matrix)

        return kernel_matrix.dot(self.alphas / np.sqrt(self.lambdas))

    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self.transform(x)
