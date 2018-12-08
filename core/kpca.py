from core.kernels import Kernels, Kernel
import numpy as np


class KPCA:
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0,
                 degree: int = 3, sigma: float = None, n_components: int = None):
        self.kernel = Kernel(kernel, alpha, coefficient, degree, sigma)
        self.n_components = n_components
        self.alphas = None
        self.lambdas = None

    def _check_n_components(self, n_features: int) -> None:
        if self.n_components is None:
            self.n_components = n_features
        else:
            self.n_components = min(n_features, self.n_components)

    def fit(self, x: np.ndarray):
        # Get the kernel matrix.
        kernel_matrix = self.kernel.calc_array(x)
        # Get the kernel's dimension. Kernel is symmetric, i.e K x K, so get the K.
        kernel_dimension = kernel_matrix.shape[0]

        # Correct the number of components if needed.
        self._check_n_components(kernel_dimension)

        # Create an array with all its values equal to 1 divided by the K of the kernel.
        one_n = np.ones((kernel_dimension, kernel_dimension)) / kernel_dimension

        # Center the kernel matrix.
        kernel_centered = kernel_matrix - one_n.dot(kernel_matrix) - kernel_matrix.dot(one_n) \
                          + np.linalg.multi_dot([one_n, kernel_matrix, one_n])

        # Get the eigenvalues and eigenvectors of the kernel, in descending order.
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_centered)

        # Get alphas and lambdas.
        self.alphas = np.column_stack((eigenvectors[:, -i] for i in range(1, self.n_components + 1)))
        self.lambdas = [eigenvalues[-i] for i in range(1, self.n_components + 1)]

        return self.alphas, self.lambdas

    def transform(self, x: np.ndarray):
        # pair_dist = np.array([np.sum((x - row) ** 2) for row in X])
        kernel_matrix = self.kernel.calc_array(x)
        return kernel_matrix.dot(self.alphas / self.lambdas)

    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self.transform(x)
