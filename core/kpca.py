from typing import Union
from core.decomposer import _Decomposer, NotFittedException, InvalidNumberOfComponents
from core.kernels import Kernels, Kernel
import numpy as np


class KPCA(_Decomposer):
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0, degree: int = 3,
                 sigma: float = None, n_components: Union[int, float] = None):
        super().__init__()
        self.kernel = Kernel(kernel, alpha, coefficient, degree, sigma)
        self.n_components = n_components
        self.alphas = None
        self.lambdas = None
        self.explained_var = None
        self._x_fit = None

    def _check_n_components(self, n_features: int) -> None:
        """
        Adds a value to n components if needed.

        If user has not given a value, set it with the number of features minus 1.

        If the number passed is bigger than the number of features, set it with the number of features.

        If proportion of variance has been given,
        calculate the number of features which give the closest pov possible.

        If an invalid number has been passed, raise an exception.

        :param n_features: the number of features.
        """
        # If num of components has not been passed, return n_features - 1.
        if self.n_components is None:
            self.n_components = n_features - 1
        # If n_components passed is bigger than n_features, use n_features.
        elif self.n_components > n_features:
            self.n_components = n_features
        # If n components have been given a correct value, pass
        elif 1 <= self.n_components <= n_features:
            pass
        # If pov has been passed, return as many n_components as needed.
        elif 0 < self.n_components < 1:
            self.n_components = self._pov_to_n_components()
        # Otherwise raise exception.
        else:
            raise InvalidNumberOfComponents('The number of components should be between 1 and {}, '
                                            'or between (0, 1) for the pov, '
                                            'in order to choose the number of components automatically.\n'
                                            'Got {} instead.'
                                            .format(n_features, self.n_components))

        # Keep explained var for n components only.
        self.explained_var = self.explained_var[:self.n_components]

    @staticmethod
    def _one_ns(shape: int) -> np.ndarray:
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
        # If kernel is 1D, which means that we only have 1 test sample,
        # expand its dimension in order to be 2D.
        if kernel_matrix.ndim == 1:
            return np.expand_dims(kernel_matrix, axis=0)

        # Get the kernel's shape.
        m, n = kernel_matrix.shape

        # Create one n matrices.
        one_m, one_n = KPCA._one_ns(m), KPCA._one_ns(n)

        # Center the kernel matrix.
        return kernel_matrix - one_m.dot(kernel_matrix) - kernel_matrix.dot(one_n) \
               + np.linalg.multi_dot([one_m, kernel_matrix, one_n])

    @staticmethod
    def _center_symmetric_matrix(kernel_matrix: np.ndarray) -> np.ndarray:
        """
        Centers a matrix. Slightly more efficient than the _center_matrix function.

        :param kernel_matrix: the matrix to be centered.
        :return: the centered matrix.
        """
        # Get the kernel's shape.
        n = kernel_matrix.shape[0]
        # Create one n matrix.
        one_n = KPCA._one_ns(n)

        # Center the kernel matrix.
        return kernel_matrix - one_n.dot(kernel_matrix) - kernel_matrix.dot(one_n) \
               + np.linalg.multi_dot([one_n, kernel_matrix, one_n])

    def _pov_to_n_components(self) -> int:
        """
        Gets the number of components needed in order to succeed the pov given.

        :return: the number of components.
        """
        # Get the proportion of variance.
        pov = np.cumsum(self.explained_var)

        # Get the index of the nearest pov value with the given pov preference.
        nearest_value_index = (np.abs(pov - self.n_components)).argmin()

        return nearest_value_index + 1

    def fit(self, x: np.ndarray) -> np.ndarray:
        """
        Creates eigenvalues and eigenvectors.

        :param x: the array to be fitted.
        :return: n_components number of eigenvalues and eigenvectors.
        """
        self._x_fit = x

        # Get the kernel matrix.
        kernel_matrix = self.kernel.calc_matrix(self._x_fit)

        # Center the kernel matrix.
        kernel_matrix = KPCA._center_symmetric_matrix(kernel_matrix)

        # Get the eigenvalues and eigenvectors of the kernel, in ascending order.
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)

        # Sort the eigenvalues and eigenvectors in descending order.
        self.alphas = np.flip(eigenvectors, axis=1)
        self.lambdas = np.flip(eigenvalues)

        # Calculate explained var.
        self.explained_var = self.lambdas / np.sum(self.lambdas)

        # Correct the number of components if needed.
        self._check_n_components(self._x_fit.shape[1])

        # Get as many alphas and lambdas (eigenvectors and eigenvalues) as the number of components.
        self.alphas = np.delete(self.alphas, np.s_[self.n_components:], axis=1)
        self.lambdas = np.delete(self.lambdas, np.s_[self.n_components:])

        return kernel_matrix

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Projects the given data to the created feature space.

        :param x: the data to be projected.
        :return: The projected data.
        """
        # If KPCA has not been fitted yet, raise an Exception.
        if self._x_fit is None:
            raise NotFittedException('KPCA has not been fitted yet!')

        # Calculate the kernel matrix.
        kernel_matrix = self.kernel.calc_matrix(self._x_fit, x)

        # Center the kernel matrix.
        kernel_matrix = KPCA._center_matrix(kernel_matrix)

        # Return the projected data.
        return kernel_matrix.T.dot(self.alphas / np.sqrt(np.abs(self.lambdas)))

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Equivalent to fit().transform(), but slightly more efficient.

        :param x: the data to be fitted and then transformed.
        :return: the projected data.
        """
        # Calc kernel and center it.
        kernel_matrix = KPCA._center_matrix(self.fit(x))

        # Return the projected data.
        return kernel_matrix.T.dot(self.alphas / np.sqrt(np.abs(self.lambdas)))

    def get_params(self) -> dict:
        """
        Getter for the kpca's parameters.

        :return: the kpca's parameters.
        """
        params = self.kernel.get_params()
        params['n_components'] = self.n_components if self.n_components is not None else 'auto'

        return params
