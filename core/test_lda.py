from unittest import TestCase
import numpy as np
import helpers
from core import Lda


class TestLda(TestCase):
    __array = np.array([[16, 12, 7, 1],
                        [93, 43, 3, 0],
                        [3, 74, 9, 0],
                        [13, 46, 1, 0],
                        [57, 62, 8, 0],
                        [3, 8, 3, 1],
                        [3, 8, 2, 3]])

    _x = __array[:, :-1]
    _y = __array[:, -1]

    _plotter = helpers.plotter.Plotter('tests/plots/lda')

    def test_fit(self):
        lda = Lda()
        w = lda.fit(self._x, self._y)
        print('W:\n{}\n'.format(w))
        self._plotter.scatter_pcs(w, np.array([0, 1, 2]))

        w = lda.transform(self._x)
        print('Transformed X:\n{}\n'.format(w))
        self._plotter.scatter_pcs(w, self._y)

    def test_transform(self):
        self.fail()

    def test_fit_transform(self):
        self.fail()

    def test_get_params(self):
        self.fail()
