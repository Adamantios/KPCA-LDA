from helpers import plotter
from unittest import TestCase
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from core import Lda


class TestLda(TestCase):
    _x, _y = datasets.load_iris(return_X_y=True)

    _plotter = plotter.Plotter('tests/plots/lda')

    def test_fit(self):
        lda = Lda()
        w = lda.fit(self._x, self._y)
        print('W:\n{}\n'.format(w))

        w = lda.transform(self._x)
        print('Transformed X:\n{}\n'.format(w))
        self._plotter.scatter(w, self._y)

        print('Scikit:\n')

        scikit_lda = LinearDiscriminantAnalysis(solver='eigen')
        scikit_w = scikit_lda.fit_transform(self._x, self._y)
        print('Transformed X:\n{}\n'.format(scikit_w))
        self._plotter.scatter(scikit_w, self._y)

    def test_transform(self):
        self.fail()

    def test_fit_transform(self):
        self.fail()

    def test_get_params(self):
        self.fail()
