from unittest import TestCase

import numpy as np

from core import Lda

array = np.array([[16, 12, 1],
                  [93, 43, 0],
                  [3, 74, 0],
                  [13, 46, 0],
                  [57, 62, 0],
                  [3, 8, 1],
                  [3, 8, 3]])

x = array[:, :-1]
y = array[:, -1]


class TestLda(TestCase):
    def test_fit(self):
        lda = Lda()
        sb, sw = lda.fit(x, y)
        print('{}\n{}\n'.format(sb, sw))

    def test_transform(self):
        self.fail()

    def test_fit_transform(self):
        self.fail()

    def test_get_params(self):
        self.fail()
