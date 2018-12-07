import pprint
from unittest import TestCase

import numpy as np

from core import KPCA, Kernel


class TestKPCA(TestCase):
    kpca = KPCA(np.array([[1, 1, 3],
                          [3, 4, 6],
                          [3, 6, 6]]),
                Kernel.LINEAR)

    pprint.pprint(kpca.fit())
