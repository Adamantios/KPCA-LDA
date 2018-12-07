from core.kpca import KPCA, Kernel
import numpy as np


def main():
    kpca = KPCA(Kernel.LINEAR, np.array([[1, 1, 3],
                                         [3, 4, 6],
                                         [3, 6, 6]]),
                2)
    kpca.test()


if __name__ == '__main__':
    main()
