from core.kpca import KPCA, Kernel
import numpy as np


def main():
    kpca = KPCA(np.array([[1, 1, 3],
                          [3, 4, 6],
                          [3, 6, 6]]),
                Kernel.LINEAR)
    kpca.test()


if __name__ == '__main__':
    main()
