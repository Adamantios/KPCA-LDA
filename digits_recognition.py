from core.kpca import KPCA, Kernel


def main():
    kpca = KPCA(Kernel.LINEAR)
    kpca.test()


if __name__ == '__main__':
    main()
