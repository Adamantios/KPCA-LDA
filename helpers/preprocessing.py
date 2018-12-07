import numpy
from typing import Tuple


def symmetrize_dataset(x: numpy.ndarray, y: numpy.ndarray, keep: int = None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Applies a mask to the original data, in order to keep a certain number of instances for each class.

    If keep value is None, the data will be symmetrical.

    :param keep: an upper bound for the number of instances to keep for each class.
        Default value: None
    :param x: the data
    :param y: the class
    :return: New symmetrized x and y
    """
    # If keep is None, then set it with the minimum of the class instances numbers.
    if keep is None:
        unique_elements, counts = numpy.unique(y, return_counts=True)
        keep = min(counts)

    # Create a mask with False values, with the same shape as y.
    mask = numpy.zeros(y.shape, dtype=numpy.bool)

    # For every class, add a True value in the mask with a maximum number of keep.
    for target in numpy.unique(y):
        mask[numpy.where(y == target)[0][:keep]] = 1

    # Return the data masked.
    return x[mask], y[mask]


def cut_images(images: numpy.ndarray, top: int = 2, bottom: int = 1) -> numpy.ndarray:
    """
    Cuts top rows from the top and bottom rows from the bottom of the passed images.

    :param images: Numpy array containing the images to be cut.
    :param top: the number of the rows to be cut from the top.
    :param bottom: the number of the rows to be cut from the bottom.
    :return: Numpy array containing the cut images.
    """
    top *= 28
    bottom *= 28

    # Return the array with the bottom rows deleted, from the array with the top rows deleted.
    return numpy.delete(numpy.delete(images, slice(0, top), 1), slice(784 - top - bottom, 784 - top), 1)
