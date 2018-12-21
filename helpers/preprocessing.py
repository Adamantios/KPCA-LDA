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
