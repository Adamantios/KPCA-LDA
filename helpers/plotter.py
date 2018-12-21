import random
import numpy as np
from os.path import join
from typing import Generator, Tuple, Callable
from matplotlib import pyplot as plt
from helpers.utils import create_folder
from helpers.datasets import get_eeg_name
from mpl_toolkits.mplot3d import Axes3D


class TooManyDimensionsError(Exception):
    pass


class Plotter:
    # TODO add getters setters.
    def __init__(self, folder='plots'):
        # Create a folder for the plots.
        self._folder = create_folder(folder)

        self.subfolder: str = ''
        self.suptitle: str = ''
        self.title: str = ''
        self.filename: str = 'plot'
        self.extension: str = 'png'
        self.xlabel: str = ''
        self.ylabel: str = ''
        self.zlabel: str = ''

    def _create_plot_folder(self) -> None:
        """" Create a plot's subfolder. """
        create_folder(self._folder + '/' + self.subfolder)

    def _save_and_show(self, fig: plt.Figure) -> None:
        """ Save and plot a figure. """
        filename = self.filename + '.' + self.extension
        self._save_path = join(self._folder, self.subfolder, filename)

        fig.savefig(self._save_path)
        plt.show()

    def _plot_eeg(self, eeg):
        """
        Plots an eeg.

        :param eeg: the eeg to be plotted.
        """
        self._create_plot_folder()

        # Create a subplot.
        fig, ax = plt.subplots(figsize=(9, 4))
        # Create a super title.
        fig.suptitle('{}\n{}'.format(self.suptitle, self.title), fontsize='large')
        # Create the plot.
        ax.plot(eeg)

        # Remove xticks, add xlabel and ylabel.
        ax.set_xticks([])
        ax.set_xlabel("1 second", fontsize='large')
        ax.set_ylabel("EEG Value", fontsize='large')

        self._save_and_show(fig)

    @staticmethod
    def _random_picker(x: np.ndarray, num: int) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Create a generator which contains a certain number of randomly chosen values from a np array and an index.

        :param x: the np array
        :param num: the number of the random values to be generated.
            If the number is None, bigger than the list or less than zero, randomizes the whole list.
        :return: Generator with a random value and its index.
        """
        # If the number is None, bigger than the list or less than zero, set the number with the lists length.
        if num is None or num > len(x) or num < 0:
            num = len(x)

        # Get num random samples from the list.
        rand_samples = random.sample(range(len(x)), num)

        # For each random sample, yield the sample and its index.
        for sample, i in zip(rand_samples, range(1, len(rand_samples) + 1)):
            yield sample, i

    def plot_classified_eegs(self, x, y_pred, y_true, num=None):
        """
        Plots and saves a certain number of classified eegs.

        :param x: the eegs.
        :param y_pred: the predicted values of the classified eeg.
        :param y_true: the real value of the classified eegs.
        :param num: the number of eegs to be plotted.
        """
        # Plot num of eegs randomly.
        for eeg, i in self._random_picker(x, num):
            self.suptitle = 'Classified as {}'.format(get_eeg_name(y_pred[eeg]))
            self.title = 'Correct condition is {}'.format(get_eeg_name(y_true[eeg]))
            self.filename = '{}{}'.format(self.filename, str(i))
            self._plot_eeg(x[eeg])

    def heatmap_correlation(self, data: np.ndarray) -> None:
        """
        Create and save a heatmap, representing correlation.

        :param data: the correlated data to be plotted.
        """
        self._create_plot_folder()

        # Create a subplot.
        fig, ax = plt.subplots()

        # Create the heatmap.
        img = ax.matshow(data, aspect='auto')

        # Add a colorbar, showing the percentage of the correlation.
        plt.colorbar(img)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        self._save_and_show(fig)

    def scatter(self, x: np.ndarray, y: np.ndarray, class_labels: Callable[[int], str] = None) -> None:
        """
        Plots and saves a scatterplot with the first one, two or three features.

        :param x: the features to plot.
        :param y: the class labels.
        :param class_labels: an optional function which gets the class labels from their indexes.
        """
        if class_labels is None:
            def class_labels(index: int): return index

        # If we only have one principal component in a 1D array, i.e. M convert it to a 2D M x 1.
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        # If the principal components are more than two, the plot cannot be represented.
        elif x.shape[1] > 3:
            raise TooManyDimensionsError('Cannot plot more than 3 dimensions.')

        self._create_plot_folder()

        # Create a figure.
        fig = plt.figure(figsize=(10, 8))

        # Use a style.
        plt.style.use('seaborn-white')

        # Get the class labels and count each label's instances.
        labels, counts = np.unique(y, return_counts=True)

        # If there is one pc, plot 1D.
        if x.shape[1] == 1:
            # Create an ax.
            ax = fig.add_subplot(111)

            # For every class, scatter it's principal components.
            for i, count in zip(labels, counts):
                ax.scatter(x[y == i, 0], np.zeros((count, 1)), alpha=0.5,
                           label='{} class'.format(class_labels(i)))
                ax.legend()

            ax.set_title(self.title)
            # Set xlabel and clear x and y ticks.
            ax.set_xlabel(self.xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

        # If there are 2 pcs plot 2D.
        elif x.shape[1] == 2:
            # Create an ax.
            ax = fig.add_subplot(111)

            # For every class, scatter it's principal components.
            for i, count in zip(labels, counts):
                ax.scatter(x[y == i, 0], x[y == i, 1], alpha=0.5, label='{} class'.format(class_labels(i)))
                ax.legend()

            ax.set_title(self.title)
            # Set x and y labels and clear x and y ticks.
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.set_xticks([])
            ax.set_yticks([])

        # If there are 3 pcs plot 3D.
        else:
            # Create a 3D ax.
            ax = fig.add_subplot(111, projection='3d')
            # For every class, scatter it's principal components.
            for i, count in zip(labels, counts):
                ax.scatter(x[y == i, 0], x[y == i, 1], x[y == i, 2], alpha=0.5,
                           label='{} class'.format(class_labels(i)))
                ax.legend()

            ax.set_title(self.title)
            # Set x, y and z labels, and clear x, y and z ticks.
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.set_zlabel(self.zlabel)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        self._save_and_show(fig)

    def pov_analysis(self, explained_var_ratio: np.ndarray) -> None:
        """
        Create a pov vs k plot.

        :param explained_var_ratio: the explained variance ratio.
        """
        self._create_plot_folder()

        # Convert values to percentage, round them and cumulative sum all the variance ratios, so that we get an array,
        # which contains the total proportion of variance explained with as many components as it's index.
        pov = np.cumsum(np.round(explained_var_ratio, decimals=4) * 100)

        # Create a subplot.
        fig, ax = plt.subplots(figsize=(10, 8))
        # Use a style.
        plt.style.use('seaborn-darkgrid')

        # Set labels, title and y tick limits and create the plot.
        ax.set_ylabel('POV')
        ax.set_xlabel('Number of Features')
        ax.set_title('POV Analysis')
        ax.set_ylim(0, 100)
        ax.plot(range(1, len(pov) + 1, 1), pov)

        # If there are less than 9 features, specify the tick labels.
        if len(pov) < 9:
            ax.set_xticks(range(1, len(pov) + 1, 1))

        self._save_and_show(fig)
