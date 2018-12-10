import random
import numpy as np
from typing import Generator, Tuple
from matplotlib import pyplot as plt
from helpers.utils import create_folder
from helpers.datasets import get_eeg_name


class Plotter:
    def __init__(self, folder='plots'):
        self._folder = folder
        # Create a folder for the plots.
        create_folder(folder)

    def _plot_digit(self, digit: np.ndarray, suptitle: str, title: str, subfolder: str, filename: str,
                    extension: str) -> None:
        """
        Plots and saves an image of a digit.

        :param digit: the digit to be plotted.
        :param suptitle: the super title of the figure.
        :param title: the title of the figure.
        :param subfolder: the subfolder for the image to be saved.
        :param filename: the image's filename.
        :param extension: the extension of the file to be saved.
            Acceptable values: 'png', 'jpg', 'jpeg'.
        """
        # Create a subfolder for the image.
        create_folder(self._folder + '/' + subfolder)

        # Change the shape of the image to 2D.
        digit.shape = (28, 28)

        # Create a subplot.
        fig, ax = plt.subplots(figsize=(3, 3.5))
        # Create a super title.
        fig.suptitle(suptitle + '\n' + title, fontsize='large')
        # Create an image from the digit's pixels. The pixels are not rgb so cmap should be gray.
        ax.imshow(digit, cmap='gray')
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Save and plot the image.
        fig.savefig(self._folder + '/' + subfolder + '/' + filename + '.' + extension)
        plt.show()

    def _plot_eeg(self, eeg, suptitle, title, subfolder, filename, extension):
        """
        Plots an eeg.

        :param eeg: the eeg to be plotted.
        :param suptitle: the plot's supertitle.
        :param title: the plot's title.
        :param subfolder: the subfolder where the plot will be placed.
        :param filename: the plot's filename.
        :param extension: the extension of the filename.
        """
        # Create a subfolder for the plot's image.
        create_folder(self._folder + '/' + subfolder)
        # Create a subplot.
        fig, ax = plt.subplots(figsize=(9, 4))
        # Create a super title.
        fig.suptitle(suptitle + '\n' + title, fontsize='large')
        # Create the plot.
        ax.plot(eeg)

        # Remove xticks, add xlabel and ylabel.
        ax.set_xticks([])
        ax.set_xlabel("1 second", fontsize='large')
        ax.set_ylabel("EEG Value", fontsize='large')

        # Save and plot the image.
        fig.savefig(self._folder + '/' + subfolder + '/' + filename + '.' + extension)
        plt.show()

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

    def plot_classified_digits(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, num: int = None,
                               subfolder: str = 'digits', filename: str = 'digit', extension: str = 'png') -> None:
        """
        Plots and saves a certain number of classified digits.

        :param x: the digit.
        :param y_pred: the predicted value of the classified digit.
        :param y_true: the real value of the classified digit.
        :param num: the number of digits to be plotted.
        :param subfolder: the subfolder for the plots to be saved.
        :param filename: the name of the files.
            Every filename is going to contain the filename, followed by its index.
        :param extension: the extension of the file to be saved.
            Acceptable values: 'png', 'jpg', 'jpeg'.
        """
        # Plot num of digits randomly.
        for digit, i in self._random_picker(x, num):
            self._plot_digit(x[digit], 'Classified as {}'.format(y_pred[digit]),
                             'Correct digit is {}'.format(y_true[digit]), subfolder, filename + str(i), extension)

    def plot_classified_eegs(self, x, y_pred, y_true, num=None, subfolder='eegs', filename='eeg',
                             extension='png'):
        """
        Plots and saves a certain number of classified eegs.

        :param x: the eegs.
        :param y_pred: the predicted values of the classified eeg.
        :param y_true: the real value of the classified eegs.
        :param num: the number of eegs to be plotted.
        :param subfolder: the subfolder for the plots to be saved.
        :param filename: the name of the files.
            Every filename is going to contain the filename, followed by its index.
        :param extension: the extension of the file to be saved.
            Acceptable values: 'png', 'jpg', 'jpeg'.
        """
        # Plot num of eegs randomly.
        for eeg, i in self._random_picker(x, num):
            self._plot_eeg(x[eeg], 'Classified as ' + get_eeg_name(y_pred[eeg]),
                           'Correct condition is ' + get_eeg_name(y_true[eeg]), subfolder, filename + str(i), extension)

    def heatmap_correlation(self, data: np.ndarray, xlabel: str, ylabel: str, subfolder: str = 'heatmaps',
                            filename: str = 'heatmap_correlation', extension: str = 'png') -> None:
        """
        Create and save a heatmap, representing correlation.

        :param data: the correlated data to be plotted.
        :param xlabel: the x ax's label.
        :param ylabel: the y ax's label.
        :param subfolder: the subfolder for the heatmap to be saved.
        :param filename: the heatmap's filename.
        :param extension: the extension of the file to be saved.
            Acceptable values: 'png', 'jpg', 'jpeg'.
        """
        # Create a subfolder for the heatmap.
        create_folder(self._folder + '/' + subfolder)

        # Create a subplot.
        fig, ax = plt.subplots()

        # Create the heatmap.
        img = ax.matshow(data, aspect='auto')
        # Add a colorbar, showing the percentage of the correlation.
        plt.colorbar(img)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Save and plot the heatmap.
        plt.savefig(self._folder + '/' + subfolder + '/' + filename + '.' + extension)
        plt.show()

    def scatter_pcs(self, x: np.ndarray, y: np.ndarray, subfolder: str = 'scatters', filename: str = 'scatter_pcs',
                    extension: str = 'png') -> None:
        """
        Plots and saves a scatterplot with the first one or the first two principal components.

        :param x: the principal components to plot.
        :param y: the class labels.
        :param subfolder: the subfolder for the heatmap to be saved.
        :param filename: the heatmap's filename.
        :param extension: the extension of the file to be saved.
            Acceptable values: 'png', 'jpg', 'jpeg'.
        """
        # If we only have one principal component in a 1D array, i.e. M convert it to a 2D M x 1.
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        # If the principal components are more than two, the plot cannot be represented.
        elif x.shape[1] > 2:
            raise ValueError('Principal components cannot be more than 2.')

        # Create a subfolder for the scatter.
        create_folder(self._folder + '/' + subfolder)

        # Create a figure.
        plt.figure(figsize=(8, 6))

        # Get the class labels and count each label's instances.
        labels, counts = np.unique(y, return_counts=True)

        # For every class, scatter it's principal components.
        for i, count, color in zip(range(labels.size), counts, ['red', 'blue']):
            # If there is one pc, plot 1D.
            if x.shape[1] == 1:
                plt.scatter(x[y == i, 0], np.zeros((count, 1)), color=color, alpha=0.5)
            # Plot 2D.
            else:
                plt.scatter(x[y == i, 0], x[y == i, 1], color=color, alpha=0.5)

        # Set xlabel.
        plt.xlabel('pc1')

        # Set the title and ylabel, if needed.
        if x.shape[1] == 2:
            plt.title('The first two principal components')
            plt.ylabel('pc2')
        else:
            plt.title('The first principal component')

        # Save and plot the scatterplot.
        plt.savefig(self._folder + '/' + subfolder + '/' + filename + '.' + extension)
        plt.show()

    def pca_analysis(self, explained_var_ratio: np.ndarray, decimals: int = 2, subfolder: str = 'scatters',
                     filename: str = 'scatter_pcs', extension: str = 'png') -> None:
        # Create a subfolder for the scatterplot.
        create_folder(self._folder + '/' + subfolder)

        # Convert values to percentage, round them and cumulative sum all the variance ratios, so that we get an array,
        # which contains the total proportion of variance explained with as many components as it's index.
        pov = np.cumsum(np.round(explained_var_ratio, decimals=2 + decimals) * 100)

        # plt.figure(figsize=(6, 6))

        # Set labels, title, y tick limits and style and create the plot.
        plt.ylabel('POV')
        plt.xlabel('Number of Features')
        plt.title('PCA Analysis')
        plt.ylim(0, 100)
        plt.style.context('seaborn-paper')
        plt.plot(pov)

        # Save and plot the scatterplot.
        plt.savefig(self._folder + '/' + subfolder + '/' + filename + '.' + extension)
        plt.show()
