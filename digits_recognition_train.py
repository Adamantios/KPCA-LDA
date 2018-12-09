import time
import numpy
import pandas
import helpers
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from typing import Tuple, Optional

# Create a logger and a plotter.
logger, plotter = helpers.Logger(filename='digits_recognition_results'), helpers.Plotter()


def get_x_y() -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Loads the x and y values for the train and test of the model.

    :return: x_train, y_train, x_test and y_test.
    """
    logger.log('Loading Dataset...')
    x_train, y_train = helpers.datasets.load_digits()
    logger.log(str(len(y_train)) + ' training data loaded')
    x_test, y_test = helpers.datasets.load_digits(train=False)
    logger.log(str(len(y_test)) + ' testing data loaded')

    return x_train, y_train, x_test, y_test


def preprocess(x_train: numpy.ndarray, y_train: numpy.ndarray, x_test: numpy.ndarray) \
        -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, list]:
    """
    Preprocess the data:
        Symmetrize x_train and y_train.

        Scale x_train and x_test using MinMax Scaler.

        Apply PCA keeping 81% of the information.

    :param x_train: the train features.
    :param y_train: the train labels.
    :param x_test: the test features.
    :return: preprocessed x_train, y_train, x_test and pca.components_ used.
    """
    logger.log('Preprocessing...')

    logger.log('\tSymmetrize training dataset...')
    x_train, y_train = helpers.preprocessing.symmetrize_dataset(x_train, y_train)
    logger.log('\t' + str(len(y_train)) + ' training data remained')

    logger.log('\tCutting images...')
    x_train = helpers.preprocessing.cut_images(x_train)
    x_test = helpers.preprocessing.cut_images(x_test)

    logger.log('\tScale data using MinMaxScaler with params:')
    scaler = preprocessing.MinMaxScaler()
    logger.log('\t' + str(scaler.get_params()))
    scaler.fit(x_train.astype(numpy.float))
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    logger.log('\tApplying Principal Component Analysis with params:')
    # Keep 81% of the information.
    pca = PCA(.81, random_state=0, whiten=True)
    logger.log('\t' + str(pca.get_params()))
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    return x_train, y_train, x_test, pca.components_


def fit_predict(x_train: numpy.ndarray, y_train: numpy.ndarray, x_test: numpy.ndarray) -> Tuple[numpy.ndarray, list]:
    """
    Fit a linear SVM and make a prediction based on the given data.

    :param x_train: the train features.
    :param y_train: the train labels.
    :param x_test: the test features.
    :return: The predicted data and the support vectors used.
    """
    logger.log('Creating SVM model with params:')
    model = SVC(C=5.2, gamma=0.01)
    logger.log(model.get_params())

    logger.log('Fitting model...')
    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    end_time = time.perf_counter()
    logger.log('Model has been fit in {:.3} seconds.'.format(end_time - start_time))

    logger.log('Predicting...')
    start_time = time.perf_counter()
    y_predicted = model.predict(x_test)
    end_time = time.perf_counter()
    logger.log('Prediction finished in {:.3} seconds.'.format(end_time - start_time))

    return y_predicted, model.support_vectors_


def show_prediction_info(y_true: numpy.ndarray, y_predicted: numpy.ndarray, support_vectors: list) -> None:
    """
    Logs prediction metrics and number of support vectors used.

    :param y_true: the true labels.
    :param y_predicted: the predicted labels.
    :param support_vectors: the support vectors used.
    """
    logger.log('Model\'s scores:')
    # Calculate accuracy, precision, recall and f1.
    accuracy = metrics.accuracy_score(y_true, y_predicted)
    precision = metrics.precision_score(y_true, y_predicted, average='macro')
    recall = metrics.recall_score(y_true, y_predicted, average='macro')
    f1 = metrics.f1_score(y_true, y_predicted, average='macro')
    logger.log('Accuracy: {:.4}'.format(accuracy))
    logger.log('Precision: {:.4}'.format(precision))
    logger.log('Recall: {:.4}'.format(recall))
    logger.log('F1: {:.4}'.format(f1))

    logger.log('Accuracy for each class: ')
    # Get the confusion matrix.
    cm = metrics.confusion_matrix(y_true, y_predicted)
    # Divide confusion matrix row's values with their sums.
    cm = cm / cm.sum(axis=1)[:, numpy.newaxis]
    # Get the diagonal, which is the accuracy of each class.
    logger.log(cm.diagonal())

    logger.log(str(len(support_vectors)) + ' support vectors used.')


def display_classification_results(x: numpy.ndarray, y_true: numpy.ndarray, y_predicted: numpy.ndarray,
                                   num_of_correct: Optional[int] = 4, num_of_missed: Optional[int] = 4) -> None:
    """
    Plots a number of correctly classified and a number of misclassified digits randomly chosen.

    :param x: the digit features.
    :param y_true: the true labels.
    :param y_predicted: the predicted labels.
    :param num_of_correct: the number of correctly classified digits to be plotted.
        If the number is None, bigger than the array or less than zero, plots all the correctly classified digits.
    :param num_of_missed: the number of misclassified digits to be plotted.
        If the number is None, bigger than the array or less than zero, plots all the misclassified digits.
    """
    logger.log('Plotting some random correctly classified digits.')
    # Get indexes of correctly classified digits.
    digits_indexes = numpy.where(y_true == y_predicted)[0]
    # Plot some random correctly classified digits.
    plotter.plot_classified_digits(x[digits_indexes, :], y_predicted[digits_indexes], y_true[digits_indexes],
                                   num=num_of_correct, filename='correct')

    logger.log('Plotting some random misclassified digits.')
    # Get indexes of misclassified digits.
    digits_indexes = numpy.where(y_true != y_predicted)[0]
    # Plot some random misclassified digits.
    plotter.plot_classified_digits(x[digits_indexes, :], y_predicted[digits_indexes], y_true[digits_indexes],
                                   num=num_of_missed, filename='misclassified')


def main():
    # Get x and y pairs.
    x_train, y_train, x_test, y_test = get_x_y()

    # logger.log('Creating heatmap of the pixels correlation...')
    # plotter.heatmap_correlation(pandas.DataFrame(x_train).corr(), 'Pixels', 'Pixels')

    # Preprocess data.
    x_train_clean, y_train_clean, x_test_clean, pca_components = preprocess(x_train, y_train, x_test)

    # logger.log('Creating heatmap of the preprocessed attributes principal components correlation...')
    # plotter.heatmap_correlation(pandas.DataFrame(pca_components).corr(),
    #                             'Principal components', 'Principal components', filename='heatmap_pca_corr')

    # logger.log('Creating heatmap of the preprocessed attributes principal components and pixels correlation...')
    # plotter.heatmap_correlation(pca_components, 'Pixels', 'Principal components', filename='heatmap_pca')

    # Create model, fit and predict.
    y_predicted, support_vectors = fit_predict(x_train_clean, y_train_clean, x_test_clean)

    # Show prediction information.
    show_prediction_info(y_test, y_predicted, support_vectors)

    # Show some of the classification results.
    display_classification_results(x_test, y_test, y_predicted)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
