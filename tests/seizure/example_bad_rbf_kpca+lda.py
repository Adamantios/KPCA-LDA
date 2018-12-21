import time
import numpy
import helpers
from core import KPCA, Kernels, Lda
from sklearn import metrics, neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Create a logger and a plotter.
logger, plotter = helpers.Logger(folder='logs/seizure-tests', filename='example_bad'), helpers.Plotter(
    folder='plots/seizure-tests')


def get_x_y():
    logger.log('Loading Dataset...')
    x, y = helpers.datasets.load_seizure()
    logger.log(str(len(y)) + ' data loaded')

    logger.log('Splitting to 70% train and 30% test data...')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    logger.log(str(len(y_train)) + ' training data available.')
    logger.log(str(len(y_test)) + ' testing data available.')

    return x_train, y_train, x_test, y_test


def preprocess(x_train, y_train, x_test):
    logger.log('Preprocessing...')

    logger.log('\tScaling data, using Min Max Scaler with params:')
    scaler = preprocessing.MinMaxScaler((-1, 1))
    logger.log('\t' + str(scaler.get_params()))
    x_train = scaler.fit_transform(x_train.astype(float))
    x_test = scaler.transform(x_test.astype(float))

    logger.log('\tApplying Principal Component Analysis with params:')
    pca = KPCA(Kernels.RBF, sigma=1, n_components=178)
    logger.log('\t' + str(pca.get_params()))
    pca.fit_transform(x_train)

    # Plot pca pov vs k.
    plotter.subfolder = 'pca_analysis/all_components'
    plotter.filename = 'rbf-bad'
    plotter.xlabel = 'Number of Principal Components'
    plotter.title = 'PCA Analysis\nParameters: {}'.format(pca.get_params())
    plotter.pov_analysis(pca.explained_var)

    logger.log('\tApplying Principal Component Analysis with params:')
    kpca_pov = 0.9
    pca = KPCA(Kernels.RBF, sigma=1, n_components=kpca_pov)
    logger.log('\t' + str(pca.get_params()))
    x_train = pca.fit_transform(x_train)

    # Plot pca pov vs k.
    plotter.subfolder = 'pca_analysis/pov_{}'.format(kpca_pov)
    plotter.title = 'PCA Analysis\nParameters: {}'.format(pca.get_params())
    plotter.pov_analysis(pca.explained_var)

    plotter.subfolder = 'scatters/pca/rbf-bad'
    plotter.filename = 'pov_{}_3pcs'.format(kpca_pov)
    plotter.scatter(pca.alphas[:, :3], y_train, class_labels=helpers.datasets.get_eeg_name)

    plotter.filename = 'pov_{}_2pcs'.format(kpca_pov)
    plotter.scatter(pca.alphas[:, :2], y_train, class_labels=helpers.datasets.get_eeg_name)

    plotter.filename = 'pov_{}_1pcs'.format(kpca_pov)
    plotter.scatter(pca.alphas[:, 0], y_train, class_labels=helpers.datasets.get_eeg_name)

    x_test = pca.transform(x_test)

    logger.log('\tApplying Linear Discriminant Analysis with params:')
    lda = Lda()
    logger.log('\t' + str(lda.get_params()))
    x_train = lda.fit_transform(x_train, y_train)
    x_test = lda.transform(x_test)

    # Plot lda pov vs k.
    plotter.subfolder = 'lda_analysis/pov_{}'.format(kpca_pov)
    plotter.filename = 'rbf-bad'
    plotter.xlabel = 'Number of LDA Features'
    plotter.title = 'LDA POV vs K\nParameters: {}'.format(pca.get_params())
    plotter.pov_analysis(lda.explained_var)

    plotter.subfolder = 'scatters/lda/rbf-bad'
    plotter.filename = 'pov_{}_3lds'.format(kpca_pov)
    plotter.scatter(x_train[:, :3], y_train, class_labels=helpers.datasets.get_eeg_name)

    plotter.filename = 'pov_{}_2lds'.format(kpca_pov)
    plotter.scatter(x_train[:, :2], y_train, class_labels=helpers.datasets.get_eeg_name)

    plotter.filename = 'pov_{}_1ld'.format(kpca_pov)
    plotter.scatter(x_train[:, 0], y_train, class_labels=helpers.datasets.get_eeg_name)

    return x_train, y_train, x_test


def fit_predict(x_train, y_train, x_test):
    logger.log('Creating KNN model with params:')
    model = neighbors.KNeighborsClassifier()
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

    return y_predicted


def show_prediction_info(y_test, y_predicted, save: bool = True, folder: str = 'results/seizure-tests',
                         filename: str = 'example-bad', extension: str = 'xlsx',
                         sheet_name: str = 'results'):
    # Get the accuracy of each class.
    accuracies = helpers.utils.cm_to_accuracies(metrics.confusion_matrix(y_test, y_predicted))

    # Create results dictionary.
    results = {'Epileptic Seizure Accuracy': [accuracies[4]],
               'Tumor Located Accuracy': [accuracies[3]],
               'Healthy Area Accuracy': [accuracies[2]],
               'Eyes Closed Accuracy': [accuracies[1]],
               'Eyes Opened Accuracy': [accuracies[0]],
               'Accuracy': [metrics.accuracy_score(y_test, y_predicted)],
               'Precision': [metrics.precision_score(y_test, y_predicted, average='macro')],
               'Recall': [metrics.recall_score(y_test, y_predicted, average='macro')],
               'F1': [metrics.f1_score(y_test, y_predicted, average='macro')]}

    # Log results.
    logger.log('Model\'s Results:')
    for key, values in results.items():
        for value in values:
            logger.log('{text}: {number:.{points}g}'.format(text=key, number=value, points=4))

    # Create excel if save is True.
    if save:
        helpers.utils.create_excel(results, folder, filename, extension, sheet_name)


def display_classification_results(x_test, y_test, y_predicted):
    logger.log('Plotting some random correctly classified EEGs.')
    # Get indexes of misclassified digits.
    eegs_indexes = numpy.where(y_test == y_predicted)[0]
    # Plot some random misclassified digits.
    plotter.filename = 'correct'
    plotter.subfolder = 'eegs'
    plotter.plot_classified_eegs(x_test[eegs_indexes, :], y_predicted[eegs_indexes], y_test[eegs_indexes], num=4)

    logger.log('Plotting some random misclassified EEGs.')
    # Get indexes of misclassified digits.
    eegs_indexes = numpy.where(y_test != y_predicted)[0]
    # Plot some random misclassified digits.
    plotter.filename = 'misclassified'
    plotter.plot_classified_eegs(x_test[eegs_indexes, :], y_predicted[eegs_indexes], y_test[eegs_indexes], num=4)


def main():
    # Get x and y pairs.
    x_train, y_train, x_test, y_test = get_x_y()

    # Preprocess data.
    x_train_clean, y_train_clean, x_test_clean = preprocess(x_train, y_train, x_test)

    # Create model, fit and predict.
    y_predicted = fit_predict(x_train_clean, y_train_clean, x_test_clean)

    # Show prediction information.
    show_prediction_info(y_test, y_predicted)

    # Show some of the classification results.
    display_classification_results(x_test, y_test, y_predicted)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
