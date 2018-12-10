import time
import numpy
import pandas
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA

import helpers

# Create a logger and a plotter.
logger, plotter = helpers.Logger(filename='seizure_detection_train_results'), helpers.Plotter()


def get_x_y():
    logger.log('Loading Dataset...')
    x, y = helpers.datasets.load_seizure()
    logger.log(str(len(y)) + ' data loaded')

    logger.log('Splitting to 60% train and 40% test data...')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    logger.log(str(len(y_train)) + ' training data available.')
    logger.log(str(len(y_test)) + ' testing data available.')

    return x_train, y_train, x_test, y_test


def preprocess(x_train, y_train, x_test):
    num_of_features = len(x_train[1])
    logger.log('Preprocessing...')

    # logger.log('\tSymmetrize training dataset...')
    # x_train, y_train = helpers.preprocessing.symmetrize_dataset(x_train, y_train, 1100)
    # logger.log('\t' + str(len(y_train)) + ' training data remained')

    logger.log('\tScaling data, using Quantile Transformer with params:')
    scaler = preprocessing.QuantileTransformer(random_state=0)
    logger.log('\t' + str(scaler.get_params()))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    logger.log('\tApplying Principal Component Analysis with params:')
    pca = KernelPCA(kernel='linear', random_state=0)
    logger.log('\t' + str(pca.get_params()))
    x_train = pca.fit_transform(x_train)
    # Plot pca pov vs k.
    plotter.pca_analysis(helpers.utils.calc_explained_var_ratio(x_train), num_of_features,
                         subfolder='pca_analysis/all_components', filename='linear')

    x_test = pca.transform(x_test)

    return x_train, y_train, x_test, pca.alphas_


def fit_predict(x_train, y_train, x_test):
    logger.log('Creating SVM model with params:')
    model = SVC(C=30, gamma='auto', class_weight={1: .05, 2: .05, 3: .1, 4: .25, 5: .35}, random_state=0)
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


def show_prediction_info(y_test, y_predicted, support_vectors, save: bool = True, folder: str = 'results',
                         filename: str = 'seizure_detection_train', extension: str = 'xlsx',
                         sheet_name: str = 'results'):
    # Get the accuracy of each class.
    accuracies = helpers.utils.cm_to_accuracies(metrics.confusion_matrix(y_test, y_predicted))

    # Create results dictionary.
    results = {'Epileptic Seizure Accuracy': accuracies[0],
               'Tumor Located Accuracy': accuracies[1],
               'Healthy Area Accuracy': accuracies[2],
               'Eyes Closed Accuracy': accuracies[3],
               'Eyes Opened Accuracy': accuracies[4],
               'Accuracy': metrics.accuracy_score(y_test, y_predicted),
               'Precision': metrics.precision_score(y_test, y_predicted, average='macro'),
               'Recall': metrics.recall_score(y_test, y_predicted, average='macro'),
               'F1': metrics.f1_score(y_test, y_predicted, average='macro'),
               'Support Vectors': len(support_vectors)}

    # Log results.
    logger.log('Model\'s Results:')
    for key, value in results.items():
        logger.log('{text}: {number:.{points}g}'.format(text=key, number=value, points=4))

    # Create excel if save is True.
    if save:
        helpers.utils.create_excel(results, folder, filename, extension, sheet_name)


def display_classification_results(x_test, y_test, y_predicted):
    logger.log('Plotting some random correctly classified EEGs.')
    # Get indexes of misclassified digits.
    eegs_indexes = numpy.where(y_test == y_predicted)[0]
    # Plot some random misclassified digits.
    plotter.plot_classified_eegs(x_test[eegs_indexes, :], y_predicted[eegs_indexes], y_test[eegs_indexes], num=4,
                                 filename='correct')

    logger.log('Plotting some random misclassified EEGs.')
    # Get indexes of misclassified digits.
    eegs_indexes = numpy.where(y_test != y_predicted)[0]
    # Plot some random misclassified digits.
    plotter.plot_classified_eegs(x_test[eegs_indexes, :], y_predicted[eegs_indexes], y_test[eegs_indexes], num=4,
                                 filename='misclassified')


def main():
    # Get x and y pairs.
    x_train, y_train, x_test, y_test = get_x_y()

    # logger.log('Creating heatmap of the training samples correlation...')
    # plotter.heatmap_correlation(pandas.DataFrame(x_train).corr(), 'Features', 'Features')

    # Preprocess data.
    x_train_clean, y_train_clean, x_test_clean, eigenvectors = preprocess(x_train, y_train, x_test)

    # logger.log('Creating heatmap of the principal components correlation...')
    # plotter.heatmap_correlation(pandas.DataFrame(pca_components).corr(),
    #                             'Principal components', 'Principal components', filename='heatmap_pca_corr')

    # logger.log('Creating heatmap of the principal components and initial features correlation...')
    # plotter.heatmap_correlation(pca_components, 'Features', 'Principal components', filename='heatmap_pca')

    # Create model, fit and predict.
    # y_predicted, support_vectors = fit_predict(x_train_clean, y_train_clean, x_test_clean)

    # Show prediction information.
    # show_prediction_info(y_test, y_predicted, support_vectors)

    # Show some of the classification results.
    # display_classification_results(x_test, y_test, y_predicted)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
