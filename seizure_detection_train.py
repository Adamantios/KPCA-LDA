import time
import numpy
import pandas
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

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
    pca = PCA(whiten=True, random_state=0)
    logger.log('\t' + str(pca.get_params()))
    pca.fit(x_train)

    # Plot pca pov vs k.
    plotter.pca_analysis(pca.explained_variance_ratio_)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    return x_train, y_train, x_test, pca.components_


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


def show_prediction_info(y_test, y_predicted, support_vectors):
    logger.log('Model\'s scores:')
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    precision = metrics.precision_score(y_test, y_predicted, average='macro')
    recall = metrics.recall_score(y_test, y_predicted, average='macro')
    f1 = metrics.f1_score(y_test, y_predicted, average='macro')
    logger.log('Accuracy: {:.4}'.format(accuracy))
    logger.log('Precision: {:.4}'.format(precision))
    logger.log('Recall: {:.4}'.format(recall))
    logger.log('F1: {:.4}'.format(f1))

    # Calculate and print accuracy for each class.
    logger.log('Accuracy for each class: ')
    cm = metrics.confusion_matrix(y_test, y_predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    logger.log(cm.diagonal())

    logger.log(str(len(support_vectors)) + ' support vectors used.')


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
    x_train_clean, y_train_clean, x_test_clean, pca_components = preprocess(x_train, y_train, x_test)

    # logger.log('Creating heatmap of the principal components correlation...')
    # plotter.heatmap_correlation(pandas.DataFrame(pca_components).corr(),
    #                             'Principal components', 'Principal components', filename='heatmap_pca_corr')

    # logger.log('Creating heatmap of the principal components and initial features correlation...')
    # plotter.heatmap_correlation(pca_components, 'Features', 'Principal components', filename='heatmap_pca')

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
