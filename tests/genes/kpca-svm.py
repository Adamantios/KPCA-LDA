import time
import numpy
import pandas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import helpers
from core import KPCA, Kernels
from sklearn import metrics, neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create a logger and a plotter.
logger, plotter = helpers.Logger(filename='genes_kpca+svm_gridsearch'), helpers.Plotter(
    folder='plots/genes-tests')


def get_x_y():
    logger.log('Loading Dataset...')
    x, y = helpers.datasets.load_digits()
    logger.log(str(len(y)) + ' data loaded')

    logger.log('Splitting to 70% train and 30% test data...')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    logger.log(str(len(y_train)) + ' training data available.')
    logger.log(str(len(y_test)) + ' testing data available.')

    return x_train, y_train, x_test, y_test


def preprocess(x_train, y_train, x_test):
    logger.log('Preprocessing...')

    logger.log('\tScaling data, using Min Max Scaler with params:')
    scaler = preprocessing.MinMaxScaler()
    logger.log('\t' + str(scaler.get_params()))
    scaler.fit(x_train.astype(float))
    x_train = scaler.transform(x_train.astype(float))
    x_test = scaler.transform(x_test.astype(float))

    logger.log('\tApplying Principal Component Analysis with params:')
    pca = KPCA(Kernels.RBF, sigma=4.5, n_components=0.99)
    logger.log('\t' + str(pca.get_params()))
    x_train = pca.fit_transform(x_train)

    return x_train, y_train, x_test


def show_prediction_info(grid_results):
    logger.log("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))
    means = grid_results.cv_results_['mean_test_score']
    stds = grid_results.cv_results_['std_test_score']
    params = grid_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        logger.log("%f (%f) with: %r" % (mean, stdev, param))

    results = {'Mean': means,
               'Std': stds,
               'Params': params}

    helpers.utils.create_excel(results, 'excel-results', 'kpca_svm_grid_search', 'xlsx', 'gridsearch_results')


def main():
    # Get x and y pairs.
    x_train, y_train, x_test, y_test = get_x_y()

    # logger.log('Creating heatmap of the training samples correlation...')
    # plotter.heatmap_correlation(pandas.DataFrame(x_train).corr(), 'Features', 'Features')

    # Preprocess data.
    x_train_clean, y_train_clean, x_test_clean = preprocess(x_train, y_train, x_test)

    model = SVC()

    # Grid parameters.
    param_grid = [
        {'C': [10, 100, 1000], 'kernel': ['linear']},
        {'C': [10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4]},
        {'C': [10, 100, 1000], 'kernel': ['rbf']},
    ]

    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, scoring='accuracy', verbose=10, cv=4)

    grid_results = grid_search.fit(x_train_clean, y_train_clean)

    show_prediction_info(grid_results)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
