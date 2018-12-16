from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVC
from sklearn.decomposition import KernelPCA

import helpers

# Create a logger and a plotter.
logger, plotter = helpers.Logger(folder='logs/digits-tests', filename='kpca_grid_search'), helpers.Plotter(
    folder='plots/digits-tests')


def get_x_y():
    logger.log('Loading Dataset...')
    x, y = helpers.datasets.load_digits()
    logger.log(str(len(y)) + ' data loaded')
    return x, y


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

    helpers.utils.create_excel(results, 'excel-results', 'kpca_grid_search', 'xlsx', 'gridsearch_results')


def main():
    # Get x and y pairs.
    x_train, y_train = get_x_y()

    # Grid
    pipe = Pipeline(steps=[('quantile_transformer', QuantileTransformer(random_state=0)),
                           ('pca', KernelPCA()),
                           ('svc', LinearSVC())])

    # Grid parameters.
    param_grid = dict(pca__kernel=['linear', 'poly', 'rbf'],
                      pca__gamma=[0.05, 0.01, 0.005, 0.001],
                      pca__degree=[2, 3, 4],
                      pca__coef0=[0, 10, 100, 1000])

    grid_search = GridSearchCV(pipe, param_grid, n_jobs=-1, scoring='accuracy', verbose=10, cv=3)

    grid_results = grid_search.fit(x_train, y_train)

    show_prediction_info(grid_results)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
