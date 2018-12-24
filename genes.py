import time
import numpy
import helpers
from core import KPCA, Kernels, Lda
from sklearn import metrics, neighbors
from sklearn import preprocessing
from definitions import CREATE_PLOTS, SAVE_PRED_RESULTS

# Create a logger.
logger = helpers.Logger(folder='logs', filename='genes')

# If plots are enabled, create a plotter.
if CREATE_PLOTS:
    plotter = helpers.Plotter(folder='plots')


def get_x_y():
    logger.log('Loading Dataset...')
    x_train, y_train = helpers.datasets.load_genes()
    logger.log(str(len(y_train)) + ' training data loaded')

    x_test, y_test = helpers.datasets.load_genes(train=False)
    logger.log(str(len(y_test)) + ' test data loaded')

    return x_train, y_train, x_test, y_test


def plot_pca(pca, y_train):
    # Plot pca pov vs k.
    pca_params = pca.get_params()
    plotter.subfolder = 'pov_analysis'
    plotter.filename = 'pca'
    plotter.xlabel = 'Number of Principal Components'
    plotter.title = 'POV vs K\nKernel: {} Components: {}'. \
        format(pca_params['kernel'], pca_params['n_components'])
    plotter.pov_analysis(pca.explained_var)

    # Plot first three PCs.
    plotter.subfolder = 'scatters/pca'
    plotter.filename = '3pcs'
    plotter.xlabel = 'pc1'
    plotter.ylabel = 'pc2'
    plotter.zlabel = 'pc3'
    plotter.title = 'The first three Principal Components\nKernel: {} Components: {}'. \
        format(pca_params['kernel'], pca_params['n_components'])
    plotter.scatter(pca.alphas[:, :3], y_train, class_labels=helpers.datasets.get_gene_name)

    # Plot first two PCs.
    plotter.title = 'The first two Principal Components\nKernel: {} Sigma: {}, Components: {}'. \
        format(pca_params['kernel'], pca_params['sigma'], pca_params['n_components'])
    plotter.filename = '2pcs'
    plotter.scatter(pca.alphas[:, :2], y_train, class_labels=helpers.datasets.get_gene_name)

    # Plot first PC.
    plotter.title = 'The first Principal Component\nKernel: {} Sigma: {}, Components: {}'. \
        format(pca_params['kernel'], pca_params['sigma'], pca_params['n_components'])
    plotter.filename = '1pc'
    plotter.scatter(pca.alphas[:, 0], y_train, class_labels=helpers.datasets.get_gene_name)


def plot_lda(lda, x_train, y_train):
    # Plot lda pov vs k.
    lda_params = lda.get_params()
    plotter.subfolder = 'pov_analysis'
    plotter.filename = 'lda'
    plotter.xlabel = 'Number of LDA Features'
    plotter.title = 'LDA POV vs K\nComponents: {}'.format(lda_params['n_components'])
    plotter.pov_analysis(lda.explained_var)

    # Plot first three LDA features.
    plotter.subfolder = 'scatters/lda'
    plotter.title = 'The first 3 LDA features.\nComponents: {}'.format(lda_params['n_components'])
    plotter.xlabel = 'First LDA Feature'
    plotter.ylabel = 'Second LDA Feature'
    plotter.zlabel = 'Third LDA Feature'
    plotter.filename = '3lds'
    plotter.scatter(x_train[:, :3], y_train, class_labels=helpers.datasets.get_gene_name)

    # Plot first two LDA features.
    plotter.title = 'The first 2 LDA features.\nComponents: {}'.format(lda_params['n_components'])
    plotter.filename = '2lds'
    plotter.scatter(x_train[:, :2], y_train, class_labels=helpers.datasets.get_gene_name)

    # Plot first LDA feature.
    plotter.title = 'The first LDA feature.\nComponents: {}'.format(lda_params['n_components'])
    plotter.filename = '1ld'
    plotter.scatter(x_train[:, 0], y_train, class_labels=helpers.datasets.get_gene_name)


def preprocess(x_train, y_train, x_test):
    logger.log('Preprocessing...')

    # Symmetrize dataset.
    logger.log('\tSymmetrizing training dataset...')
    x_train, y_train = helpers.preprocessing.symmetrize_dataset(x_train, y_train)
    logger.log('\t' + str(len(y_train)) + ' training data remained')

    # Scale data
    logger.log('\tScaling data, using Min Max Scaler with params:')
    scaler = preprocessing.MinMaxScaler()
    logger.log('\t' + str(scaler.get_params()))
    x_train = scaler.fit_transform(x_train.astype(float))
    x_test = scaler.transform(x_test.astype(float))

    # Apply KPCA
    logger.log('\tApplying Principal Component Analysis with params:')
    pca = KPCA(Kernels.LINEAR, n_components=0.9)
    logger.log('\t' + str(pca.get_params()))
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    # Apply LDA
    logger.log('\tApplying Linear Discriminant Analysis with params:')
    lda = Lda()
    logger.log('\t' + str(lda.get_params()))
    x_train = lda.fit_transform(x_train, y_train)
    x_test = lda.transform(x_test)

    if CREATE_PLOTS:
        plot_pca(pca, y_train)
        plot_lda(lda, x_train, y_train)

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


def show_prediction_info(y_test, y_predicted, folder: str = 'results/genes-tests', filename: str = 'example-bad',
                         extension: str = 'xlsx', sheet_name: str = 'results'):
    # Get the accuracy of each class.
    accuracies = helpers.utils.cm_to_accuracies(metrics.confusion_matrix(y_test, y_predicted))

    # Create results dictionary.
    results = {'Breast invasive carcinoma Accuracy': [accuracies[0]],
               'Colon adenocarcinoma Accuracy': [accuracies[1]],
               'Kidney renal clear cell carcinoma Accuracy': [accuracies[2]],
               'Lung adenocarcinoma Accuracy': [accuracies[3]],
               'Prostate adenocarcinoma Accuracy': [accuracies[4]],
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
    if SAVE_PRED_RESULTS:
        helpers.utils.create_excel(results, folder, filename, extension, sheet_name)


def display_classification_results(x_test, y_test, y_predicted):
    logger.log('Plotting some random correctly classified Genes.')
    # Get indexes of misclassified Genes.
    genes_indexes = numpy.where(y_test == y_predicted)[0]
    # Plot some random misclassified Genes.
    plotter.filename = 'correct'
    plotter.subfolder = 'genes'
    plotter.plot_classified_genes(x_test[genes_indexes, :], y_predicted[genes_indexes], y_test[genes_indexes], num=4)

    logger.log('Plotting some random misclassified Genes.')
    # Get indexes of misclassified Genes.
    genes_indexes = numpy.where(y_test != y_predicted)[0]
    # Plot some random misclassified Genes.
    plotter.filename = 'misclassified'
    plotter.plot_classified_genes(x_test[genes_indexes, :], y_predicted[genes_indexes], y_test[genes_indexes], num=4)


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
    if CREATE_PLOTS:
        display_classification_results(x_test, y_test, y_predicted)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
