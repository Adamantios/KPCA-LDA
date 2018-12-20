import time
import numpy
import pandas
import helpers
from core import KPCA, Kernels
from sklearn import metrics, neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create a logger and a plotter.
logger, plotter = helpers.Logger(folder='logs/genes-tests', filename='rbf_kpca+lda'), \
                  helpers.Plotter(folder='plots/genes-tests')


def get_x_y():
    logger.log('Loading Dataset...')
    x, y = helpers.datasets.load_genes()
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
    pca = KPCA(Kernels.RBF, sigma=4.5, n_components=20531)
    logger.log('\t' + str(pca.get_params()))
    pca.fit_transform(x_train)

    # Plot pca pov vs k.
    plotter.pov_analysis(pca.explained_var, subfolder='pca_analysis/all_components', filename='rbf')

    logger.log('\tApplying Principal Component Analysis with params:')
    pca = KPCA(Kernels.RBF, sigma=4.5, n_components=0.99)
    logger.log('\t' + str(pca.get_params()))
    x_train = pca.fit_transform(x_train)

    # Plot pca pov vs k.
    plotter.pov_analysis(pca.explained_var, subfolder='pca_analysis/pov_0.99', filename='rbf')

    plotter.scatter_pcs(pca.alphas[:, :3], y_train, class_labels=helpers.datasets.get_eeg_name,
                        subfolder='scatters/pca/rbf', filename='pov_0.99_3pcs')
    plotter.scatter_pcs(pca.alphas[:, :2], y_train, class_labels=helpers.datasets.get_eeg_name,
                        subfolder='scatters/pca/rbf', filename='pov_0.99_2pcs')
    plotter.scatter_pcs(pca.alphas[:, 0], y_train, class_labels=helpers.datasets.get_eeg_name,
                        subfolder='scatters/pca/rbf', filename='pov_0.99_1pc')

    x_test = pca.transform(x_test)

    logger.log('\tApplying Linear Discriminant Analysis with params:')
    lda = LinearDiscriminantAnalysis()
    logger.log('\t' + str(lda.get_params()))
    x_train = lda.fit_transform(x_train, y_train)
    x_test = lda.transform(x_test)

    # Plot lda pov vs k.
    plotter.pov_analysis(lda.explained_variance_ratio_, subfolder='lda_analysis/pov_0.9', filename='rbf')

    plotter.scatter_pcs(x_train[:, :3], y_train, class_labels=helpers.datasets.get_eeg_name,
                        subfolder='scatters/lda/rbf', filename='pov_0.9_3pcs')
    plotter.scatter_pcs(x_train[:, :2], y_train, class_labels=helpers.datasets.get_eeg_name,
                        subfolder='scatters/lda/rbf', filename='pov_0.9_2pcs')
    plotter.scatter_pcs(x_train[:, 0], y_train, class_labels=helpers.datasets.get_eeg_name,
                        subfolder='scatters/lda/rbf', filename='pov_0.9_1pc')

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


def show_prediction_info(y_test, y_predicted, save: bool = True, folder: str = 'results',
                         filename: str = 'seizure_detection_train', extension: str = 'xlsx',
                         sheet_name: str = 'results'):
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
    if save:
        helpers.utils.create_excel(results, folder, filename, extension, sheet_name)


def display_classification_results(x_test, y_test, y_predicted):
    pass


def main():
    # Get x and y pairs.
    x_train, y_train, x_test, y_test = get_x_y()

    # logger.log('Creating heatmap of the training samples correlation...')
    # plotter.heatmap_correlation(pandas.DataFrame(x_train).corr(), 'Features', 'Features')

    # Preprocess data.
    x_train_clean, y_train_clean, x_test_clean = preprocess(x_train, y_train, x_test)

    # logger.log('Creating heatmap of the principal components correlation...')
    # plotter.heatmap_correlation(pandas.DataFrame(pca_components).corr(),
    #                             'Principal components', 'Principal components', filename='heatmap_pca_corr')

    # logger.log('Creating heatmap of the principal components and initial features correlation...')
    # plotter.heatmap_correlation(pca_components, 'Features', 'Principal components', filename='heatmap_pca')

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