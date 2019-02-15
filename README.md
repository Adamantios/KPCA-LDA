# KPCA-LDA
KPCA+LDA project for Statistical ML Lesson.  
Core folder includes simple implementations of KPCA and LDA from scratch.  
KPCA+LDA is being applied for [seizure detection](http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition) 
and [cancer recognition from rna seq](http://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq).  
This project produces results, POV vs number of features diagrams for KPCA and LDA, Data visualisation in 1D, 2D and 3D and classification examples for EEGs.

## Usage
#### Install requirements
`pip install -r path_to_KPCA+LDA_project/requirements.txt`

#### Download datasets
1. [Seizure Detection](http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)  
2. [RNA Seq](http://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq)

#### Add files to project
* Create a folder on root ex. `datasets` and move downloaded datasets there.
* Open `definitions.py` from root.
* Set `DATASETS` variable with the datasets folder name ex. `DATASETS='datasets'`.
* Set DATASET_SEIZURE with the filename of the seizures dataset ex. `DATASET_SEIZURE='seizure.csv'`.
* Set DATA_GENES with the filename of the RNA Seq data ex. `DATA_GENES='genes.csv'`.
* Set LABELS_GENES with the filename conaining the classes for RNA Seq dataset ex. `LABELS_GENES='genes_labels.csv'`.
* Extra configurations can be optionaly made through the `definitions.py` file.

## Examples
#### Log
```
----------- 2018-12-23 23:24:09.387296 -----------

Loading Dataset...
6900 training data loaded
4600 test data loaded
Preprocessing...
	Scaling data, using Min Max Scaler with params:
	{'copy': True, 'feature_range': (-1, 1)}
	Applying Principal Component Analysis with params:
	{'kernel': 'RBF', 'alpha': 'auto', 'coefficient': 0, 'degree': 3, 'sigma': 0.3, 'n_components': 0.95}
	Applying Linear Discriminant Analysis with params:
	{'n_components': 'auto', 'remove_zeros': True}
Creating KNN model with params:
{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 50, 'p': 2, 'weights': 'uniform'}
Fitting model...
Model has been fit in 0.0334 seconds.
Predicting...
Prediction finished in 0.398 seconds.
Model's Results:
Epileptic Seizure Accuracy: 0.612
Tumor Located Accuracy: 0.6761
Healthy Area Accuracy: 0.3641
Eyes Closed Accuracy: 0.375
Eyes Opened Accuracy: 0.9478
Accuracy: 0.595
Precision: 0.5863
Recall: 0.595
F1: 0.5869
```
