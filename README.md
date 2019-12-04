# PlantCLEF Image Classification with Transfer Learning Project
Final project of DA514 Machine Learning Course of MSc in Data Analytics in Sabancı University.

## Project Purpose
The purpose of the project recognizing plant images and classify various plants by using output of last layer of the CNN model which is trained for PlantCLEF competition.

## Dataset
Output of last layer of CNN model pretrained for the competiton is used. Dataset consists of following features:
* 1024 CNN Features
* Date
* Location
* Latitude
* Longitude
* Class ID

## Project Outline

### Handling Missing Values in Coordinates

Missing coordinate values (rows with location values are not missing) are filled using Nominatim Api. Rows where both coordinates and location columns are empty are remained empty in this step.

### Date Categories

Creating month, season, quarter and half year categories from date feature.

### Location Clustering

Location cluster feature is created with DBSCAN algorithm by using coordinates. Largest cluster that contains samples from France divided into to cluster from 46th latitude.

Clusters of new samples (also both clusters of each validation set and test set) will be predicted with Logistic Regression in next steps.

### Handling Missing Values

To impute remaining missing values in location cluster and month categories, 1024 CNN features are used.

To determine imputation model and its hyper parameters, GridSearchCV is implemented to Logistic Regression, KNN and Random Forest models. Logistic Regression is determined to use for imputation during model evalution with cross validation.

### Model Evaluation

Cross Validation is implemented to evaluate different models. StratifiedKFold is used during preprocessing of training and validation sets. Then, cross validation with hyper parameter tuning is implemented.

Artificial Neural Network, Logistic Regression, KNN, SGDClassifier and Random Forest Classifier models are tested and ANN is selected as final model. (Only evaluation of ANN model is shown in the code.)

### Final Test and Results

Final model test is made with ANN on the test set. 84.27% accuracy is achieved on test dataset.
