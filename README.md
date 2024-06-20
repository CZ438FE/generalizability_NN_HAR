# Training pipeline


## Purpose


This repository was constructed for the study "Generalizability of neural networks for human activity recognition systems based on active low-back exoskeleton data", written by Christoph Zink, M.Sc. in cooperation with Prof. Christian Ledig from the University of Bamberg at the Chair of Explainable Machine Learning and Joachim Fischer, M. Sc. of the company German Bionic. The code was written in its entirety by Christoph zink. This repository contains the code utilized for the preprocessing, construction, training and evaluation of the various models for this study. The study was conducted in cooperation with the company German Bionic, therefore some knowledge owned by this company was used. Whenever this was done, it was imported into this repository via an anonymization_file.json. To enable usage of this repository without this file the dependency on this file was limited as far as possible, enabling usage of the provided code after minor adjustments.



## Content

This repository contains all the needed functionality for the processing of the labeled data and the training and evaluation of the models.

It consists of several chapters with own functionality. The following entails a short description of each chapter, a more detailed description and explanation of the available parameters is given in the READMEs in the respective folders.


### Create_windows

Is used to cut the collection of labeled data into sliding windows of the given **window_size** (in ms) and creates a new windows after a given **step_length** (in ms) has passed. Additionally if performs some basic logical checks and transforms the time column to UNIX in ms, if needed. For data consistency purposes the data is also resampled, the timespan to resample may also be given via the **resampling_rate** (in ms) parameter. Before the resampling is applied the data is being filled via a specified **method** and afterwards flattening the data may be applied (convert data to one-dimensional array) via the **flatten** keyword. 


### Generate_features

Uses a collection of windowed data to generate a set of features for each of these windows. It returns a csv file for every window containing all features.


### Preprocess

Contains some general-purpose prepocessing utilities, which can be individually triggered by specifying a method:

**reduce_data** uses a given level of granularity to iterate over a set of files containing features and keeps only the columns, which are useful for a Human activity recognition (HAR) analyzer at the given granularity level. The main purpose of this method is therefore data-reduction via removing the features, which are not being used for the baseline model at the given granularity.

**balancing_over** implements a stable way to oversample the data for the granularity level, returns a collection of files with several windows and the respective data with balanced label distribution at the given granularity. The implemented oversampling algorithm is a naive algorithm, which adds randomly drawn samples from the minority classes to the dataset, until the class distribution is balanced.

**prepare_dataset** prepares oversampled data for the training of the model. Therefore the data of each window is being saved as an individual file. The label gets prepared and saved together with the respective filename in the labels.csv. Saving each window individually is necessary to enable minibatch-wise training.


### Train_model

Provides all the necessary utilities to train a model. All the hyperparameters may be given via the command line. Also provides the possibility to estimate the performance on a validation-set, as well as to visualize the change of loss and accuracy over the minibatches. It enables training Feedforward Neural Networks (FFNN), Convolutional Neural Networks (CNN) and recurrent neural networks using long short-term memory units (LSTM).

### Evaluate_Model

Loads a trained Model and evaluates the performance of said model on new test-data. Afterwards a classification matrix, as well as several scores are reported to evaluate the performance of the model.


### Evaluate_Hierarchical
As the hierarchically classifying models consists of various submodels, this provides the utilities to gather the performance of said classifiers and evaluate them in a joined manner. Reports the same metrics as evaluate_model, albeit for the collection of three classifiers.


### Evaluate_classwise
Reports some evaluations on the performance of a classifier when the respective k-class classification problem is being broken up into a collection of k class vs rest classification problems. Also allows for the plotting of class-wise ROC curves.


### Compare_Models
Provides the utilities to perform statistical analysis of the trained models, i.e. evaluates if the difference in classification performance are statistically significant. Currently this entails McNemars test, the Friedman test and the Wilcoxon signed rank test.



## Process through the pipeline:

### Models to be trained on pre-computed featues:

Create_windows -> Generate_features -> Preprocess method reduce_data -> Preprocess method balancing_over -> Preprocess prepare_dataset -> train_model -> evaluate_model -> Evaluate_Hierarchical

### Models classifying directly on the raw data:

Create_windows -> Preprocess method balancing_over -> Preprocess prepare_dataset (with --grid if a CNN or LSTM shall be trained) -> train_model -> evaluate_model

