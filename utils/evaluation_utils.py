import os
import json
import warnings
import random
import time
import math
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import classification_report, cohen_kappa_score, matthews_corrcoef, fbeta_score
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon, friedmanchisquare

from config.log import logger
from utils import file_utils, time_utils, train_model_utils, plot_utils, data_utils


def valid_table(table: dict):
    """Checks if all the given parameters in the table are valid for the evaluation of the model
    """
    if not os.path.exists(table.get("test_data")):
        logger.error(f"The given test_data folder does not exist: {table.get('test_data')}")
        return "invalid_test_data_given"

    if table.get("dryrun"):
        logger.warning("Storing results is highly recommended")

    if not isinstance(table.get("dryrun"), bool):
        logger.error("Got dryrun of nonbool dtype")
        return "got_dryrun_of_nonbool_dtype"

    if not os.path.exists(table.get("model_folder")):
        logger.error(f"The given model_folder does not exist: {table.get('model_folder')}")
        return "invalid_model_folder_given"

    model_files = []
    json_files, error = file_utils.get_files(table.get("model_folder"), "json")
    if error:
        return error

    if "SVM" not in table.get("model_folder"):
        pt_files, error = file_utils.get_files(table.get("model_folder"), "pt")
        if error:
            return error
        model_files.extend(pt_files)

    model_files.extend(json_files)

    if not model_files:
        logger.error(f"The given model_folder does not contain a model file: {table.get('model_folder')}")
        return "invalid_model_folder_given"

    test_files, error = file_utils.get_files(table.get("test_data"), "csv")
    if error:
        logger.error(f"The test_data folder does not contain any csv files: {table.get('test_data')}")
        return error

    if not os.path.exists(os.path.join(table.get("test_data"), "log.json")):
        logger.error("test_data folder does not contain a log file with the prior treatments")
        return "test_data_log_file_not_found"

    if table.get("test_data") == table.get("model_folder"):
        logger.warning(f"Got same folder for the storage of model and the test data")
        return "received_identical_model_and_data_folder"

    if table.get("output_date") is not None:
        error = time_utils.valid_time_string(table.get("output_date"))
        if error:
            return error

    return None


def valid_preprocessing_of_data_for_model(model_folder: str, test_data: str):
    """Uses the log file to test if the preprocessing steps of the test data are suited for the given model
    """
    if not os.path.exists(model_folder):
        logger.error("Given model folder does not exist")
        return "given_model_folder_does_not_exist"

    if not os.path.exists(os.path.join(model_folder, "log.json")):
        logger.error("Given model folder does not contain a log file")
        return "given_model_folder_does_not_contain_log_file"

    if not os.path.exists(test_data):
        logger.error("Given test_data folder does not exist")
        return "given_test_data_folder_does_not_exist"

    if not os.path.exists(os.path.join(test_data, "log.json")):
        logger.error("Given test_data folder does not contain a log file")
        return "given_test_data_folder_does_not_contain_log_file"

    with open(os.path.join(test_data, "log.json")) as f:
        test_data_log_file = json.load(f)

    with open(os.path.join(model_folder, "log.json")) as f:
        model_log_file = json.load(f)

    # As the log.json of the SVM folder only contains which level the seen SVM classifies, no info regarding prior processing exists
    if "SVM" in model_folder:
        return None

    for preprocessing_step in test_data_log_file.keys():
        if preprocessing_step not in model_log_file.keys() and preprocessing_step != "balancing_evaluation":
            logger.error(f"The Preprocessing step {preprocessing_step} was executed on the test-data, but not on the model")
            return "preprocessing_steps_differ"

    if "create_windows" not in test_data_log_file.keys():
        logger.error(f"Evaluating models for models without data in windowed format not yet implemented")
        return "received_unwindowed_data_not_yet_implemented"

    test_data_window_length = test_data_log_file.get("create_windows").get("window_length")
    model_window_length = model_log_file.get("create_windows").get("window_length")
    test_data_balancing_type = [key for key in test_data_log_file if "balancing" in key][0]
    if test_data_window_length != model_window_length:
        logger.error(f"The model was trained on windows of length {model_window_length}, whereas the length seen in the test data is {test_data_window_length}")
        return "differing_window_lengths_found"

    if test_data_log_file.get("create_windows").get("flatten") != model_log_file.get("create_windows").get("flatten"):
        logger.error(f"The model was trained on  flattened windows: {model_log_file.get('create_windows').get('flatten')}, whereas the test data is flattened: {test_data_log_file.get('create_windows').get('flatten')}")
        return "differing_flatten_values_found"

    if test_data_log_file.get("create_windows").get("method") != model_log_file.get("create_windows").get("method"):
        logger.error(f"The model data was filled via : {model_log_file.get('create_windows').get('method')}, whereas the test data was filled via: {test_data_log_file.get('create_windows').get('method')}")
        return "differing_filling_methods_values_found"

    if test_data_log_file.get(test_data_balancing_type).get("granularity") != model_log_file.get("balancing_over").get("granularity"):
        logger.error(f"The model data was balanced based on granularity : {model_log_file.get('balancing_over').get('granularity')}, whereas the test data was balanced based on granularity: {test_data_log_file.get('balancing_over').get('granularity')}")
        return "differing_balancing_granularities_found"

    if "prepare_dataset" not in test_data_log_file.keys():
        logger.error(f"Tried evaluation of data that was not preprocessed with method prepare_dataset yet")
        return "Unprepared_data_given"

    if test_data_log_file.get("prepare_dataset").get("grid") != model_log_file.get("prepare_dataset").get("grid"):
        logger.error(f"The model data contains grid data: {model_log_file.get('prepare_dataset').get('grid')}, whereas the test data contains convolutional data: {test_data_log_file.get('prepare_dataset').get('grid')}")
        return "differing_grid_values_found"

    list_of_used_folders = [model_log_file.get("train_model").get("training_data"), model_log_file.get("train_model").get("validation_data")]
    if test_data in list_of_used_folders:
        logger.error(f"Evaluating the model on data which was used for training is not allowed due to data leakage problems")
        return "training_folder_cannot_be_used_for_evaluation"

    return None


def load_trained_model(model_folder: str, log_file: dict):
    """returns the loaded model from the folder
    """
    network_type, error = detect_network_type(model_folder)
    if error:
        return None, error

    if os.path.exists( os.path.join(model_folder, "best_val_model.pt")):
        return train_model_utils.load_model(model_folder, network_type, log_file, "best_val_model.pt")

    return train_model_utils.load_model(model_folder, network_type, log_file)


def detect_network_type(model_folder: str):
    """reads in the log file of the given folder to return the seen model type
    """
    with open(os.path.join(model_folder, "log.json")) as f:
        model_log_file = json.load(f)

    if "SVM" in model_folder:
        if model_log_file.get("train_model").get("type") != "SVM":
            logger.error(f"received a model in a SVM folder which is not a SVM : {model_log_file.get('train_model').get('type')}")
            return None, "found_non_svm_model_in_svm_folder"

        if model_log_file.get("train_model").get("hierarchical_model") not in ["top", "walking", "lifting"]:
            logger.error(f"received a SVM for separation of unknown classes: {model_log_file.get('train_model').get('hierarchical_model')}")
            return None, "found_svm_for_separation_of_unknown_classes"

        return "SVM", None

    if "balancing_over" not in model_log_file.keys():
        logger.error("Log file of model folder does not contain information regarding prior oversampling")
        return None, "no_oversampling_detected"

    if "prepare_dataset" not in model_log_file.keys():
        logger.error("Log file of model folder does not contain information regarding preparing the dataset before training")
        return None, "no_prepare_dataset_detected"

    if "generate_features" in model_log_file.keys() and "reduce_data" in model_log_file.keys():
        return "FFNN", None

    elif model_log_file.get("create_windows").get("flatten") and "generate_features" not in model_log_file.keys() and not model_log_file.get("prepare_dataset").get("grid"):
        return "FFNN", None

    elif model_log_file.get("create_windows").get("flatten") and "generate_features" not in model_log_file.keys() and model_log_file.get("prepare_dataset").get("grid") and model_log_file.get("train_model").get("type") == "CNN":
        return "CNN", None

    elif model_log_file.get("create_windows").get("flatten") and "generate_features" not in model_log_file.keys() and model_log_file.get("prepare_dataset").get("grid") and model_log_file.get("train_model").get("type") == "RNN":
        return "RNN", None

    logger.error("Could not detect the correct model type based on the log file")
    return None, "detecting_model_type_ failed"


def create_predictions(model: nn.Module, test_data_folder: str, model_folder: str, store_local: bool, saving_folder: str, verbose: bool, frac=1.0):
    """does the heavy lifting for the evaluation, e.g. returns classification for each valid datapoint in the test data and returns a df with the predictions
    """
    with open(os.path.join(model_folder, "log.json")) as f:
        model_folder_log_file = json.load(f)
    hierarchical_model: bool = model_folder_log_file.get("train_model").get("hierarchical_model")

    with open(os.path.join(test_data_folder, "log.json")) as f:
        test_folder_log_file = json.load(f)

    all_files, error = file_utils.get_files(test_data_folder, "csv")
    if error:
        return None, None, None, None, error

    all_files.remove(os.path.join(test_data_folder, "labels.csv"))

    all_files, error = train_model_utils.if_hierarchical_data_remove_unneeded_files(all_files, test_data_folder, hierarchical_model)
    if error:
        return None, None, None, None, error
    
    joined_filesize = file_utils.count_filesizes_of_list(all_files)
    file_utils.estimate_needed_time(len(all_files), "evaluate_model", os.cpu_count(), joined_filesize)

    data_in_channel_format = model_folder_log_file.get("train_model").get("type") in ["CNN", "RNN"]
    if "RNN" in model_folder or "LSTM" in model_folder:
        model = model.double()


    processing_neural_network = model_folder_log_file.get("train_model").get("type") in ["FFNN", "CNN", "RNN"]
    if processing_neural_network:
        model = model.eval()
    if not processing_neural_network:
        # As the baseline Method was fitted with featurenames, a Userwarning is being given for each window, therefore those are suppressed
        warnings.filterwarnings("ignore", category=UserWarning)

    hierarchical_data_handled = "generate_features" in model_folder_log_file.keys()
    standardizing_df, standardization_type, error = data_utils.prepare_standardization(model_folder, hierarchical_data_handled, hierarchical_model)
    if error:
        return None, None, None, None, error

    processing_aggregated_data = False
    if "balancing_evaluation" in test_folder_log_file:
        processing_aggregated_data = True if test_folder_log_file['balancing_evaluation']['classes_to_aggregate'] else False

    prediction_time = []
    probabilities = []

    for file_nr, file in enumerate(all_files):
        if (file_nr/len(all_files)) > frac:
            break

        if file_nr == 0 or file_nr % 2000 == 0:
            logger.info(f"[{file_nr}/{len(all_files)}] files classified")

        loader, y_transformed = create_loader_for_file(file, data_in_channel_format, hierarchical_data_handled, standardizing_df, standardization_type, hierarchical_model)

        if file_nr == 0:
            all_true_labels = y_transformed
        else:
            all_true_labels = torch.cat((all_true_labels, y_transformed), 0)

        with torch.no_grad():
            for minibatch_nr, minibatch in enumerate(loader):

                start = time.time()
                y_pred = model.predict(minibatch[0])
                prediction_time.append(time.time()-start)

                if isinstance(y_pred, np.ndarray):
                    y_pred = transform_svm_output_to_valid_tensor(y_pred, 3)

                probabilities.append(y_pred.tolist()[0])

                # transform the ground truth for this batch into an appropriate format for confusion matrix
                y_pred_transformed = train_model_utils.transform_predictions_for_confusion_matrix(y_pred)
                                
                if minibatch_nr == 0 and file_nr == 0:
                    y_pred_joined = y_pred_transformed
                else:
                    y_pred_joined = torch.cat((y_pred_joined, y_pred_transformed), 0)

    probabilities_df = pd.DataFrame(np.array(probabilities).reshape(len(probabilities), -1)) 
    if processing_aggregated_data:
        nr_wanted_classes = len(torch.unique(all_true_labels))
        y_pred_joined = train_model_utils.aggregate_predictions(y_pred_joined, nr_wanted_classes)

    if len(torch.unique(all_true_labels)) == len(torch.unique(y_pred_joined)) and not torch.equal(torch.unique(all_true_labels), torch.unique(y_pred_joined)):
        if torch.max(y_pred_joined) > torch.max(all_true_labels):
            y_pred_joined -=1
        else:
            all_true_labels -=1

    # Create the final confusion matrix of the data
    task_category = "binary" if len(torch.unique(all_true_labels)) == 2 else "multiclass"
    conf_mat = ConfusionMatrix(num_classes=len(torch.unique(all_true_labels)), task = task_category)
    confusion_matrix = conf_mat(y_pred_joined, all_true_labels)

    accuracy = round((torch.diagonal(confusion_matrix, 0).sum().item() * 100) / (torch.sum(confusion_matrix)).item(), 2)

    if verbose:
        logger.info(f"This is the Confusion Matrix of the test-data after Training: (n = {torch.sum(confusion_matrix).item()})\n{confusion_matrix}")
    logger.info(f"The final test-accuracy is {accuracy}%")

    plot_utils.plot_confusion_matrix(confusion_matrix, store_local, saving_folder, "test_data", hierarchical_model)
    
    results_df = pd.DataFrame({"file": all_files, "true_label": all_true_labels, "prediction": y_pred_joined})

    # Reverse the label, so that the seen label in the file name has the same meaning as the label in the respective columns
    results_df["true_label"] = results_df["true_label"].to_numpy() + 1
    results_df["prediction"] = results_df["prediction"].to_numpy() + 1

    probabilities_df = probabilities_df.set_axis([f"Raw_prediction_label_{int(label) +1}" for label in probabilities_df.columns], axis=1)
    results_df = pd.concat([results_df, probabilities_df], axis=1)
    return confusion_matrix, accuracy, results_df, pd.Series(prediction_time), None


def create_loader_for_file(file: str, data_in_channel_format: bool, hierarchical_data_handled: bool, standardizing_df: pd.DataFrame, standardization_type: str, hierarchical_model: str):
    """reads in a single file and returns a loader and y_transformed 
    """
    joined_df, label = train_model_utils.read_prepared_data_file(file)
    joined_df = joined_df.T

    if hierarchical_data_handled:
        joined_df = data_utils.standardize(joined_df, standardization_type, standardizing_df)

    # create Tensors from the objects
    x = np.stack([joined_df[col].to_numpy() for col in joined_df.columns], 1).astype(np.float64)
    x = torch.tensor(x, dtype=torch.float).reshape(1, -1)
    y = torch.LongTensor([label])

    # When data_in_channel_format is being processed, there might me a reshaping of x needed to bring it to [nr_windows(1),  nr_channels, resampled_timepoints_per_window]
    if data_in_channel_format:
        nr_channels = len(pd.read_csv(file, usecols=[0], dtype=np.float64, header=None))
        x = train_model_utils.reshape_tensor_to_first_identical_dim(x, y, nr_channels)

    Dataset = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))
    del x

    loader = DataLoader(Dataset, batch_size=1, shuffle=False)

    y_transformed = train_model_utils.bring_to_correct_format(y)
    del y

    return loader, y_transformed


def calculate_metrics(results_df: pd.DataFrame, model_folder: str):
    """calculates a df of performance metrics, as well as matthews correlation coeff and cohens kappa
    """
    if "true_label" not in results_df.columns:
        logger.error("Results_df does not contain a column with the true label, further metrics cannot be calculated")
        return pd.DataFrame(), 0.0, 0.0,  "true_label_col_not_found"

    if "prediction" not in results_df.columns:
        logger.error("Results_df does not contain a column with the prediction, further metrics cannot be calculated")
        return pd.DataFrame(), 0.0, 0.0, "prediction_col_not_found"

    metrics_df = calculate_precision_recall_f1(results_df, model_folder)
    logger.info(f"These are the resulting metrics for the the classes:\n\n{metrics_df.head(15)}")

    cohen_k_score = calculate_cohen_kappa_score(results_df)
    interpret_cohen_kappa_score(cohen_k_score)

    mathhews_corr_coeff = calculate_matthews_corr_coeff(results_df)
    interpret_matthews_corrcoef(mathhews_corr_coeff)

    return metrics_df, cohen_k_score, mathhews_corr_coeff, None


def calculate_precision_recall_f1(results_df: pd.DataFrame, model_folder: str):
    """returns a df containing the precision, recall and f1 score for every class, as well as the micro and macro avg per class
    """
    with open(os.path.join(model_folder, "log.json")) as f:
        model_log_file = json.load(f)
    hierarchical_model = model_log_file.get("train_model").get("hierarchical_model")

    differences = list(set(results_df["true_label"]) - set(results_df["prediction"]))
    if len(differences) > 0:
        logger.warning(f"These labels were never predicted: {differences}")

    result_overview = classification_report(results_df["true_label"], results_df["prediction"], output_dict=True)
    overview_df = pd.DataFrame(result_overview)

    # Replace the numerical class names with the meaningful, human-readable ones
    label_names = plot_utils.get_class_names_from_len(len(results_df["true_label"].unique()), hierarchical_model)
    len_diff = len(overview_df.columns) - len(label_names)
    label_names.extend(overview_df.columns[-len_diff:])
    overview_df.columns = label_names
    overview_df = overview_df.T

    return overview_df


def calculate_cohen_kappa_score(df: pd.DataFrame):
    """ returns the cohen-kappa-score, see: https://en.wikipedia.org/wiki/Cohen%27s_kappa
    """
    return cohen_kappa_score(df["true_label"], df["prediction"])


def interpret_cohen_kappa_score(score: float):
    """ logs an interpretation of the cohen_kappa_score
    """
    # the interpretation of these values is based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/

    interpretation = "not better than a random assignment of labels."
    if score > 0.2:
        interpretation = "minimal better than a random assignment of labels."
    if score > 0.4:
        interpretation = "weakly better than a random assignment of labels."
    if score > 0.6:
        interpretation = "moderately better than a random assignment of labels."
    if score > 0.8:
        interpretation = "strongly better than a random assignment of labels."
    if score > 0.9:
        interpretation = "almost perfect"

    logger.info(f"With a cohen-kappa-score of {round(score,2)} this score assigns that the model is {interpretation}")


def calculate_matthews_corr_coeff(df: pd.DataFrame):
    """ returns matthew's correlation coefficient
    """
    return matthews_corrcoef(df["true_label"], df["prediction"])


def interpret_matthews_corrcoef(coeff: float):
    """ logs an interpretation of matthews correlation coefficient (a discrete version of pearson's corr coeff)
    """
    # the interpretation of these values is based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/

    interpretation = "are almost independent of the true label"
    if coeff > 0.2:
        interpretation = "correlate hardly with the true label."
    if coeff > 0.4:
        interpretation = "correlate weakly with the true label."
    if coeff > 0.6:
        interpretation = "correlate moderately with the true label."
    if coeff > 0.8:
        interpretation = "correlate strongly with the true label."
    if coeff > 0.9:
        interpretation = "correlate almost perfectly with the true label"

    logger.info(f"With a matthews correlation coefficient  of {round(coeff,2)} this coeff assigns that the model predictions {interpretation}")


def transform_svm_output_to_valid_tensor(input_ndarray: np.ndarray, classes_to_separate=3):
    """transforms the prediction of the SVM into the same format as the one used by the NN
    """
    prediction_tensor = torch.from_numpy(input_ndarray)

    # The NN return their predictions encoded in a one-hot encoder, therefore the format is adjusted
    return torch.nn.functional.one_hot(prediction_tensor.to(torch.int64), num_classes=classes_to_separate)


def valid_table_hierarchical(table:dict):
    """Checks if all the given parameters in the table are valid for the evaluation of hierarchical models
    """
    for classifier in ["top_level", "lifting", "walking"]:
        if not os.path.exists(table.get(f"{classifier}_folder")):
            logger.error(f"the given folder for the {classifier} classifier does not exist: {table.get(f'{classifier}_folder')}")
            return f"{classifier}_folder_not_found"
    
        log_file_loc = os.path.join(table.get(f"{classifier}_folder"), "log.json")
        if not os.path.exists(log_file_loc):
            logger.error(f"the given folder for the {classifier} classifier does not contain a log file: {table.get(f'{classifier}_folder')}")
            return f"{classifier}_folder_does_not_contain_a_log_file"
        
        # Test if the model within the folder was evaluated on data in the format of features
        with open(log_file_loc) as f:
            log_file: dict = json.load(f)

        if "evaluate_model_second_location" not in log_file.keys():
            logger.error(f"The folder for the given {classifier} classifier does not contain evaluated data")
            return "received _model_not_evaluated_prior"
        if "generate_features" not in log_file.get("evaluate_model_second_location").keys():
            logger.error(f"The given {classifier} classifier was not evaluated on data in the format of features")
            return "received _model_not_evaluated_on_featured_data"

        # test if the predictions.csv exists
        predictions_loc = os.path.join(table.get(f"{classifier}_folder"), "predictions.csv")
        if not os.path.exists(predictions_loc):
            logger.error(f"the given folder for the {classifier} classifier does not contain a predictions file: {table.get(f'{classifier}_folder')}")
            return f"{classifier}_folder_does_not_contain_a_predictions_file"
        predictions_df = pd.read_csv(predictions_loc, usecols=["true_label"])
        if len(predictions_df["true_label"].unique()) > 3:
            logger.error(f"Folder for {classifier} classifier contains more than three true labels")
            return "too_much_true_labels_within_predictions_df"
    
    if table.get("output_date") is not None:
        error = time_utils.valid_time_string(table.get("output_date"))
        if error:
            return error
    
    if len(set([table.get("top_level_folder"), table.get("lifting_folder"), table.get("walking_folder")])) < 3:
        logger.error(f"The given folders are not three distinct folders")
        return "received_identical_folders"

    return None


def aggregate_in_joined_predictions_df(top_level_folder:str, lifting_folder:str, walking_folder:str, aggregated_lifting_lowering: bool) : 
    """returns a df containing all the true labels and predictions as a joined 6 / 7 class problem
    """
    if top_level_folder and lifting_folder and walking_folder:       
        return join_hierarchical_classifiers(top_level_folder, lifting_folder, walking_folder, aggregated_lifting_lowering)
    
    return pd.read_csv(os.path.join(top_level_folder, "predictions.csv"))


def create_confusion_matrix(predictions_df:pd.DataFrame, store_local:bool, saving_folder:str, hierarchical_model:str, verbose:bool):
    """creates a joined confusion matrix from the predictions df
    """
    all_true_labels = torch.from_numpy(predictions_df["true_label"].to_numpy()-1)
    y_pred_joined = torch.from_numpy(predictions_df["prediction"].to_numpy()-1)

    # Create the final confusion matrix of the data
    task = "binary" if len(torch.unique(all_true_labels)) == 2 else "multiclass"
    conf_mat = ConfusionMatrix(num_classes = len(torch.unique(all_true_labels)), task = task)

    confusion_matrix = conf_mat(y_pred_joined, all_true_labels)

    accuracy = round((torch.diagonal(confusion_matrix, 0).sum().item() * 100) / (torch.sum(confusion_matrix)).item(), 2)

    if verbose:
        logger.info(f"This is the Confusion Matrix of the joined Model: (n = {torch.sum(confusion_matrix).item()})\n{confusion_matrix}")
    logger.info(f"The final Accuracy is {accuracy}%")

    plot_utils.plot_confusion_matrix(confusion_matrix, store_local, saving_folder, "test_data", hierarchical_model)

    return confusion_matrix, accuracy, None


def create_samples_per_activity_numerical(aggregated_lifting_lowering: bool, weighting: str, log_file: dict, predictions_df: pd.DataFrame):
    """Returns the number of samples to be used for the final df
    """
    implemented_weighting_types = ['balanced', 'original', 'undersample_majority_classes']
    if weighting not in implemented_weighting_types:
        logger.error(f"Received not yet implemented weighting type {weighting}")
        return {}

    if weighting == 'balanced':
        if aggregated_lifting_lowering:
            return {1: 4781, 2: 4781, 3: 4781, 4: 4781, 5: 4781, 6: 4781}
        return {1: 4781, 2: 4781, 3: 4781, 4: 4781, 5: 4781, 6: 4781, 7: 4781}
    elif weighting == 'undersample_majority_classes':
        if aggregated_lifting_lowering:
            return {1: 1000, 2: 1000, 3: 1000, 4: 874, 5: 734, 6: 1000}
        return {1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 874, 6: 734, 7: 1000}
    

    if 'evaluate_hierarchical' not in log_file.keys():
        return predictions_df['true_label'].value_counts().to_dict()

    # For the original weighting
    value_counts_top = pd.read_csv(os.path.join(log_file['evaluate_hierarchical']['top_level_folder'], "predictions.csv"), usecols=['true_label'])['true_label'].value_counts()
    value_counts_lifting = pd.read_csv(os.path.join(log_file['evaluate_hierarchical']['lifting_folder'], "predictions.csv"), usecols=['true_label'])['true_label'].value_counts()
    value_counts_walking = pd.read_csv(os.path.join(log_file['evaluate_hierarchical']['walking_folder'], "predictions.csv"), usecols=['true_label'])['true_label'].value_counts()

    if len(value_counts_lifting) == 2:
        samples_dict = {1: value_counts_lifting[1]}
        samples_dict[2] = value_counts_lifting[2]
    else:
        samples_dict = {1: value_counts_lifting[1]} 
        samples_dict[2] = value_counts_lifting[2]
        samples_dict[3] = value_counts_lifting[3]
    samples_dict[len(samples_dict) + 1] = value_counts_walking[1]
    samples_dict[len(samples_dict) + 1] = value_counts_walking[2]
    samples_dict[len(samples_dict) + 1] = value_counts_walking[3]
    samples_dict[len(samples_dict) + 1] =  value_counts_top[3]

    return samples_dict


def detect_as_multi_class_label(top_level_label: int, subclassifier_true_label: int, aggregated_lifting_lowering: bool):
    """aggregates the labels in a hierarchical classification into a common range, i.e. three analyzers classifying from [1; 3] are aggregated into a common [1: 7]
    """
    resting_class = 6 if aggregated_lifting_lowering else 7 
    nr_lifting_classes = 2 if aggregated_lifting_lowering else 3
    if top_level_label == 3:
        return resting_class
    
    if top_level_label == 1:
        return subclassifier_true_label
    
    return subclassifier_true_label + nr_lifting_classes


def get_top_level_and_sublevel_value(top_df: pd.DataFrame, lift_df: pd.DataFrame, walk_df: pd.DataFrame, colname: str, window_uuid: str, label_based_on_filename: int):
    """Gets the top-level and sub-level value (of a column) based on the window_uuid
    """
    top_level_value = int(top_df[colname][top_df['window_uuid'] == window_uuid])   
    top_level_value_true = int(top_df["true_label"][top_df['window_uuid'] == window_uuid])

    if top_level_value == 3 or top_level_value_true == 3:
        sub_level_value = 0
    elif  top_level_value == 1:
        relevant_subdf = lift_df[colname][lift_df['window_uuid'] == window_uuid]
        if relevant_subdf.empty and label_based_on_filename ==0:
            sub_level_value = 0
        else:       
            sub_level_value = int(relevant_subdf)
    else:
        relevant_subdf = walk_df[colname][walk_df['window_uuid'] == window_uuid]
        if relevant_subdf.empty and label_based_on_filename ==0:
            sub_level_value = 0
        else:       
            sub_level_value = int(relevant_subdf)
    
    return top_level_value, sub_level_value


def join_hierarchical_classifiers(top_level_folder: str, lifting_folder: str, walking_folder: str, aggregated_lifting_lowering: bool):
    """Joins the classifications made by three hierarchical models into a joined df
    """
    predictions_top_df = pd.read_csv(os.path.join(top_level_folder, "predictions.csv"), usecols=["file", "true_label", "prediction"])
    predictions_lifting_df = pd.read_csv(os.path.join(lifting_folder, "predictions.csv"), usecols=["file", "true_label", "prediction"])
    predictions_walking_df = pd.read_csv(os.path.join(walking_folder, "predictions.csv"), usecols=["file", "true_label", "prediction"])

    predictions_top_df['window_uuid'] = [file.split("/")[-1].split("_prepared_")[1] for file in predictions_top_df['file']]
    predictions_lifting_df['window_uuid'] = [file.split("/")[-1].split("_prepared_")[1] for file in predictions_lifting_df['file']]
    predictions_walking_df['window_uuid'] = [file.split("/")[-1].split("_prepared_")[1] for file in predictions_walking_df['file']]
    if not set(predictions_lifting_df['window_uuid']).issubset(set(predictions_top_df['window_uuid'])):
        logger.error(f"Some window_uuid existing in the predictions_lifting_df do not exist in the predictions_top_df")
        return pd.DataFrame()
    if not set(predictions_walking_df['window_uuid']).issubset(set(predictions_top_df['window_uuid'])):
        logger.error(f"Some window_uuid existing in the predictions_walking_df do not exist in the predictions_top_df")
        return pd.DataFrame()

    true_labels = [0] * len(predictions_top_df)
    predictions = [0] * len(predictions_top_df)
    window_uuids = [0] * len(predictions_top_df)

    for index, window_uuid in enumerate(predictions_top_df['window_uuid']):
        label_based_on_filename = predictions_top_df["file"].iloc[index].split("/")[-1].split("_prepared")[0]
        top_level_true_label, sub_level_true_label = get_top_level_and_sublevel_value(predictions_top_df, predictions_lifting_df, predictions_walking_df, "true_label", window_uuid, label_based_on_filename)
        top_level_prediction, sub_level_prediction = get_top_level_and_sublevel_value(predictions_top_df, predictions_lifting_df, predictions_walking_df, "prediction", window_uuid, 0)

        true_labels[index] = detect_as_multi_class_label(top_level_true_label, sub_level_true_label, aggregated_lifting_lowering)
        predictions[index] = detect_as_multi_class_label(top_level_prediction, sub_level_prediction, aggregated_lifting_lowering)
        window_uuids[index] = window_uuid

    joined_predictions_df = pd.DataFrame({"window_uuid": window_uuids, "true_label": true_labels, "prediction": predictions, "file": predictions_top_df['file']})

    return joined_predictions_df


def reweight_joined_predictions(predictions_df: pd.DataFrame, samples_per_activities: dict):
    """ Adjusts the number of samples per activity based on the samples_per_activities-dict
    """
    random.seed(42)
    reweighted_df = pd.DataFrame()
    for label in samples_per_activities.keys():
        needed_samples = samples_per_activities[label]
        available_indexes = predictions_df[predictions_df['true_label'] == label].index
        indexes_to_keep = []

        # Ensure that no window is discarded completely when additional samples should be added
        if needed_samples >= len(available_indexes):
            indexes_to_keep.extend(available_indexes.to_list())
            needed_samples -= len(available_indexes)

        indexes_to_keep.extend([available_indexes[random.randint(0, len(available_indexes) -1)] for _ in range(needed_samples)]) 

        reweighted_df = pd.concat([reweighted_df, predictions_df.loc[indexes_to_keep]])

    return reweighted_df


def reweight_classes(top_level_folder: str, lifting_folder: str, walking_folder: str, aggregated_lifting_lowering: bool, weighting: str, log_file: dict):
    """ reweights the classified activities in the predictions.csvs in the given folder, aggregates hierarchical predictions if needed
    """  
    predictions_df = aggregate_in_joined_predictions_df(top_level_folder, lifting_folder, walking_folder, aggregated_lifting_lowering)
    
    samples_per_activities = create_samples_per_activity_numerical(aggregated_lifting_lowering, weighting, log_file, predictions_df)
    
    # Create a df containing concrete observations of the respective activities in a balanced way, i.e. each activity is observed as specified in the weighting    
    predictions_df = reweight_joined_predictions(predictions_df, samples_per_activities)
    return predictions_df


def add_probability_scores(predictions_df: pd.DataFrame):
    """Calculates the probability for the classification of all samples, needed for ROC
    """
    raw_prediction_cols = [col for col in predictions_df.columns if "Raw_prediction_label" in col]
    if len(raw_prediction_cols) < 2:
        logger.error("Cannot calculate probabilities for prediction, when less than 2 cols with the raw predictions are given")
        return predictions_df

    probability_correct = []
    for index in range(len(predictions_df)):
        true_col =  f"Raw_prediction_label_{predictions_df['true_label'].iloc[index]}"
        exp_true_val = math.exp(predictions_df[true_col].iloc[index])
        other_cols = [col for col in raw_prediction_cols if col != true_col]
        highest_non_true_prediction = max(predictions_df[other_cols].iloc[index])
        exp_highest_non_true_val = math.exp(highest_non_true_prediction)
        probability_correct.append(exp_true_val / (exp_true_val + exp_highest_non_true_val))  

    probability_incorrect = [1 - value for value in probability_correct]

    predictions_df['probability_correct'] = probability_correct
    predictions_df['probability_incorrect'] = probability_incorrect
    return predictions_df


def valid_model_comparison_table(table: dict):
    """Tests if the given parameters can be used to perform a statistical comparison of trained and evaluated models
    """
    if not os.path.exists(table.get("model_folder_1")):
        logger.error(f"Given folder {table.get('model_folder_1')} does not exist on local machine")
        return "model_folder_not_found"
    if not os.path.exists(table.get("model_folder_2")):
        logger.error(f"Given folder {table.get('model_folder_2')} does not exist on local machine")
        return "model_folder_not_found"
    if table.get('model_folder_2') == table.get('model_folder_1'):
        logger.error(f"Received same folder for models to compare")
        return "received-identical_models"
    if "evaluate_model" not in table.get("model_folder_2") or "evaluate_model" not in table.get("model_folder_1"):
        logger.error(f"Tried comparing models that were not yet evaluated on test data")
        return "received_not_evaluated_models"
    
    if not os.path.isfile(os.path.join(table.get("model_folder_1"), "predictions.csv")):
        logger.error(f"Predictions not found in folder {table.get('model_folder_1')}")
        return "predictions_not_found"

    pred_df = pd.read_csv(os.path.join(table.get("model_folder_1"), "predictions.csv"), nrows=2)
    needed_columns = ["true_label", "prediction"]
    if not set(needed_columns).issubset(pred_df.columns):
        logger.error(f"Not all needed columns {needed_columns} exist in {os.path.join(table.get('model_folder_1'), 'predictions.csv')}")
        return "needed_cols_not_found"  

    if not os.path.isfile(os.path.join(table.get("model_folder_2"), "predictions.csv")):
        logger.error(f"Predictions not found in folder {table.get('model_folder_2')}")
        return "predictions_not_found"
    pred_df = pd.read_csv(os.path.join(table.get("model_folder_2"), "predictions.csv"), nrows=2)
    if not set(needed_columns).issubset(pred_df.columns):
        logger.error(f"Not all needed columns {needed_columns} exist in {os.path.join(table.get('model_folder_2'), 'predictions.csv')}")
        return "needed_cols_not_found"  
    
    pred_df_model_1 = pd.read_csv(os.path.join(table.get("model_folder_1"), "predictions.csv"), usecols=["true_label"])
    pred_df_model_2 = pd.read_csv(os.path.join(table.get("model_folder_2"), "predictions.csv"), usecols=["true_label"])
    if set(pred_df_model_1['true_label'].unique().tolist()) != set(pred_df_model_2['true_label'].unique().tolist()):
        logger.error(f"Found differing true labels in two models to compare")
        return "differing_labels_found_for_model"

    if table.get("alpha") < 0.0:
        logger.error("Received alpha value smaller 0. This is not meaningful for statistical tests")
        return "received_too_small_alpha_value"
    if table.get("alpha") > 1.0:
        logger.error("Received alpha value bigger 1. This is not meaningful for statistical tests")
        return "received_too_big_alpha_value"
    if table.get("alpha") > 0.05:
        logger.warning(f"Choose unusually high value of alpha {table.get('alpha')}. High probability of arriving at significant results based on chance")
    
    return None    


def ensure_window_uuid_is_present(df: pd.DataFrame):
    """creates the window_uuid from the file col if needed
    """
    if "window_uuid" in df.columns:
        return df
    if "file" not in df.columns:
        logger.error(f"Neither col window_uuid nor file found in given predictions.csv. No statistical tests possible")
        return pd.DataFrame()
    
    raw_uuids = [file.split("/")[-1] for file in df["file"]]
    if "_prepared_" in raw_uuids[0]:
        raw_uuids = [elem.split("_prepared_")[-1] for elem in raw_uuids]
    df["window_uuid"] = raw_uuids

    return df


def perform_statistical_model_comparison(predictions_model_1: pd.DataFrame, predictions_model_2: pd.DataFrame, other_model_folders: list, alpha: float) -> dict:
    """performs all needed statistical tests to compare models
    """
    predictions_model_1 = ensure_window_uuid_is_present(predictions_model_1)
    predictions_model_2 = ensure_window_uuid_is_present(predictions_model_2)
    if predictions_model_1.empty or predictions_model_2.empty:
        return {}

    needed_columns = ["true_label", "prediction", "file"]
    if not set(needed_columns).issubset(set(predictions_model_1.columns)):
        logger.error(f"Not all needed columns {needed_columns} found in the cols of model 1")
        return {}
    if not set(needed_columns).issubset(set(predictions_model_2.columns)):
        logger.error(f"Not all needed columns {needed_columns} found in the cols of model 2")
        return {}
    
    # Perform all statistical test
    statistical_results = {}
    statistical_results = perform_mc_nemar_test(predictions_model_1, predictions_model_2, statistical_results, alpha)
    statistical_results = perform_wilcoxon_signed_rank_test(predictions_model_1, predictions_model_2, statistical_results, alpha)
    statistical_results = perform_friedman_test(predictions_model_1, predictions_model_2, other_model_folders, statistical_results, alpha)

    return statistical_results


def misclassified(df: pd.Series):
    return df.nunique() > 1
    

def create_McNemar_contingency_table(predictions_1: pd.DataFrame, predictions_2: pd.DataFrame) -> list:
    """Creates the contingency table of misclassifications of both classifiers as a [2, 2]list    
    """
    # Check if the given Dfs are correct
    needed_cols = ["true_label", "prediction", "window_uuid"]
    if not (set(needed_cols).issubset(set(predictions_1.columns)) and set(needed_cols).issubset(set(predictions_2.columns))):
        logger.error("The given Dataframes do not contain all needed columns. Performing McNemars test not possible")
        return []
    
    if set(predictions_1['window_uuid']) != set(predictions_2['window_uuid']):
        max_set = predictions_1['window_uuid'].unique() if predictions_1['window_uuid'].nunique() > predictions_2['window_uuid'].nunique() else predictions_1['window_uuid'].unique()
        predictions_1 = predictions_1[predictions_1["window_uuid"].isin(max_set)]
        predictions_2 = predictions_2[predictions_2["window_uuid"].isin(max_set)]
        predictions_1 = predictions_1[predictions_1["window_uuid"].isin(predictions_2["window_uuid"].unique())]

    samples_misclassified_by_both = 0
    samples_misclassified_model_1_correctly_class_model_2 = 0
    samples_misclassified_model_2_correctly_class_model_1 = 0
    samples_correctly_classified_both = 0

    predictions_1["correct_classified"] = predictions_1['true_label'] == predictions_1['prediction']
    predictions_2["correct_classified"] = predictions_2['true_label'] == predictions_2['prediction']

    # Step 1 Get the number of samples misclassified by both classifiers
    for index, window_uuid in enumerate(predictions_1['window_uuid']):
        if predictions_1["correct_classified"].iloc[index] and predictions_2[predictions_2['window_uuid'] == window_uuid]['correct_classified'].iloc[0]:
            samples_correctly_classified_both += 1
            continue
        elif not predictions_1["correct_classified"].iloc[index] and not predictions_2[predictions_2['window_uuid'] == window_uuid]['correct_classified'].iloc[0]:
            samples_misclassified_by_both += 1
            continue
        elif predictions_1["correct_classified"].iloc[index] and not predictions_2[predictions_2['window_uuid'] == window_uuid]['correct_classified'].iloc[0]:
            samples_misclassified_model_2_correctly_class_model_1 +=1
            continue
        elif not predictions_1["correct_classified"].iloc[index] and predictions_2[predictions_2['window_uuid'] == window_uuid]['correct_classified'].iloc[0]:
            samples_misclassified_model_1_correctly_class_model_2 += 1
            continue
        logger.error("Counting logic contains error, all cases should be caught by above statements")
        return []

    return [[samples_misclassified_by_both, samples_misclassified_model_1_correctly_class_model_2], [samples_misclassified_model_2_correctly_class_model_1, samples_correctly_classified_both]]


def interpret_mc_nemars_test(mc_nemar_p_value: float, alpha: float):
    """logs an interpretation of the mc_nemars_test
    """
    if mc_nemar_p_value > alpha:
        logger.info(f"Based on a p value of {round(mc_nemar_p_value, 3)}, the Null-Hypotheses could not be rejected, i.e. there is no significant difference in disagreement of both classifiers with an alpha of {alpha}")
    else:
        logger.info(f"Based on a p value of {round(mc_nemar_p_value, 3)}, the Null-Hypotheses could be rejected, i.e. there is significant difference in disagreement of both classifiers with an alpha of {alpha}")


def perform_mc_nemar_test(predictions_1: pd.DataFrame, predictions_2: pd.DataFrame, statical_tests: dict, alpha: float) -> dict:
    """Perform McNemars test to test if two classifiers have different error rates when evaluated on a test set
    """ 
    if "McNemars_test" in statical_tests.keys():
        logger.warning("Tried to perform McNemars_test when it already has been performed on this data")
        return statical_tests

    # Ensure that every window_uuid is counted only once for the statistical test
    predictions_1_internal = predictions_1.drop_duplicates(inplace=False).copy(True)
    predictions_2_internal = predictions_2.drop_duplicates(inplace=False).copy(True)

    # Create the correct contingency table
    cont_table = create_McNemar_contingency_table(predictions_1_internal, predictions_2_internal)
    if not cont_table:
        return statical_tests

    # perform the test using statsmodels
    result = mcnemar(cont_table, exact=False, correction=True)
    mc_nemar_test_statistic = result.statistic
    mc_nemar_p_value = result.pvalue

    interpret_mc_nemars_test(mc_nemar_p_value, alpha)

    statical_tests["McNemars_test"] = {"mc_nemar_test_statistic": mc_nemar_test_statistic,
                                       "mc_nemar_p_value": mc_nemar_p_value}

    return statical_tests


def calculate_accuracy(df: pd.DataFrame):
    if not {"true_label", "prediction"}.issubset(df.columns):
        logger.error(f"The needed columns true_label and prediction are missing from the columns of the dataframe")
        return 0
    return sum(df["true_label"] == df["prediction"]) / len(df)


def calculate_f1_score(df: pd.DataFrame):
    if not {"true_label", "prediction"}.issubset(df.columns):
        logger.error(f"The needed columns true_label and prediction are missing from the columns of the dataframe")
        return 0
    return fbeta_score(y_true=df["true_label"], y_pred = df["prediction"],  beta=1, average="macro")


def tester_from_filename(filename: str) -> int:
    return int(filename.split("/")[-1].split("tester")[1].split("(")[0]) 


def perform_wilocoxon_test(differences: list):
    """performs both the wilcoxon test and a test confirming that the median of differences can be assumed to be positive
    """
    res = wilcoxon(differences)
    res_greater = wilcoxon(differences, alternative='greater')
    return res, res_greater


def interpret_wilcoxon_test(res, res_greater, alpha: float, statistic: str):
    """prints an interpretation of the results of the Wilcoxon signed-rank test to the terminal
    """
    if res.pvalue < alpha:
        logger.info(f"Based on a p_value of {res.pvalue} and an alpha value of {alpha} the null hypothesis that mean of differences of the statistic {statistic} is 0 can be rejected")
        if res_greater.pvalue < alpha:
            logger.info(f"Based on a p_value of {res_greater.pvalue} and an alpha value of {alpha} the null hypothesis that median of differences of statistic {statistic} is negative can be rejected")
        return
    logger.info(f"Based on a p_value of {res.pvalue} and an alpha value of {alpha} the null hypothesis that mean of differences of the statistic {statistic} is 0 can not be rejected")


def perform_wilcoxon_signed_rank_test(predictions_1: pd.DataFrame, predictions_2: pd.DataFrame, statical_tests: dict, alpha: float) -> dict:
    """perform the wilcoxon signed-rank test by treating each subject within the validation data as an own independent dataset and comparing the ranks of accuracies of the classifiers
    """
    accuracies_model_1 = []
    accuracies_model_2 = []
    f1_model_1 = []
    f1_model_2 = []
    mccs_model_1 = []
    mccs_model_2 = []

    # extract the tester from the respective filename for easier indexing
    predictions_1["tester"] = [tester_from_filename(filename) for filename in predictions_1["file"]]
    predictions_2["tester"] = [tester_from_filename(filename) for filename in predictions_2["file"]]

    if set(predictions_1["tester"].unique().tolist()) != set(predictions_2["tester"].unique().tolist()):
        logger.error(f"Different tester found within the two given dataframes")
        return statical_tests

    for tester_nr in predictions_1["tester"].unique():
        # Get the relevant part of the df
        predictions_1_sub_df = predictions_1[predictions_1["tester"] == tester_nr].copy(True)
        predictions_2_sub_df = predictions_2[predictions_2["tester"] == tester_nr].copy(True)

        accuracies_model_1.append(calculate_accuracy(predictions_1_sub_df)) 
        accuracies_model_2.append(calculate_accuracy(predictions_2_sub_df)) 

        f1_model_1.append(calculate_f1_score(predictions_1_sub_df)) 
        f1_model_2.append(calculate_f1_score(predictions_2_sub_df)) 

        mccs_model_1.append(calculate_matthews_corr_coeff(predictions_1_sub_df)) 
        mccs_model_2.append(calculate_matthews_corr_coeff(predictions_2_sub_df)) 

    differences_acc = [acc - accuracies_model_2[number] for number, acc in enumerate(accuracies_model_1)]
    differences_f1 = [f1 - f1_model_2[number] for number, f1 in enumerate(f1_model_1)]
    differences_mcc = [mcc - mccs_model_2[number] for number, mcc in enumerate(mccs_model_1)]

    acc_res, acc_res_greater = perform_wilocoxon_test(differences_acc)
    interpret_wilcoxon_test(acc_res, acc_res_greater, alpha, "Accuracy")

    f1_res, f1_res_greater = perform_wilocoxon_test(differences_f1)
    interpret_wilcoxon_test(f1_res, f1_res_greater, alpha, "F1-Score")

    mcc_res, mcc_greater_res = perform_wilocoxon_test(differences_mcc)
    interpret_wilcoxon_test(mcc_res, mcc_greater_res, alpha, "MCC")


    statical_tests["Wilcoxon_test"] =  {"Accuracy": {"Wilcoxon_test_statistic": acc_res.statistic,
                                                     "Wilcoxon_p_value": acc_res.pvalue,
                                                     "Wilcoxon_test_greater_statistic": acc_res_greater.statistic,
                                                     "Wilcoxon_greater_p_value": acc_res_greater.pvalue},
                                        "F1": {"Wilcoxon_test_statistic": f1_res.statistic,
                                               "Wilcoxon_p_value": f1_res.pvalue,
                                               "Wilcoxon_test_greater_statistic": f1_res_greater.statistic,
                                               "Wilcoxon_greater_p_value": f1_res_greater.pvalue},
                                        "MCC": {"Wilcoxon_test_statistic": mcc_res.statistic,
                                                "Wilcoxon_p_value": mcc_res.pvalue,
                                                "Wilcoxon_test_greater_statistic": mcc_greater_res.statistic,
                                                "Wilcoxon_greater_p_value": mcc_greater_res.pvalue}                                                  
                                       }
    return statical_tests


def interpret_friedman_test(res, alpha: float, statistic: str):
    if res.pvalue < alpha:
        logger.info(f"Based on a p_value of {res.pvalue} and an alpha value of {alpha} the null hypothesis that the means for the statistic {statistic} are equal for all datasets can be rejected")
        return
    logger.info(f"Based on a p_value of {res.pvalue} and an alpha value of {alpha} the null hypothesis that the means for the statistic {statistic} are equal for all datasets is accepted")


def perform_friedman_test(predictions_1: pd.DataFrame, predictions_2: pd.DataFrame, other_model_folders: list, statical_tests: dict, alpha: float) -> dict:
    """ performs the friedman test to test if there is a difference between the means of various metrics for the given models
    """
    if not other_model_folders:
        return statical_tests

    other_predictions = [pd.read_csv(os.path.join(folder, "predictions.csv")) for folder in other_model_folders]
    all_dfs = [predictions_1, predictions_2] + other_predictions 

    all_dfs_with_tester = []
    for df in all_dfs:
        df['tester'] = [tester_from_filename(filename) for filename in df["file"]]
        all_dfs_with_tester.append(df)

    unique_testers = [df["tester"].nunique() for df in all_dfs_with_tester]
    if len(set(unique_testers)) > 1:
        logger.error(f"Differing number of testers within the datasets received. No friedman possible")
        return statical_tests
    if set(pd.concat(all_dfs_with_tester, axis=0)["tester"]) != set(all_dfs_with_tester[0]["tester"]):
        logger.error(f"Different testers within the datasets received. No friedman possible")
        return statical_tests

    all_metrics = {}

    for df_index, df in enumerate(all_dfs_with_tester):
        all_metrics[f"Model_nr_{df_index}"] = {"Accuracy": [],
                                               "F1-Score": [],
                                               "MCC": []}
        for tester_nr in predictions_1["tester"].unique():
            sub_df = df[df["tester"] == tester_nr].copy(True)
            all_metrics[f"Model_nr_{df_index}"]["Accuracy"].append(calculate_accuracy(sub_df))
            all_metrics[f"Model_nr_{df_index}"]["F1-Score"].append(calculate_f1_score(sub_df))
            all_metrics[f"Model_nr_{df_index}"]["MCC"].append(calculate_matthews_corr_coeff(sub_df))

    all_accs = [all_metrics[key]["Accuracy"] for key in all_metrics.keys()]
    res_accs = friedmanchisquare(*all_accs)
    interpret_friedman_test(res_accs, alpha, "Accuracy")

    all_f1s = [all_metrics[key]["F1-Score"] for key in all_metrics.keys()]
    res_f1 = friedmanchisquare(*all_f1s)
    interpret_friedman_test(res_f1, alpha, "F1-Score")

    all_mccs = [all_metrics[key]["MCC"] for key in all_metrics.keys()]
    res_mccs = friedmanchisquare(*all_mccs)
    interpret_friedman_test(res_mccs, alpha, "MCC")

    statical_tests["Friedman_test"] =  {"Accuracy": {"Friedman_test_statistic": res_accs.statistic,
                                                     "Friedman_test_p_value": res_accs.pvalue},
                                        "F1": {"Friedman_test_statistic": res_f1.statistic,
                                               "Friedman_test_p_value": res_f1.pvalue},
                                        "MCC": {"Friedman_test_statistic": res_mccs.statistic,
                                                "Friedman_test_p_value": res_mccs.pvalue}                                                  
                                       }
    return statical_tests

def aggregate_raw_predictions(df:pd.DataFrame):
    """Aggregates the raw predictions of lifting and lowering
    """
    new_levels = [max([df["Raw_prediction_label_1"].iloc[index], df["Raw_prediction_label_2"].iloc[index]]) for index in range(len(df))]
    df["Raw_prediction_label_1"] = new_levels
    df["Raw_prediction_label_2"] = df["Raw_prediction_label_3"]
    df["Raw_prediction_label_3"] = df["Raw_prediction_label_4"]
    df["Raw_prediction_label_4"] = df["Raw_prediction_label_5"]
    df["Raw_prediction_label_5"] = df["Raw_prediction_label_6"]
    df["Raw_prediction_label_6"] = df["Raw_prediction_label_7"]
    df = df.drop("Raw_prediction_label_7", axis=1)
    return df