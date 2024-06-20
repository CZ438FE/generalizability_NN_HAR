import os
import sys
import math
import shutil
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import evaluation_utils, file_utils, train_model_utils
from config.test_config import test_config

def test_valid_table():
    root = test_config['TEST_ROOT']
    folder =os.path.join(root, "evaluation_utils_test", "test_valid_table")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    test_data_folder = os.path.join(folder, "test_data")
    model_folder = os.path.join(folder, "model_data")

    # test if the test folder does not exist
    got = evaluation_utils.valid_table({"test_data": test_data_folder})
    assert got == "invalid_test_data_given"

    os.makedirs(test_data_folder)

    # test if dryrun has an invalid datatype
    got = evaluation_utils.valid_table({"test_data": test_data_folder, "dryrun": 14, "data_path": folder})
    assert got == "got_dryrun_of_nonbool_dtype"

    # Test if the model folder does not exist
    got = evaluation_utils.valid_table({"test_data": test_data_folder, "dryrun": True, "model_folder": model_folder, "data_path": folder})
    assert got == "invalid_model_folder_given"

    # Test if the model folder does not contain a log.json
    os.makedirs(model_folder)
    got = evaluation_utils.valid_table({"test_data": test_data_folder, "dryrun": True, "model_folder": model_folder, "data_path": folder})
    assert got == 'no_data_in_dir'
    pd.DataFrame({"some_col": [1, 2]}).to_csv(os.path.join(model_folder, "log.json"))
    pd.DataFrame({"some_col": [1, 2]}).to_csv(os.path.join(model_folder, "model.pt"))

    # test if the test data folder does not contain a log file
    got = evaluation_utils.valid_table({"test_data": test_data_folder, "dryrun": True, "model_folder": model_folder, "data_path": folder})
    assert got == 'no_data_in_dir'
    pd.DataFrame({"some_col": [1, 2]}).to_csv(os.path.join(test_data_folder, "log.json"))
    pd.DataFrame({"some_col": [1, 2]}).to_csv(os.path.join(test_data_folder, "file.csv"))
    pd.DataFrame({"some_col": [1, 2]}).to_csv(os.path.join(model_folder, "file.csv"))

    # test if both given folders are identical
    got = evaluation_utils.valid_table({"test_data": model_folder, "dryrun": True, "model_folder": model_folder, "data_path": folder})
    assert got == "received_identical_model_and_data_folder"

    # Test if everything is valid
    got = evaluation_utils.valid_table({"test_data": test_data_folder, "dryrun": True, "model_folder": model_folder, "plot": False, "data_path": folder})
    assert got is None


def test_valid_preprocessing_of_data_for_model():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "evaluation_utils_test", "test_valid_preprocessing_of_data_for_model")
    if os.path.exists(folder):
        shutil.rmtree(folder)

    # test if the model folder does not exist
    got = evaluation_utils.valid_preprocessing_of_data_for_model(folder, "")
    assert got == "given_model_folder_does_not_exist"

    model_folder = os.path.join(folder, "model")
    os.makedirs(model_folder)

    # Test if the log file is missing
    got = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, "")
    assert got == "given_model_folder_does_not_contain_log_file"
    error = file_utils.save_json({}, os.path.join(model_folder, "log.json"))
    assert error is None

    # test if the test folder does not exist
    test_data_folder = os.path.join(folder, "test_data")
    got = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, test_data_folder)
    assert got ==  "given_test_data_folder_does_not_exist"
    os.makedirs(test_data_folder)

    # Test if the log file is missing
    got = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, test_data_folder)
    assert got == "given_test_data_folder_does_not_contain_log_file"
    error = file_utils.save_json({}, os.path.join(model_folder, "log.json"))
    assert error is None

    # Test if the preprocessing steps differ
    error = file_utils.save_json({"create_windows": True, "balancing_over": True,
                                    "train_model": True}, os.path.join(model_folder, "log.json"))
    assert error is None
    error = file_utils.save_json({"create_windows": {"window_length": 1000}, "generate_features": True,
                                    "balancing_over": True}, os.path.join(test_data_folder, "log.json"))
    assert error is None
    got = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, test_data_folder)
    assert got == "preprocessing_steps_differ"

    # Test if the window_lengths differ
    error = file_utils.save_json({"create_windows": {"window_length": 400}, "generate_features": True,
                                    "balancing_over": True}, os.path.join(model_folder, "log.json"))
    assert error is None
    got = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, test_data_folder)
    assert got == "differing_window_lengths_found"

    # test if the flatten values differ
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True},
                                    "generate_features": True, "balancing_over": True}, os.path.join(test_data_folder, "log.json"))
    assert error is None
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": False},
                                    "generate_features": True, "balancing_over": True}, os.path.join(model_folder, "log.json"))
    assert error is None
    got = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, test_data_folder)
    assert got == "differing_flatten_values_found"

    # test if the filling method differs
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"},
                                    "generate_features": True, "balancing_over": True}, os.path.join(test_data_folder, "log.json"))
    assert error is None
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something_else"},
                                    "generate_features": True, "balancing_over": True}, os.path.join(model_folder, "log.json"))
    assert error is None
    got = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, test_data_folder)
    assert got == "differing_filling_methods_values_found"

    # test if the classification granularity differs
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"},
                                    "generate_features": True, "balancing_over": {"granularity": "top"}}, os.path.join(test_data_folder, "log.json"))
    assert error is None
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"},
                                    "generate_features": True, "balancing_over": {"granularity": "mid"}}, os.path.join(model_folder, "log.json"))
    assert error is None
    got = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, test_data_folder)
    assert got == "differing_balancing_granularities_found"

    # Test when one of the sources contains data for CNNs
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"}, "generate_features": True, "balancing_over": {
                                    "granularity": "top"}, "prepare_dataset": {"convolutional": True}}, os.path.join(test_data_folder, "log.json"))
    assert error is None
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"}, "generate_features": True, "balancing_over": {
                                    "granularity": "top"}, "prepare_dataset": {"convolutional": False}}, os.path.join(model_folder, "log.json"))
    assert error is None

    # Test when everything is valid
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"}, "generate_features": True, "balancing_over": {"granularity": "top"}, "prepare_dataset": {
                                    "convolutional": True}, "train_model": {"training_data": "some_data", "validation_data": "some_other_data", }}, os.path.join(test_data_folder, "log.json"))
    assert error is None
    error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"}, "generate_features": True, "balancing_over": {"granularity": "top"}, "prepare_dataset": {
                                    "convolutional": True}, "train_model": {"training_data": "some_data", "validation_data": "some_other_data", }}, os.path.join(model_folder, "log.json"))
    assert error is None
    got = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, test_data_folder)
    assert got is None


def test_detect_network_type():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "evaluation_utils_test", "test_detect_network_type")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # test correct detection for SVMs
    svm_folder = os.path.join(folder, "SVM")
    os.makedirs(svm_folder)

    # Test when a FFN Model lies within a folder for the SVMs
    error = file_utils.save_json({"train_model": {"type": "FFNN"}}, os.path.join(svm_folder, "log.json"))
    assert error is None
    got_type, got_error = evaluation_utils.detect_network_type(svm_folder)
    assert got_error == "found_non_svm_model_in_svm_folder"

    # Test when an invalid hierarchical model was given
    error = file_utils.save_json({"train_model": {
                                    "type": "SVM", "hierarchical_model": "something_else"}}, os.path.join(svm_folder, "log.json"))
    assert error is None
    got_type, got_error = evaluation_utils.detect_network_type(svm_folder)
    assert got_error == "found_svm_for_separation_of_unknown_classes"

    # Test if everything works as intended for SVM
    error = file_utils.save_json({"train_model": {
                                    "type": "SVM", "hierarchical_model": "walking"}}, os.path.join(svm_folder, "log.json"))
    assert error is None
    got_type, got_error = evaluation_utils.detect_network_type(svm_folder)
    assert got_error is None
    assert got_type == "SVM"

    # No balancing for NN
    error = file_utils.save_json({"create_windows": {
                                    "flatten": True}, "train_model": True}, os.path.join(folder, "log.json"))
    assert error is None
    got_type, got_error = evaluation_utils.detect_network_type(folder)
    assert got_error == "no_oversampling_detected"

    # Not the correct format
    error = file_utils.save_json({"create_windows": True, "balancing_over": True,
                                    "train_model": True, "balancing_over": True}, os.path.join(folder, "log.json"))
    assert error is None
    got_type, got_error = evaluation_utils.detect_network_type(folder)
    assert got_error == "no_prepare_dataset_detected"

    # FFNN
    error = file_utils.save_json({"create_windows": True, "balancing_over": True, "train_model": True,
                                    "prepare_dataset": True, "generate_features": True, "reduce_data": True}, os.path.join(folder, "log.json"))
    assert error is None
    got_type, got_error = evaluation_utils.detect_network_type(folder)
    assert got_error is None
    assert got_type == "FFNN"

    error = file_utils.save_json({"create_windows": {"flatten": True}, "balancing_over": True, "train_model": True, "prepare_dataset": {
                                    "convolutional": False}}, os.path.join(folder, "log.json"))
    assert error is None
    got_type, got_error = evaluation_utils.detect_network_type(folder)
    assert got_error is None
    assert got_type == "FFNN"

    # CNN
    error = file_utils.save_json({"create_windows": {"flatten": True}, "balancing_over": True, "train_model": {
                                    "type": "CNN"}, "prepare_dataset": {"grid": True}}, os.path.join(folder, "log.json"))
    assert error is None
    got_type, got_error = evaluation_utils.detect_network_type(folder)
    assert got_error is None
    assert got_type == "CNN"

    # cannot be detected
    error = file_utils.save_json({"create_windows": {"flatten": False}, "balancing_over": True, "train_model": {
                                    "type": "RNN"}, "prepare_dataset": {"grid": False}}, os.path.join(folder, "log.json"))
    assert error is None
    got_type, got_error = evaluation_utils.detect_network_type(folder)
    assert got_type is None
    assert got_error == "detecting_model_type_ failed"


def test_load_trained_model():
    # test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "evaluation_utils_test", "test_load_trained_model")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"train_model": {"seed": 42, "batch_normalization": True, "n_features": 15, "out_szs": 4, "dropout_rate": 0.1, "layer_structure": "80f30"},
                "generate_features": True, "reduce_data": True, "balancing_over": True, "prepare_dataset": True}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    # Save a model in the folder be loaded
    layers, error = train_model_utils.build_layers_from_string(log_file.get("train_model").get("layer_structure"), "FFNN")
    assert error is None
    Model = train_model_utils.build_nn_model("FFNN", log_file.get("train_model").get("batch_normalization"), log_file.get("train_model").get("seed"), log_file)
    model = Model(log_file.get("train_model").get("n_features"), log_file.get("train_model").get("out_szs"), layers, log_file.get("train_model").get("dropout_rate"), log_file.get("train_model").get("batch_normalization"))

    error = train_model_utils.save_model(model, folder)
    assert error is None

    got_model, got_error = evaluation_utils.load_trained_model(folder, log_file)
    assert got_error is None
    assert str(type(got_model)) == "<class 'utils.train_model_utils.create_ffnn_model.<locals>.FeedforwardNetwork'>"


def test_create_loader_for_file():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "evaluation_utils_test", "test_create_loader_for_file")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    # Test if everything works as intended
    pd.DataFrame([[1.256, 4.365125, 2.45788]]).to_csv(
        os.path.join(folder, "11_some_identifier.csv"), index=False)
    got_loader, got_y_transformed = evaluation_utils.create_loader_for_file(os.path.join(
        folder, "11_some_identifier.csv"), False, False, pd.DataFrame(), None, "top")
    assert torch.equal(got_y_transformed, torch.zeros((1)).long())
    assert isinstance(got_loader, DataLoader)


def test_create_predictions():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "evaluation_utils_test", "test_create_predictions")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    model_folder = os.path.join(folder, "model_folder")
    os.makedirs(model_folder)

    log_file = {"train_model": {"seed": 42, "batch_normalization": False, "n_features": 3, "out_szs": 4, "dropout_rate": 0.1, "layer_structure": "10", "hierarchical_model": "top", "type": "FFNN"},
                "generate_features": True, "reduce_data": True, "balancing_over": {"label_depth": 2}, "prepare_dataset": True}
    error = file_utils.save_json(log_file, os.path.join(model_folder, "log.json"))
    assert error is None

    test_data_folder = os.path.join(folder, "test_data")
    os.makedirs(test_data_folder)
    error = file_utils.save_json(log_file, os.path.join(test_data_folder, "log.json"))
    assert error is None
    # create the training data
    pd.DataFrame([[1.648], [6.245], [3.1455]]).to_csv(os.path.join(test_data_folder, "1_some_ident.csv"), index=False, header=False)
    pd.DataFrame([[0.648], [6.245], [3.1455]]).to_csv(os.path.join(test_data_folder, "2_some_ident.csv"), index=False, header=False)

    pd.DataFrame([[1], [6], [3]]).to_csv(os.path.join(test_data_folder, "labels.csv"))
    # create a file for the standardization
    pd.DataFrame({"mean": [0, 0, 0], "scale": [1, 1, 1], "var": [1, 1, 1]}).T.to_csv(os.path.join(model_folder, "standardization_std.csv"))

    layers, error = train_model_utils.build_layers_from_string(log_file.get("train_model").get("layer_structure"), "FFNN")
    assert error is None
    Model = train_model_utils.build_nn_model("FFNN", log_file.get("train_model").get("batch_normalization"), log_file.get("train_model").get("seed"), log_file)
    model = Model(log_file.get("train_model").get("n_features"), log_file.get("train_model").get("out_szs"), layers, log_file.get("train_model").get("dropout_rate"),
                    log_file.get("train_model").get("batch_normalization"))
    error = train_model_utils.save_model(model, model_folder)
    assert error is None
    model = model.eval()

    got_confusion_matrix, got_accuracy, got_results_df, prediction_time, error = evaluation_utils.create_predictions(model, test_data_folder, model_folder, True, folder, False, 1.)
    assert error is None

    want_df = pd.DataFrame({"file": [os.path.join(test_data_folder, "2_some_ident.csv"), os.path.join(test_data_folder, "1_some_ident.csv")], "true_label": [2, 1], "prediction": [1, 1], 'Raw_prediction_label_1': [1.395216, 1.361137], 'Raw_prediction_label_2': [-0.812230, -0.963585], 'Raw_prediction_label_3': [0.153795, 0.240695], 'Raw_prediction_label_4': [-0.080446, -0.052186]})
    want_df["prediction"] = want_df["prediction"].astype(np.int32)
    if "win" in sys.platform:
        want_df["file"] = [file.replace("\\","/") for file in want_df["file"]]

    assert_frame_equal(got_results_df, want_df)
    assert got_accuracy == 50.
    assert isinstance(got_confusion_matrix, torch.Tensor) 


def test_calculate_precision_recall_f1():
    # Test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "evaluation_utils_test", "test_calculate_precision_recall_f1")
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)

    log_file = {"train_model": {"hierarchical_model": "top"}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    results_df = pd.DataFrame({"true_label": [0, 0, 1, 1, 2, 2], "prediction": [0, 0, 1, 2, 1, 1]})
    got = evaluation_utils.calculate_precision_recall_f1(results_df, folder)

    want = pd.DataFrame({"precision": [1.0, 1/3, 0., 0.5, 0.444444444444, 0.444444444],
                         "recall": [1.0, 0.5, 0., 0.5, 0.5, 0.5],
                         "f1-score": [1.0, 0.4, 0., 0.5, 0.466667, 0.466667],
                         "support": [2.0, 2.0, 2., 0.5, 6.0, 6.0], })
    want.index = ['lifting', 'walking', 'resting', 'accuracy', 'macro avg', 'weighted avg']
    print(got.index)
    assert_frame_equal(got, want)


def test_calculate_cohen_kappa_score():
    # Test if everything works as intended
    got = evaluation_utils.calculate_cohen_kappa_score(pd.DataFrame({"true_label": [0, 1, 2], "prediction": [2, 1, 0]}))
    assert got == 0.0


def test_calculate_matthews_corr_coeff():
    # Test if everything works as intended
    got = evaluation_utils.calculate_matthews_corr_coeff(pd.DataFrame({"true_label": [0, 1, 2], "prediction": [2, 1, 0]}))
    assert got == 0.0


def test_calculate_metrics():
    # Test if the true_label col is missing
    got_df, got_cohen, got_matthew, got_error = evaluation_utils.calculate_metrics(pd.DataFrame({"wrong_label": [0, 1, 2], "prediction": [2, 1, 0]}), "somewhere")
    assert got_df.empty

    # Test if the prediction col is missing
    got_df, got_cohen, got_matthew, got_error = evaluation_utils.calculate_metrics(pd.DataFrame({"true_label": [0, 1, 2], "guesses": [2, 1, 0]}), "somewhere")
    assert got_df.empty


def test_transform_svm_output_to_valid_tensor():
    input_ndarray = np.arange(1)
    want = torch.tensor([[1, 0, 0]])
    got = evaluation_utils.transform_svm_output_to_valid_tensor(
        input_ndarray, 3)
    assert torch.equal(got, want)


def test_valid_table_hierarchical():
    # test if one of the given folders does not exist
    root = test_config['TEST_ROOT']
    non_existing_folder = os.path.join(root, "evaluation_utils_test", "test_valid_table_hierarchical", "non_existing_subfolder")
    if os.path.exists(non_existing_folder):
        shutil.rmtree(non_existing_folder)
    
    table = {"top_level_folder":non_existing_folder}
    got = evaluation_utils.valid_table_hierarchical(table)
    assert got == "top_level_folder_not_found"

    # Tets if the folder does no contain a log file
    os.makedirs(non_existing_folder)
    existing_folder = non_existing_folder
    table = {"top_level_folder":existing_folder}
    got = evaluation_utils.valid_table_hierarchical(table)
    assert got == "top_level_folder_does_not_contain_a_log_file"

    # test if the log file belongs to a folder which is not an evaluation folder
    log_file_loc = os.path.join(existing_folder, "log.json")
    error = file_utils.save_json({"some_preprocessing":{"some_param":12}}, log_file_loc)
    assert error is None
    got = evaluation_utils.valid_table_hierarchical(table)
    assert got == "received _model_not_evaluated_prior"

    # Test if the log file belongs to a model not using featured data
    error = file_utils.save_json({"evaluate_model_second_location":{"some_param":12}}, log_file_loc)
    assert error is None
    got = evaluation_utils.valid_table_hierarchical(table)
    assert got == "received _model_not_evaluated_on_featured_data"

    # Test if the folder does not contain a predictions.csv
    error = file_utils.save_json({"evaluate_model_second_location":{"generate_features":12}}, log_file_loc)
    assert error is None
    got = evaluation_utils.valid_table_hierarchical(table)
    assert got == "top_level_folder_does_not_contain_a_predictions_file"

    # test if the predictions_csv contains too much activity levels
    pd.DataFrame({"true_label":[x for x in range(10)]}).to_csv(os.path.join(existing_folder, "predictions.csv"))
    got = evaluation_utils.valid_table_hierarchical(table)
    assert got == "too_much_true_labels_within_predictions_df"

    # Tets if the given folders are identical
    pd.DataFrame({"true_label":[x for x in range(3)]}).to_csv(os.path.join(existing_folder, "predictions.csv"))
    table = {"top_level_folder":existing_folder, "lifting_folder":existing_folder, "walking_folder":existing_folder, "nr_obs_per_activity":10000}
    got = evaluation_utils.valid_table_hierarchical(table)
    assert got == "received_identical_folders"

    # test if everything works as intended
    existing_folder_2 = os.path.join(root, "evaluation_utils", "some_existing_folder", "existing_subfolder_2")
    if not os.path.exists(existing_folder_2):
        os.makedirs(existing_folder_2)
    pd.DataFrame({"true_label":[x for x in range(2)]}).to_csv(os.path.join(existing_folder_2, "predictions.csv"))
    log_file_loc = os.path.join(existing_folder_2, "log.json")
    error = file_utils.save_json({"evaluate_model_second_location":{"generate_features":12}}, log_file_loc)
    assert error is None
    
    existing_folder_3 = os.path.join(root, "evaluation_utils", "some_existing_folder", "existing_subfolder_3")
    if not os.path.exists(existing_folder_3):
        os.makedirs(existing_folder_3)
    pd.DataFrame({"true_label":[x for x in range(2)]}).to_csv(os.path.join(existing_folder_3, "predictions.csv"))
    log_file_loc = os.path.join(existing_folder_3, "log.json")
    error = file_utils.save_json({"evaluate_model_second_location":{"generate_features":12}}, log_file_loc)
    assert error is None

    table = {"top_level_folder":non_existing_folder, "lifting_folder":existing_folder_2, "walking_folder":existing_folder_3, "nr_obs_per_activity":10000}

    got = evaluation_utils.valid_table_hierarchical(table)
    assert got is None
    for folder in [non_existing_folder, existing_folder, existing_folder_2, existing_folder_3]:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def test_create_confusion_matrix():
    # Test if everything works as intended
    predictions_df = pd.DataFrame({"true_label":[1,1,2,2], "prediction":[1,1,1,2]})
    got_matrix, got_acc, got_error = evaluation_utils.create_confusion_matrix(predictions_df, False, "", "top", False)
    assert got_error is None
    assert got_acc == 75.0
    assert isinstance(got_matrix, torch.Tensor)
    assert got_matrix[0,0] == 2
    assert got_matrix[1,1] == 1
    assert got_matrix[0,1] == 0
    assert got_matrix[1,0] == 1


def test_add_probability_scores():
    input_df = pd.DataFrame({"true_label":             [1, 2, 3, 2],
                             "Raw_prediction_label_1": [1, 2, 3, 4], 
                             "Raw_prediction_label_2": [3, 1, 7, 8],
                             "Raw_prediction_label_3": [4, 2, 1, 7]}
                            )
    prob_corrects = [0.04742587317756678, 0.2689414213699951, 0.0024726231566347743, 0.7310585786300049]
    prob_incorrects = [1 - prob_correct for prob_correct in prob_corrects]
    want_df = input_df.copy(True)
    want_df['probability_correct'] =  prob_corrects
    want_df['probability_incorrect'] =  prob_incorrects

    got_df = evaluation_utils.add_probability_scores(input_df)
    assert_frame_equal(got_df, want_df)

def test_detect_as_multi_class_label():
    # Test for a resting file
    assert evaluation_utils.detect_as_multi_class_label(3, 1, False) == 7
    assert evaluation_utils.detect_as_multi_class_label(3, 1, True) == 6

    # test for a holding file
    assert evaluation_utils.detect_as_multi_class_label(1, 3, False) == 3
    assert evaluation_utils.detect_as_multi_class_label(1, 2, True) == 2

    # test for a walking normal
    assert evaluation_utils.detect_as_multi_class_label(2, 1, False) == 4
    assert evaluation_utils.detect_as_multi_class_label(2, 1, True) == 3


def test_get_top_level_and_sublevel_value():
    # test for a resting file
    colname = "true_label"
    top_df = pd.DataFrame({"window_uuid": ["a", "b"], colname: [3, 1]})
    lift_df = pd.DataFrame({"window_uuid": ["a", "b"], colname: [2, 3]})
    walk_df = pd.DataFrame({"window_uuid": ["a", "b"], colname: [0, 1]})
    label_based_on_filename = 3
    got_top, got_sub = evaluation_utils.get_top_level_and_sublevel_value(top_df, lift_df, walk_df, colname, window_uuid="a", label_based_on_filename=label_based_on_filename)
    assert got_sub == 0
    assert got_top == 3 

    # test for a lifting file
    colname = "true_label"
    top_df = pd.DataFrame({"window_uuid": ["a", "b"], colname: [1, 1]})
    lift_df = pd.DataFrame({"window_uuid": ["a", "b"], colname: [2, 3]})
    walk_df = pd.DataFrame({"window_uuid": ["a", "b"], colname: [3, 1]})
    label_based_on_filename = 1
    got_top, got_sub = evaluation_utils.get_top_level_and_sublevel_value(top_df, lift_df, walk_df, colname, window_uuid="a", label_based_on_filename=label_based_on_filename)
    assert got_sub == 2
    assert got_top == 1


def test_join_hierarchical_classifiers():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "evaluation_utils_test", "test_join_hierarchical_classifiers")
    example_files = ["somewhere/1_prepared_window-uuid-1", "somewhere/2_prepared_window-uuid-2", "somewhere/3_prepared_window-uuid-3"]
    top_df = pd.DataFrame({"file": example_files, "true_label": [1, 2, 3], "prediction": [1, 2, 3]})
    # Only for the first sample the lifting subclassifer was called, but he predicted the wrong class
    lifting_df = pd.DataFrame({"file": ["somewhere/1_prepared_window-uuid-1"], "true_label": [1], "prediction": [3]})
    walking_df = pd.DataFrame({"file": ["somewhere/2_prepared_window-uuid-2"], "true_label": [2], "prediction": [2]})
    
    # Create the files
    top_df_folder = os.path.join(folder, "top-folder")
    lifting_df_folder = os.path.join(folder, "lifting-folder")
    walking_folder = os.path.join(folder, "walking-folder")
    for sub_folder in [top_df_folder, lifting_df_folder, walking_folder]:
        if os.path.exists(sub_folder):
            shutil.rmtree(sub_folder)
        os.makedirs(sub_folder)
    top_df.to_csv(os.path.join(top_df_folder, "predictions.csv"))
    lifting_df.to_csv(os.path.join(lifting_df_folder, "predictions.csv"))
    walking_df.to_csv(os.path.join(walking_folder, "predictions.csv"))
    for sub_folder in [top_df_folder, lifting_df_folder, walking_folder]:
        assert os.path.isfile(os.path.join(sub_folder, "predictions.csv"))

    # for a 7 class problem
    want = pd.DataFrame({"window_uuid": ['window-uuid-1', 'window-uuid-2', 'window-uuid-3'], "true_label": [1, 5, 7], "prediction": [3, 5, 7]})
    got = evaluation_utils.join_hierarchical_classifiers(top_df_folder, lifting_df_folder, walking_folder, False)
    assert_frame_equal(got, want)


    # for a 6 class problem
    lifting_df = pd.DataFrame({"file": ["somewhere/1_prepared_window-uuid-1"], "true_label": [1], "prediction": [2]})
    lifting_df.to_csv(os.path.join(lifting_df_folder, "predictions.csv"))
    want = pd.DataFrame({"window_uuid": ['window-uuid-1', 'window-uuid-2', 'window-uuid-3'], "true_label": [1, 4, 6], "prediction": [2, 4, 6]})
    got = evaluation_utils.join_hierarchical_classifiers(top_df_folder, lifting_df_folder, walking_folder, True)
    assert_frame_equal(got, want)


def test_reweight_joined_predictions():
    samples_per_activities = {1: 4, 2:8, 3:6}
    predictions_df = pd.DataFrame({"true_label": [1, 1, 2, 2, 3, 3, 3], "data": [10, 11, 21, 22, 30, 31, 32]})
    got = evaluation_utils.reweight_joined_predictions(predictions_df, samples_per_activities)
    want = pd.DataFrame({"true_label": [1,   1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3, 3],
                         "data":       [10, 11, 10, 10, 21, 22, 22, 21, 21, 21, 21, 21, 30, 31, 32, 32, 31, 30]},
                          index=       [ 0,  1,  0,  0,  2,  3,  3,  2,  2,  2,  2,  2,  4,  5,  6,  6,  5, 4])
    assert_frame_equal(got, want)


def test_add_probability_scores():
    predictions_df = pd.DataFrame({"true_label": [1, 2, 3],
                                   "Raw_prediction_label_1": [2, 8, 9],
                                   "Raw_prediction_label_2": [3, 5, 1],
                                   "Raw_prediction_label_3": [2, 6, 12]})
    probability_correct = [math.exp(2) / (math.exp(2) + math.exp(3)),
                           math.exp(5) / (math.exp(5) + math.exp(8)),
                           math.exp(12) / (math.exp(12) + math.exp(9))]
    probability_incorrect = [1- prob_correct for prob_correct in probability_correct]
    got = evaluation_utils.add_probability_scores(predictions_df.copy(True))
    predictions_df['probability_correct'] = probability_correct
    predictions_df['probability_incorrect'] = probability_incorrect
    assert_frame_equal(got, predictions_df)


def test_valid_model_comparison_table():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "evaluation_utils_test", "test_valid_model_comparison_table")
    model_folder_1 = os.path.join(folder, "model_1")
    model_folder_2 = os.path.join(folder, "model_2")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(model_folder_1)
    os.makedirs(model_folder_2)
    
    # received identical folders
    table = {'model_folder_1': model_folder_1, 'model_folder_2': model_folder_1}
    assert evaluation_utils.valid_model_comparison_table(table) == "received-identical_models"
    
    # Not evaluated_models
    table = {'model_folder_1': model_folder_1, 'model_folder_2': model_folder_2}
    assert evaluation_utils.valid_model_comparison_table(table) == "received_not_evaluated_models"

    # Needed cols missing
    model_folder_1 = os.path.join(folder, "model_1", "evaluate_model")
    model_folder_2 = os.path.join(folder, "model_2", "evaluate_model")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(model_folder_1)
    os.makedirs(model_folder_2)
    lacking_df = pd.DataFrame({"a col": [1,2,3], "prediction": [3, 1, 1]})
    lacking_df.to_csv(os.path.join(model_folder_1, "predictions.csv"))
    lacking_df.to_csv(os.path.join(model_folder_2, "predictions.csv"))
    table = {'model_folder_1': model_folder_1, 'model_folder_2': model_folder_2}
    assert evaluation_utils.valid_model_comparison_table(table) == "needed_cols_not_found"

    # Different true labels
    valid_df = pd.DataFrame({"true_label": [1,2,3], "prediction": [3, 1, 1]})
    valid_df.to_csv(os.path.join(model_folder_1, "predictions.csv"))
    valid_df = pd.DataFrame({"true_label": [5,6,7], "prediction": [5, 7, 7]})
    valid_df.to_csv(os.path.join(model_folder_2, "predictions.csv"))
    assert evaluation_utils.valid_model_comparison_table(table) == "differing_labels_found_for_model"


def test_ensure_window_uuid_is_present():
    # Nothing to do
    df = pd.DataFrame({"window_uuid": ["a", "b"]})
    got = evaluation_utils.ensure_window_uuid_is_present(df)
    assert_frame_equal(got, df)
    # Needed to be generated   
    df = pd.DataFrame({"file": ["somewhere/a-uuid.csv", "somewhere/b-uuid.csv"]})
    got = evaluation_utils.ensure_window_uuid_is_present(df.copy(True))
    df['window_uuid'] = ['a-uuid', 'b-uuid']
    assert_frame_equal(got, df)


def test_misclassified():
    assert evaluation_utils.misclassified(pd.Series([1, 2]))
    assert not evaluation_utils.misclassified(pd.Series([5, 5]))


def test_create_McNemar_contingency_table():
    predictions_1 = pd.DataFrame({"true_label": [1, 1, 2, 2, 3, 3],
                                  "prediction": [1, 2, 2, 2, 1, 3],
                                  "window_uuid": ["a", "b", "c", "d", "e", "f"]})
    predictions_2 = pd.DataFrame({"true_label": [1, 1, 2, 2, 3, 3],
                                  "prediction": [1, 1, 2, 2, 3, 3],
                                  "window_uuid": ["a", "b", "c", "d", "e", "f"]})

    want = [[0, 2], [0,4]]
    got = evaluation_utils.create_McNemar_contingency_table(predictions_1, predictions_2)
    assert got == want


def test_perform_mc_nemar_test():
    predictions_1 = pd.DataFrame({"true_label": [1, 1, 2, 2, 3, 3, 1, 1],
                                  "prediction": [1, 2, 2, 2, 1, 3, 1, 2],
                                  "window_uuid": ["a", "b", "c", "d", "e", "f", "a", "b"]})
    predictions_2 = pd.DataFrame({"true_label": [1, 1, 2, 2, 3, 3, 1, 1],
                                  "prediction": [1, 1, 2, 2, 3, 3, 1, 2],
                                  "window_uuid": ["a", "b", "c", "d", "e", "f", "a", "b"]})
    got = evaluation_utils.perform_mc_nemar_test(predictions_1, predictions_2, {}, 0.05)
    want = {"McNemars_test": {'mc_nemar_test_statistic': 0.5, "mc_nemar_p_value": 0.47950012218695337}}
    assert got == want


def test_accuracy():
    got = evaluation_utils.calculate_accuracy(pd.DataFrame({"blob":["invalid data"]}))
    assert got == 0

    got = evaluation_utils.calculate_accuracy(pd.DataFrame({"true_label":[0,1,2,3,4,5], "prediction": [0,1,2,3,4,5]}))
    assert got == 1

    got = evaluation_utils.calculate_accuracy(pd.DataFrame({"true_label":[0,1,2,3,2,1], "prediction": [0,1,2,3,4,5]}))
    assert round(got, 4) == round(0.6666666666666666, 4)


def test_calculate_f1_score():
    got = evaluation_utils.calculate_f1_score(pd.DataFrame({"blob":["invalid data"]}))
    assert got == 0

    got = evaluation_utils.calculate_f1_score(pd.DataFrame({"true_label":[0,1,2,3,4,5], "prediction": [0,1,2,3,4,5]}))
    assert got == 1

    got = evaluation_utils.calculate_f1_score(pd.DataFrame({"true_label":[0,1,2,3,2,1], "prediction": [0,1,2,3,4,5]}))
    assert round(got, 4) == round(0.555555555, 4)


def tester_from_filename():
    got = evaluation_utils.tester_from_filename("somehwere/some_folder/salkdhflashdflkhj_tester15(10).csv")
    assert got == 15
    
    got = evaluation_utils.tester_from_filename("/somehwere/some_folder/salkdhflashdflkhj_tester99(10).json")
    assert got == 99

def test_perform_wilocoxon_test():
    data = [12, 13, -1, 15, 17, 18, 22, 24, 38]
    got_res, got_res_greater = evaluation_utils.perform_wilocoxon_test(data)
    assert got_res.statistic == 1.0
    assert got_res.pvalue == 0.0078125
    assert got_res_greater.statistic == 44.0
    assert got_res_greater.pvalue == 0.00390625


def test_perform_wilcoxon_signed_rank_test():
    all_tester_names = ["f/h_tester0(10).csv"] * 4
    all_tester_names.extend(["f/h_tester1(10).csv"] * 4)
    all_tester_names.extend(["f/h_tester2(10).csv"] * 4)
    all_tester_names.extend(["f/h_tester3(10).csv"] * 4)
    predictions_2 = pd.DataFrame({"file": all_tester_names,
                                  "true_label": [0, 1, 2, 3] *4,
                                  "prediction": [1] *16})

    # This classifier classifies everything correctly
    predictions_1 = pd.DataFrame({"file": all_tester_names,
                                  "true_label": [0, 1, 2, 3] *4,
                                  "prediction": [0, 1, 2, 3] *4})
    
    # This classifier classifies only seldomly correctly
    predictions_2 = pd.DataFrame({"file": all_tester_names,
                                  "true_label": [0, 1, 2, 3] *4,
                                  "prediction": [0] *16})

    
    got = evaluation_utils.perform_wilcoxon_signed_rank_test(predictions_1, predictions_2, {}, 0.05)
    want = {'Wilcoxon_test': {'Accuracy': {'Wilcoxon_greater_p_value': 0.0625,
                                           'Wilcoxon_p_value': 0.125,
                                           'Wilcoxon_test_greater_statistic': 10.0,
                                           'Wilcoxon_test_statistic': 0.0},
                              'F1': {'Wilcoxon_greater_p_value': 0.0625,
                                     'Wilcoxon_p_value': 0.125,
                                     'Wilcoxon_test_greater_statistic': 10.0,
                                     'Wilcoxon_test_statistic': 0.0},
                              'MCC': {'Wilcoxon_greater_p_value': 0.0625,
                                      'Wilcoxon_p_value': 0.125,
                                      'Wilcoxon_test_greater_statistic': 10.0,
                                      'Wilcoxon_test_statistic': 0.0}}}
    assert got == want

def test_perform_friedman_test():
    root = test_config['TEST_ROOT']
    folder =os.path.join(root, "evaluation_utils_test", "test_perform_friedman_test")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


    test_files_folder = [os.path.join(folder, f"folder_{i}") for i in range(4)]
    [os.makedirs(test_data_folder) for test_data_folder in test_files_folder]

    all_tester_names = ["f/h_tester0(10).csv"] * 4
    all_tester_names.extend(["f/h_tester1(10).csv"] * 4)
    all_tester_names.extend(["f/h_tester2(10).csv"] * 4)
    all_tester_names.extend(["f/h_tester3(10).csv"] * 4)

    # This classifier classifies everything correctly
    predictions_1 = pd.DataFrame({"file": all_tester_names,
                                  "true_label": [0, 1, 2, 3] *4,
                                  "prediction": [0, 1, 2, 3] *4})
    
    # This classifier classifies seldomly correctly
    predictions_2 = pd.DataFrame({"file": all_tester_names,
                                  "true_label": [0, 1, 2, 3] *4,
                                  "prediction": [0] *16})
    
    predictions_3 = pd.DataFrame({"file": all_tester_names,
                                  "true_label": [0, 1, 2, 3] *4,
                                  "prediction": [1] *16})
    predictions_4 = pd.DataFrame({"file": all_tester_names,
                                  "true_label": [0, 1, 2, 3] *4,
                                  "prediction": [4] *16})
    predictions_3.to_csv(os.path.join(test_files_folder[0], "predictions.csv"))
    predictions_3.to_csv(os.path.join(test_files_folder[1], "predictions.csv"))
    predictions_4.to_csv(os.path.join(test_files_folder[2], "predictions.csv"))
    predictions_4.to_csv(os.path.join(test_files_folder[3], "predictions.csv"))
    assert sum(os.path.isfile(os.path.join(test_data_folder, "predictions.csv")) for test_data_folder in test_files_folder) == 4

    got = evaluation_utils.perform_friedman_test(predictions_1, predictions_2, test_files_folder, {}, 0.05)
    want = {'Friedman_test': {'Accuracy': {'Friedman_test_p_value': 0.0012497305630313797,
                                           'Friedman_test_statistic': 19.999999999999993},
                              'F1': {'Friedman_test_p_value': 0.0012497305630313797,
                                     'Friedman_test_statistic': 19.999999999999993},
                              'MCC': {'Friedman_test_p_value': 0.0012497305630313797,
                                      'Friedman_test_statistic': 19.999999999999993}},
           }
    assert got == want