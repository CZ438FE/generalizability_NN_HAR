import os
import shutil
import torch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from utils import plot_utils, train_model_utils, file_utils
from config.test_config import test_config

def test_return_dict_nr_to_label():
    # Test if an an invalid level was given
    invalid_level = "invalid_level"
    got_dict, got_error = plot_utils.return_dict_nr_to_label(invalid_level)
    assert not got_dict
    assert got_error ==  "invalid_granularity_given"


def test_change_label_nr_to_human_readable():
    # Test if {} gets returned when no valid dict containing data gets inserted
    invalid_dict = {}
    valid_dict_nr_to_label = {1: "activity one", 2: "activity two", 4: "activity four", 5: "activity five"}
    got = plot_utils.change_label_nr_to_human_readable(invalid_dict, valid_dict_nr_to_label)
    assert not got
    # Test if {} gets returned the dict containing the mapping information is empty
    valid_dict = {4: 200, 2: 250, 5: 300, 1: 800}
    invalid_dict_nr_to_label = {}
    got = plot_utils.change_label_nr_to_human_readable(valid_dict, invalid_dict_nr_to_label)
    assert not got
    # Test if the dict containing the mapping information is empty
    invalid_dict_nr_to_label = {}
    got = plot_utils.change_label_nr_to_human_readable(valid_dict, invalid_dict_nr_to_label)
    got = plot_utils.change_label_nr_to_human_readable(valid_dict, invalid_dict_nr_to_label)
    # Test if the conversion works with valid inputs
    want = {"activity\none": 800, "activity\ntwo": 250,
            "activity\nfour": 200, "activity\nfive": 300, }
    got = plot_utils.change_label_nr_to_human_readable(valid_dict, valid_dict_nr_to_label)
    assert got == want


def test_show_label_distribution():
    # test a wrong granularity is given
    invalid_granularity = "invalid_gran"
    valid_labels_distribution = {"1": 20, "2": 25, "3": 34, "4": 0}
    got = plot_utils.show_label_distribution(invalid_granularity, valid_labels_distribution, True)
    assert got is None
    # Test if no labels_distribution_dict was given
    got = plot_utils.show_label_distribution("mid", {}, True)
    assert got is None
    # Test if everything works as intended
    got = plot_utils.show_label_distribution("top", valid_labels_distribution, True)


def test_infer_level():
    # Test a label depth of 3
    got_level, got_error = plot_utils.infer_level("211")
    assert got_level == "low"
    assert got_error is None

    # Test a label depth of 2
    got_level, got_error = plot_utils.infer_level("21")
    assert got_level == "mid"
    assert got_error is None

    # Test a label depth of 1
    got_level, got_error = plot_utils.infer_level("2")
    assert got_level == "top"
    assert got_error is None

    # Test if it finds the correct depth with a list of depth 3
    got_level, got_error = plot_utils.infer_level(["211", "15", "31"])
    assert got_level == "low"
    assert got_error is None

    # Test if it finds the correct depth with a list of depth 2
    got_level, got_error = plot_utils.infer_level(["21", "15", "31"])
    assert got_level == "mid"
    assert got_error is None

    # Test if it finds the correct depth with a list of depth 1
    got_level, got_error = plot_utils.infer_level(["1", "3", "3"])
    assert got_level == "top"
    assert got_error is None

    # test if an invalid depth is given
    got_level, got_error = plot_utils.infer_level("251651651")
    assert got_error == "level_not_defined_for_maxlen_9"

    # test if an invalid datatype is given
    got_level, got_error = plot_utils.infer_level(pd.DataFrame({"251651651": [1, 1, 5]}))
    assert got_error, "invalid_dtype_given_for_inferring_level"


def test_replace_list_labels_with_human_readable_form():
    # Test if it correctly replaces with a depth of "mid"
    list_of_labels_mid = [11, 31, 24]
    got_levels, got_error = plot_utils.replace_list_labels_with_human_readable_form(list_of_labels_mid)
    want = ["Lifting", "Resting", "Walking Sideways"]
    assert got_levels == want

    # Test if it correctly replaces with a depth of "top"
    list_of_labels_mid = [2, 3, 1]
    got_levels, got_error = plot_utils.replace_list_labels_with_human_readable_form(list_of_labels_mid)
    want = ["Walking", "Resting", "Lifting"]
    assert got_levels == want

    # Test if it correctly replaces with a depth of "low"
    list_of_labels_mid = [221, 32, 15, 222]
    got_levels, got_error = plot_utils.replace_list_labels_with_human_readable_form(list_of_labels_mid)
    want = ["Upstairs Free", "Sitting", "Holding", "Upstairs Carrying"]
    assert got_levels == want

    # Test if an empty list gets returned when an invalid label type is given
    got_levels, got_error = plot_utils.replace_list_labels_with_human_readable_form(pd.DataFrame())
    assert not got_levels


def test_plot_original_samples_on_data_after_resampling():
    root = test_config['TEST_ROOT']
    saving_folder = os.path.join(root, "plot_utils_test", "test_plot_original_samples_on_data_after_resampling")
    if os.path.exists(saving_folder):
        shutil.rmtree(saving_folder)
    os.makedirs(saving_folder)

    # Test if None is returned when creating the list of human-readable levels fails
    invalid_prop_original_data_dict = {"class": pd.DataFrame(), "prop_existing_after_balancing": [None]}
    got = plot_utils.plot_original_samples_on_data_after_resampling(invalid_prop_original_data_dict, saving_folder, False)
    assert got == "invalid_dtype_given_for_inferring_level"

    # Test if None is returned when the given dict is malformed
    invalid_prop_original_data_dict = {"class": pd.DataFrame()}
    got = plot_utils.plot_original_samples_on_data_after_resampling(invalid_prop_original_data_dict, saving_folder, False)
    assert got == "malformed_prop_original_data_dict_given"

    # test if it correctly displays when everything is valid
    valid_prop_original_data_dict = {"class": [1, 2, 3], "prop_existing_after_balancing": [0.8, 0.7, 0.95]}
    got = plot_utils.plot_original_samples_on_data_after_resampling(valid_prop_original_data_dict, saving_folder, True)
    assert got is None
    assert  os.path.isfile(os.path.join(saving_folder, "proportion_of_original_data_on_data_after_resampling.png"))


def test_plot_accuracies_and_loss_over_batches():
    # test if False gets returned, when the input is empty
    got = plot_utils.plot_accuracies_and_loss_over_batches([], [], [], [], False, "", 100)
    assert not got

    # test if False gets returned, when the input is invalid
    got = plot_utils.plot_accuracies_and_loss_over_batches([200, 200, 200], [], [], [], False, "", 100)
    assert got

    # test if true gets returned, when the input has errors, which may be fixed
    got = plot_utils.plot_accuracies_and_loss_over_batches([0.1, 200, 200], [], [], [], False, "", 100)
    assert got

    # test if true gets returned, when the input has errors, which may be fixed
    got = plot_utils.plot_accuracies_and_loss_over_batches([], [], [0.1, 200, 200], [], False, "", 100)
    assert got

    # Test if the output gets saved correctly
    root = test_config['TEST_ROOT']
    saving_folder = os.path.join(root, "plot_utils_test", "test_plot_accuracies_and_loss_over_batches")
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    if os.path.exists(os.path.join(saving_folder, "losses.png")):
        os.remove(os.path.join(saving_folder, "losses.png"))
    got = plot_utils.plot_accuracies_and_loss_over_batches([], [], [0.1, 200, 200], [], True, saving_folder, 100)
    assert got
    assert os.path.isfile(os.path.join(saving_folder, "Training_minibatch.png"))
    

def test_get_class_names_from_len():
    # test if a strange len was given
    got = plot_utils.get_class_names_from_len(8, "top")
    assert got == list(range(8))

    # Test if the lne is 3
    got = plot_utils.get_class_names_from_len(3, "top")
    assert got == ["lifting", "walking", "resting"]

    # Test if the len is 7
    got = plot_utils.get_class_names_from_len(7, "top")
    assert got == ["lifting", "lowering", "holding", "walking\nstraight", "walking\nupstairs", "walking\ndownstairs", "resting"]


def test_plot_confusion_matrix():
    # test if everything works as intended
    minibatch_size = 1
    dataset = "test_set"
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "plot_utils_test", "test_plot_confusion_matrix")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    file_names = [os.path.join(folder, "1_some_file.csv"), os.path.join(folder, "1_some_file_1.csv"), os.path.join(folder, "2_some_file.csv"), os.path.join(folder, "2_some_file_1.csv")]
    pd.DataFrame({0: [1]}).to_csv(file_names[0], index=False, header=False)
    pd.DataFrame({0: [1]}).to_csv(file_names[1], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[2], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[3], index=False, header=False)
    pd.DataFrame({"filename": file_names, "label": [1, 1, 2, 2]}).to_csv(os.path.join(folder, "labels.csv"), index=False)

    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    loader, error = train_model_utils.create_loader(folder, minibatch_size, dataset, 1.0, "top")
    assert error is None
    FFN_Model = train_model_utils.create_ffnn_model(False, 42)
    model = FFN_Model(1, 2, [1], 0.0, False)
    got_confusion_matr, got_accuracy, got_y_pred_joined = train_model_utils.display_confusion_matrix(loader, model, dataset, True, folder)
    assert torch.equal(got_confusion_matr, torch.tensor([[0, 2], [0, 2]], dtype=torch.long))
    assert got_accuracy == 50.00

    got = plot_utils.plot_confusion_matrix(got_confusion_matr, True, folder, "Test_utils", "top")
    want = pd.DataFrame([[0, 2], [0, 2]]).astype(np.int64)

    assert_frame_equal(got, want)
    assert os.path.isfile(os.path.join(folder, "Test_utils_confusion_matrix.png"))


def test_plot_course():
    # test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "plot_utils_test", "test_plot_course")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    success = plot_utils.plot_course([1, 2], "meter", "kmh", "testing", True, folder)
    assert success
    assert os.path.isfile(os.path.join(folder, "testing_meter.png"))
