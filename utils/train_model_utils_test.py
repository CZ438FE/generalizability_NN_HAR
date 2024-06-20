import os
import shutil
import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pandas.testing import assert_frame_equal
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from joblib import dump

from utils import train_model_utils, file_utils
from config.test_config import test_config

def test_valid_table():
    # Set a great amount of different_starting variables
    # create two folders which contain a file
    root = test_config['TEST_ROOT']
    base_folder = os.path.join(root, "train_model_utils_test")
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
    os.makedirs(base_folder)
    folder_with_files_1 = os.path.join(base_folder, "folder_contains_file")
    if not os.path.exists(folder_with_files_1):
        os.makedirs(folder_with_files_1)
    folder_with_files_2 = os.path.join(base_folder, "folder_also_contains_file")
    if not os.path.exists(folder_with_files_2):
        os.makedirs(folder_with_files_2)
    df = pd.DataFrame({"some_values": [1, 2, 3]})
    df.to_csv(os.path.join(folder_with_files_1, "somefile.csv"), index=False)
    df.to_csv(os.path.join(folder_with_files_2, "somefile.csv"), index=False)

    # Set a folder without files in it, if it exists delete it and create in new and empty
    folder_without_files_in_it = os.path.join(base_folder, "folder_contains_no_files")
    if os.path.exists(folder_without_files_in_it):
        shutil.rmtree(folder_without_files_in_it)
    os.makedirs(folder_without_files_in_it)

    # set a non_existing_folder
    non_existing_folder = os.path.join(base_folder, "folder_which_does_not_exist")
    if os.path.exists(non_existing_folder):
        shutil.rmtree(non_existing_folder)

    # Set other variables
    invalid_output_date = "2022:08:10T12:00:00"
    valid_output_date = "2022-08-10T06:00"

    # These values are used for minibatch size and epochs
    too_small_int = -200
    valid_int = 2000

    # Set valid and invalid layer_structures
    invalid_layer_structure = "250f4sdfsd5f100f20"
    too_small_layers = "400f0f0f0f-200"
    valid_layer_structure = "400f300f200f100f50f30"
    # these values are used for the learning- and dropout rate
    too_small_rate = -0.21448
    valid_rate = 0.21

    # Test if the trainings-data-folder does not exist
    got = train_model_utils.valid_table({"training_data": non_existing_folder, "dryrun": True})
    assert got == "training_data_folder_does_not_exist"

    # Test if there are no files in folder of trainings-data
    got = train_model_utils.valid_table({"training_data": folder_without_files_in_it, "dryrun": True})
    assert got == "no_data_in_dir"

    # Test if the test data folder does not exist
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": non_existing_folder, "dryrun": True, "minibatch_size": 200})
    assert got == "validation_data_folder_does_not_exist"

    # Test if there are no files in the validation_folder
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": folder_without_files_in_it, "dryrun": True, "minibatch_size": 200})
    assert got == "no_data_in_dir"

    # Test if the validation folder is identical to the trainings_data folder
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": folder_with_files_1, "dryrun": True, "minibatch_size": 200})
    assert got == "validation_data_folder_identical_to_trainings_data_folder"

    # Test if the output_date is malformed
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": None,
                                        "dryrun": True, "output_date": invalid_output_date, "minibatch_size": 200})
    assert got == "got_time_str_of_invalid_len"

    base_folder = os.path.join(root, "train_model_utils_test")

    # Test if the amount of epochs is too small
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": None, "dryrun": True, "output_date": valid_output_date,
                                        "minibatch_size": valid_int, "epochs": too_small_int, "data_path": base_folder, "type": "FFNN"})
    assert got == "epoch_amount_too_small"

    # Test if the layer_structure is invalid
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": None, "dryrun": True, "output_date": valid_output_date, "minibatch_size": valid_int,
                                        "epochs": valid_int, "layer_structure": invalid_layer_structure, "data_path": base_folder, "type": "FFNN"})
    assert got == "layerstring_malformed"

    # Test if the layer_structure is invalid: too small layers (width of 0)
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": None, "dryrun": True, "output_date": valid_output_date, "minibatch_size": valid_int,
                                        "epochs": valid_int, "layer_structure": too_small_layers, "type": "FFNN", "data_path": base_folder})
    assert got == "too_small_width_detected"

    # Test if the dropout rate is too small
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": None, "dryrun": True, "output_date": valid_output_date, "minibatch_size": valid_int,
                                        "epochs": valid_int, "layer_structure": valid_layer_structure, "dropout_rate": too_small_rate, "type": "FFNN", "data_path": base_folder})
    assert got == "dropout_rate_out_of_allowed_boundaries"

    # test if the learning rate is too small
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": None, "dryrun": True, "output_date": valid_output_date, "minibatch_size": valid_int, "epochs": valid_int,
                                        "layer_structure": valid_layer_structure, "learning_rate": too_small_rate, "type": "FFNN", "dropout_rate": None, "data_path": base_folder})
    assert got == "learning_rate_out_of_allowed_boundaries"

    # test if everything works when all the inputs are valid
    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": None, "dryrun": True, "output_date": valid_output_date, "minibatch_size": valid_int, "epochs": valid_int,
                                        "layer_structure": valid_layer_structure, "learning_rate": valid_rate, "type": "FFNN", "dropout_rate": None, "hierarchical": False, "granularity": "top", "data_path": base_folder})
    assert got is None

    got = train_model_utils.valid_table({"training_data": folder_with_files_1, "validation_data": folder_with_files_2, "dryrun": True, "output_date": valid_output_date, "minibatch_size": valid_int, "epochs": valid_int,
                                        "layer_structure": valid_layer_structure, "learning_rate": valid_rate, "dropout_rate": valid_rate, "type": "FFNN", "hierarchical": False, "granularity": "top", "data_path": base_folder})
    assert got is None


def test_valid_layerstring():
    valid_layer_str = "200f80f50"
    # Test if it correctly throws an error, when building a cnn is tried, but no conv or pooling layer are given
    got = train_model_utils.valid_layerstring(valid_layer_str, "CNN", "")
    assert got == "no_conv_or_pool_layer_given"

    # Test if a not yet implemented network type is being given
    got = train_model_utils.valid_layerstring(valid_layer_str, "LSTM", "")
    assert  got == "creating_layerstructure_for_given_type_not_implemented"


def test_valid_feedforward_layers():
    # test if correctly returns, when an invalid layer_width is given
    too_small_layers = "400f0f0f0f-200"
    got = train_model_utils.valid_feedforward_layers(too_small_layers.split("f"))
    assert got == "too_small_width_detected"

    # Test if the correctly returns, when the layer string is broken
    invalid_layer_structure = "250f4sdfsd5f100f20"
    got = train_model_utils.valid_feedforward_layers(invalid_layer_structure.split("f"))
    assert got == "layerstring_malformed"

    # Test if the a too large layer was detected
    too_big_layer_structure = "250f40000000f20"
    got = train_model_utils.valid_feedforward_layers(too_big_layer_structure.split("f"))
    assert got == "too_great_width_detected"

    # test if it correctly returns, when the layer structure is correct
    valid_layer_struct = "400f300f200f100f50f30"
    got = train_model_utils.valid_feedforward_layers(valid_layer_struct.split("f"))
    assert got is None


def test_valid_conv_and_pool_layers():
    # test when a layer was given which is neither a pool nor a convolution layer
    got = train_model_utils.valid_conv_and_pool_layers(["another_operation", "max_pool_2_2", "conv_2_2_2"])
    assert got == "received_layer_which_is_neither_conv_nor_pool"

    # test when everything is valid
    got = train_model_utils.valid_conv_and_pool_layers(["max_pool_2_2", "conv_2_2_2"])
    assert got is None


def test_valid_pooling_layer():
    # Test when a malformed layer str is given
    got = train_model_utils.valid_pooling_layer(["pool_2_2"])
    assert got == "got_layer_of_non_str_dt"

    got = train_model_utils.valid_pooling_layer("pool_2_2")
    assert got == "got_malformed_pool_str"

    got = train_model_utils.valid_pooling_layer("pool_max_2_2")
    assert got == "received_invalid_pooling_method"

    # Test when everything is valid
    got = train_model_utils.valid_pooling_layer("mean_pool_2_2")
    assert got is None


def test_valid_convolution_layer():
    # Test when a malformed layer str is given
    got = train_model_utils.valid_convolution_layer(["conv_5_1"])
    assert got ==  "got_layer_of_non_str_dt"

    got = train_model_utils.valid_convolution_layer("conv_5_1")
    assert got == "got_malformed_conv_str"

    got = train_model_utils.valid_convolution_layer("4_conv_5_1")
    assert got == "got_malformed_conv_str"

    # test when everything is valid
    got = train_model_utils.valid_convolution_layer("conv_5_1_1")
    assert got is None


def test_valid_amount_of_convolution_filters():
    # test when the amount is too small
    got = train_model_utils.valid_amount_of_convolution_filters(0)
    assert got == "received_too_small_amount_of_filters"

    # test when the amount is too big
    got = train_model_utils.valid_amount_of_convolution_filters(100000000000)
    assert got == "received_too_big_amount_of_filters"

    # test when the amount is valid
    got = train_model_utils.valid_amount_of_convolution_filters(5)
    assert got is None


def test_valid_kernel_size_and_stride():
    # test when the kernel size is too small
    got = train_model_utils.valid_kernel_size_and_stride(0, 5)
    assert got == "received_too_small_kernel_size"
    
    # test when the kernel size is too big
    got = train_model_utils.valid_kernel_size_and_stride(100000, 5)
    assert got ==  "received_too_big_kernel_size"

    # test when the stride is too small
    got = train_model_utils.valid_kernel_size_and_stride(5, 0)
    assert got == "received_too_small_stride"

    # test when the stride is bigger than the kernel size
    got = train_model_utils.valid_kernel_size_and_stride(5, 15)
    assert got == "got_stride_bigger_than_kernel_size"

    # test when everything is valid
    got = train_model_utils.valid_kernel_size_and_stride(5, 2)
    assert got is None


def test_split_layerstring_into_conv_and_ff_layers():
    # test when a layer is given which cannot be identifies as a ff nor a conv/pooling layer
    got_conv_layer, got_ff_layer, got_error = train_model_utils.split_layerstring_into_conv_and_ff_layers("another_operationf80f60f30")
    assert got_conv_layer == []
    assert got_ff_layer == []
    assert got_error == "could_not_interpret_layer_another_operation"

    # test when no pooling or conv layers were given
    got_conv_layer, got_ff_layer, got_error = train_model_utils.split_layerstring_into_conv_and_ff_layers("100f80f60f30")
    assert got_conv_layer == []
    assert got_ff_layer == []
    assert got_error == "no_conv_or_pool_layer_given"

    # test when no ff layer were given
    got_conv_layer, got_ff_layer, got_error = train_model_utils.split_layerstring_into_conv_and_ff_layers("conv_5_4_1fmax_pool_2_2fconv_5_3_1")
    assert got_conv_layer == []
    assert got_ff_layer == []
    assert got_error == "no_ff_layer_given"

    # test when everything is valid
    got_conv_layer, got_ff_layer, got_error = train_model_utils.split_layerstring_into_conv_and_ff_layers("conv_5_4_1fmax_pool_2_2fconv_5_3_1f860f320f32")
    assert got_conv_layer == ["conv_5_4_1", "max_pool_2_2", "conv_5_3_1"]
    assert got_ff_layer == ["860", "320", "32"]
    assert got_error is None
    assert got_error is None


def test_build_layers_from_string():
    # Test if an unspecified network type was given
    got_layer, got_error = train_model_utils.build_layers_from_string("djshflhdlf", "invalid_network_type")
    assert got_error == "building_layers_from_string_not_implemented_for_other_types"


def test_set_loss_function():
    # test if an invalid Loss function gets inserted
    invalid_loss = "such_an_invalid_loss"
    got_loss, got_error = train_model_utils.set_loss_function(invalid_loss)
    assert got_loss is None

    # test if something gets returned when a valid loss function is given
    got_loss, got_error = train_model_utils.set_loss_function("CEL")
    assert str(type(got_loss)) == "<class 'torch.nn.modules.loss.CrossEntropyLoss'>"


def test_set_optimizer():
    valid_optimizer = "adam"
    too_small_learning_rate = -2
    too_big_learning_rate = 400
    invalid_optimizer = "eve"
    # test if the learning rate is too small

    class Model():
        def parameters(x):
            return [torch.Tensor(1), torch.Tensor(10)]
    model = Model()
    got_optim, got_error = train_model_utils.set_optimizer(valid_optimizer, too_small_learning_rate, model)
    assert got_optim is None
    # Test if the learning rate is too big
    got_optim, got_error = train_model_utils.set_optimizer(valid_optimizer, too_big_learning_rate, model)
    assert got_optim is None
    # Test if an invalid optimizer is given
    got_optim, got_error = train_model_utils.set_optimizer(invalid_optimizer, 0.01, model)
    assert got_optim is None
    # Test if everything is valid
    got_optim, got_error = train_model_utils.set_optimizer(valid_optimizer, 0.01, model)
    assert str(type(got_optim)) == "<class 'torch.optim.adam.Adam'>"


def test_find_feature_columns():
    # Test if everything works when no column needs to be filtered
    df = pd.DataFrame({"some_col": [1, 2], "another_col": [8, 9], "yet_another_col": [19, 3]})
    got = train_model_utils.find_feature_columns(df.columns)
    assert ["some_col", "another_col", "yet_another_col"] ==got
    # Test if there are columns to remove
    df = pd.DataFrame({"some_col": [1, 2], "another_col": [8, 9], "yet_another_col": [
                        19, 3], "time": [89, 21], "label": [1, 5], "time_start": [100, 151]})
    got = train_model_utils.find_feature_columns(df.columns)
    assert ["some_col", "another_col", "yet_another_col"] ==got


def test_bring_to_correct_format():
    input_tensor = torch.Tensor(np.array([1, 2, 1, 3, 1]))
    got = train_model_utils.bring_to_correct_format(input_tensor)
    want = torch.LongTensor([0, 1, 0, 2, 0]).reshape(-1,)
    assert torch.equal(got, want)


def test_transform_predictions_for_confusion_matrix():
    data = np.array([[1.2, 2.145, 3.164], [-1.246, 1.035,0.3256], [5.2156, 8.4215, 0.321654]])
    start_tensor = torch.Tensor(data)
    want_tensor = torch.Tensor(np.array([[2], [1], [1]])).to(torch.int32).reshape(-1,)
    got = train_model_utils.transform_predictions_for_confusion_matrix(start_tensor)
    assert torch.equal(got, want_tensor)


def test_sum_identical_entries():
    # Test if everything works as intended
    tensor_1 = torch.tensor([1, 2, 1, 2, 1])
    tensor_2 = torch.tensor([0, 2, 1, 2, 4])
    got = train_model_utils.sum_identical_entries(tensor_1, tensor_2)
    assert got.item() == 3


def test_create_ffnn_model():
    # This test needs to be done before several other tests, as these depend on using this ffnn model

    # Test if the class gets returned correctly
    got = train_model_utils.create_ffnn_model(False, 42)
    # test if a type got returned
    assert isinstance(got, type)
    # test if the correct type got returned
    assert str(got) == "<class 'utils.train_model_utils.create_ffnn_model.<locals>.FeedforwardNetwork'>"

    # Test if the construction of the needed model can be done via this class
    model = got(5, 1, [3], 0.0, False)
    assert isinstance(model, got)

    # test if the construction of a model class goes as expected
    want_param_0 = nn.parameter.Parameter(torch.tensor([[0.3419,  0.3712, -0.1048,  0.4108, -0.0980],
                                                        [0.0902, -0.2177,  0.2626, 0.3942, -0.3281],
                                                        [0.3887,  0.0837,  0.3304,  0.0606,  0.2156]], requires_grad=True))
    want_param_1 = nn.parameter.Parameter(torch.tensor([-0.0631,  0.3448,  0.0661], requires_grad=True))
    want_param_2 = nn.parameter.Parameter(torch.tensor([[-0.2695,  0.1472, -0.2660]], requires_grad=True))
    want_param_3 = nn.parameter.Parameter(torch.tensor([-0.0677], requires_grad=True))
    param_dict = {0: want_param_0,
                  1: want_param_1,
                  2: want_param_2,
                  3: want_param_3}
    # test if all the values are as expected
    for number, params in enumerate(model.parameters()):
        for tensor_number, tensor_elem in enumerate(params):
            if tensor_elem.shape == torch.Size([]):
                assert round(tensor_elem.item(), 4) == param_dict[number][tensor_number]
            # As some tensors contain subtensors, indexing is needed one layer deeper
            else:
                for sub_tensor_number, sub_tensor_elem in enumerate(tensor_elem):
                    assert round(tensor_elem[sub_tensor_number].item(), 4) == param_dict[number][tensor_number][sub_tensor_number]


def test_build_nn_model():
    # test if everything works as intended
    got_model = train_model_utils.build_nn_model("FFNN", False, 42, {})
    assert str(got_model) == "<class 'utils.train_model_utils.create_ffnn_model.<locals>.FeedforwardNetwork'>"

    got_model = train_model_utils.build_nn_model("CNN", False, 42, {})
    assert str(got_model) == "<class 'utils.train_model_utils.create_convolutional_model.<locals>.ConvolutionalNetwork'>"

    got_model = train_model_utils.build_nn_model("RNN", False, 42, {"create_windows": {"window_length": 1000, "resampling_rate": 10}})
    assert str(got_model) == "<class 'utils.train_model_utils.create_rnn_model.<locals>.RecurrentNetwork'>"


def test_build_fully_connected_layers():
    # Test if everything works as intended
    got = train_model_utils.build_fully_connected_layers(10, ["conv_5_2_2", "max_pool_2_2", "8"], False, 0.0, 3)
    want = [nn.Linear(10, 8, True), nn.ReLU(inplace=True), nn.Linear(8, 3, True)]
    for number, layer in enumerate(got):
        assert str(layer) == str(want[number])


def test_create_convolutional_model():
    # test if everything works as intended
    got = train_model_utils.create_convolutional_model(False, 42)

    # test if a type got returned
    assert isinstance(got, type)

    # test if the correct type got returned
    assert str(got) == "<class 'utils.train_model_utils.create_convolutional_model.<locals>.ConvolutionalNetwork'>"

    # Test if the construction fails when the given layers cannot be used to create a CNN
    model = got(5, 1, ["3"], 0.0, False)
    assert isinstance(model, got)


def test_build_conv_and_pool_layers():
    # test if everything works as intended without dropout and batchnormalizatioon
    layers = ["conv_1_1_1", "max_pool_1_1", "20", "10"]
    n_features = 300
    dropout_rate = 0.0
    batch_normalization = False
    got_conv_and_pooling_layers, got_remaining_layers, got_error = train_model_utils.build_conv_and_pool_layers(layers, n_features, dropout_rate, batch_normalization)
    conv_layer, _, error = train_model_utils.build_convolution_layer(n_features, "conv_1_1_1")
    assert error is None
    pool_layer, error = train_model_utils.build_pooling_layer("max_pool_1_1")
    assert error is None

    want_conv_and_pooling_layers = [str(conv_layer), str(nn.ReLU(inplace=True)), str(pool_layer)]
    want_remaining_layers = [20, 10]
    assert got_error is None


    assert [str(x) for x in got_conv_and_pooling_layers] == want_conv_and_pooling_layers
    assert got_remaining_layers ==want_remaining_layers

    # Test with dropout and batch norm
    layers = ["conv_1_1_1", "max_pool_1_1", "20", "10"]
    n_features = 300
    dropout_rate = 0.5
    batch_normalization = True
    got_conv_and_pooling_layers, got_remaining_layers, got_error = train_model_utils.build_conv_and_pool_layers(layers, n_features, dropout_rate, batch_normalization)

    conv_layer, _, error = train_model_utils.build_convolution_layer(n_features, "conv_1_1_1")
    assert error is None
    pool_layer, error = train_model_utils.build_pooling_layer("max_pool_1_1")
    assert error is None

    want_conv_and_pooling_layers = [str(conv_layer), str(nn.ReLU(inplace=True)), str(nn.Dropout(dropout_rate)), str(pool_layer)]
    want_remaining_layers = [20, 10]
    assert got_error is None
    assert [str(x) for x in got_conv_and_pooling_layers] ==want_conv_and_pooling_layers
    assert got_remaining_layers == want_remaining_layers


def test_build_convolution_layer():
    # Test if a malformed string was given
    got_layer, got_nr_filters, got_error = train_model_utils.build_convolution_layer(10, "malformed_string")
    assert got_error == "received_malformed_convolution_string_malformed_string"

    got_layer, got_nr_filters, got_error = train_model_utils.build_convolution_layer(10, "conv_10_387_29_378_38_26")
    assert got_error == "received_malformed_convolution_string_conv_10_387_29_378_38_26"

    # Test if the string cannot be separated as needed
    got_layer, got_nr_filters, got_error = train_model_utils.build_convolution_layer(10, "conv_10_i-am-no-int_29")
    assert got_error == "information_extraction_failed_from_convolution_string_conv_10_i-am-no-int_29"

    # Test if everything works as intended
    got_layer, got_nr_filters, got_error = train_model_utils.build_convolution_layer(10, "conv_10_5_3")
    assert got_error is None
    assert got_nr_filters == 10
    assert str(got_layer) == str(nn.Conv1d(10, 10, 5, 3))


def build_pooling_layer():
    # Test if a malformed string was given
    got_layer, got_error = train_model_utils.build_pooling_layer("malformed_string")
    assert got_error == "received_malformed_pooling_string_malformed_string"

    got_layer, got_error = train_model_utils.build_pooling_layer("pool_10_387_29_378_38_26")
    assert got_error == "received_malformed_pooling_string_pool_10_387_29_378_38_26"

    # Test if the string cannot be separated as needed
    got_layer, got_error = train_model_utils.build_pooling_layer("max_pool_i-am-no-int_29")
    assert got_error == f"information_extraction_failed_from_pooling_str_max_pool_i-am-no-int_29"

    # test if an invalid pooling method was given
    got_layer, got_error = train_model_utils.build_pooling_layer("invalid-pooling-method_pool_10_29")
    assert got_error == "invalid_pooling_method_given"

    # Test if everything works as intended
    got_layer, got_error = train_model_utils.build_pooling_layer("max_pool_2_2")
    assert got_error is None
    assert str(got_layer), str(nn.MaxPool1d(2, 2))


def test_create_loader():
    # test if the given dir does not exist
    minibatch_size = 1
    dataset = "testing"
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_create_loader")
    if os.path.exists(folder):
        shutil.rmtree(folder)

    got_loader, got_error = train_model_utils.create_loader(folder, minibatch_size, dataset, 1., "top")
    assert got_loader is None
    assert got_error == "given_dir_does_not_exist"

    # Test if the labels csv does not exist
    os.makedirs(folder)
    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    file_names = [os.path.join(folder, "1_some_file.csv"), os.path.join(folder, "1_some_file_1.csv"), os.path.join(folder, "2_some_file.csv"), os.path.join(folder, "2_some_file_1.csv")]
    pd.DataFrame({0: [1]}).to_csv(file_names[0], index=False)
    pd.DataFrame({0: [1]}).to_csv(file_names[1], index=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[2], index=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[3], index=False)

    got_loader, got_error = train_model_utils.create_loader(folder, minibatch_size, dataset, 1., "top")

    assert got_loader is None
    assert got_error == "no_labels_csv_in_given_dir"

    # test if everything works as intended
    pd.DataFrame({"filename": file_names, "label": [1, 1, 2, 2]}).to_csv(os.path.join(folder, "labels.csv"), index=False)

    got_loader, got_error = train_model_utils.create_loader(folder, minibatch_size, dataset, 1., "top")
    assert got_error is None

    # Test if the loader object gets returned correctly
    assert isinstance(got_loader, DataLoader)


def test_save_model():
    # test if everything works as intended
    got = train_model_utils.create_ffnn_model(False, 42)
    model = got(5, 1, [3], 0.0, False)
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_save_model")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    got = train_model_utils.save_model(model, folder)
    assert got is None
    assert os.path.isfile(os.path.join(folder, "model.pt"))


def test_display_confusion_matrix():
    # test if everything works as intended
    minibatch_size = 1
    dataset = "test_set"
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_display_confusion_matrix")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    file_names = [os.path.join(folder, "1_some_file.csv"), os.path.join(folder, "1_some_file_1.csv"), os.path.join(folder, "2_some_file.csv"), os.path.join(folder, "2_some_file_1.csv")]
    pd.DataFrame({0: [1]}).to_csv(file_names[0], index=False, header=False)
    pd.DataFrame({0: [1]}).to_csv(file_names[1], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[2], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[3], index=False, header=False)
    pd.DataFrame({"filename": file_names, "label": [1, 1, 2, 2]}).to_csv(os.path.join(folder, "labels.csv"), index=False)

    loader, error = train_model_utils.create_loader(folder, minibatch_size, dataset, 1, "top")
    assert error is None
    FFNN_Model = train_model_utils.create_ffnn_model(False, 42)
    model = FFNN_Model(1, 2, [1], 0.0, False)
    got_confusion_matr, got_accuracy, got_y_pred_joined = train_model_utils.display_confusion_matrix(loader, model, dataset, False, "")
    assert torch.equal(got_confusion_matr,torch.tensor([[0, 2], [0, 2]], dtype=torch.long))
    assert got_accuracy == 50.00


def test_evaluate_on_validation_data():
    # test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_evaluate_on_validation_data")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"balancing_over": {"label_depth": 2}, "train_model": {"seed": 42, "batch_normalization": False, "n_features": 1, "out_szs": 2, "dropout_rate": 0.0, "layer_structure": "1"}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    file_names = [os.path.join(folder, "1_some_file.csv"), os.path.join(folder, "1_some_file_1.csv"), os.path.join(folder, "2_some_file.csv"), os.path.join(folder, "2_some_file_1.csv")]

    pd.DataFrame({0: [1]}).to_csv(file_names[0], index=False, header=False)
    pd.DataFrame({0: [1]}).to_csv(file_names[1], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[2], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[3], index=False, header=False)
    pd.DataFrame({"filename": file_names, "label": [1, 1, 2, 2]}).to_csv(os.path.join(folder, "labels.csv"), index=False)
    minibatch_size = 1
    dataset = "test_utils"
    test_losses = []
    test_accuracies = []
    loader, got_error = train_model_utils.create_loader(folder, minibatch_size, dataset, 1, "top")
    assert got_error is None
    criterion = nn.CrossEntropyLoss()
    TabularModel = train_model_utils.create_ffnn_model(False, 42)
    model = TabularModel(1, 2, [1], 0.0, False)
    test_accuracies, test_losses, error = train_model_utils.evaluate_on_validation_data(model, folder, test_losses, test_accuracies, loader, criterion, 1., True, "minibatch", 1000, "FFNN", log_file)
    assert error is None
    assert [0.50] == test_accuracies
    assert [1.200448751449585] == test_losses


def test_Dataset_from_local_drive():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_Dataset_from_local_drive")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    file_names = [os.path.join(folder, "1_some_file.csv"), os.path.join(folder, "1_some_file_1.csv"), os.path.join(folder, "2_some_file.csv"), os.path.join(folder, "2_some_file_1.csv")]
    df_1 = pd.DataFrame([[1]])
    df_1.to_csv(file_names[0], index=False, header=None)
    pd.DataFrame([[1]]).to_csv(file_names[1], index=False, header=None)
    pd.DataFrame([[2]]).to_csv(file_names[2], index=False, header=None)
    pd.DataFrame([[2]]).to_csv(file_names[3], index=False, header=None)
    pd.DataFrame({"filename": file_names, "label": [1, 1, 2, 2]}).to_csv(os.path.join(folder, "labels.csv"), index=False)

    log_file = {"prepare_dataset": {"grid": False}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    got = train_model_utils.Dataset_from_local_drive(folder, "top")
    assert isinstance(got, TensorDataset)


def test_read_prepared_data_file():
    # Test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_read_prepared_data_file")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    df_1 = pd.DataFrame([[0., 1., 3.]])
    df_1.to_csv(os.path.join(folder, "1_some_file.csv"), index=False, header=False)
    got_tuple = train_model_utils.read_prepared_data_file(os.path.join(folder, "1_some_file.csv"))
    assert_frame_equal(got_tuple[0], df_1)
    assert got_tuple[1] == 1


def test_Dataset_from_RAM():
    # Test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_Dataset_from_RAM")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    file_names = [os.path.join(folder, "1_some_file.csv"), os.path.join(folder, "1_some_file_1.csv"), os.path.join(folder, "2_some_file.csv"), os.path.join(folder, "2_some_file_1.csv")]
    df_1 = pd.DataFrame({0: [1]})
    df_1.to_csv(file_names[0], index=False)
    pd.DataFrame({0: [1]}).to_csv(file_names[1], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[2], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[3], index=False, header=False)
    pd.DataFrame({"filename": file_names, "label": [1, 1, 2, 2]}).to_csv(os.path.join(folder, "labels.csv"), index=False)

    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    got_Dataset, got_error = train_model_utils.Dataset_from_RAM(folder, 1.5, "top")
    assert got_error is None
    assert isinstance(got_Dataset, TensorDataset)


def test_return_dataset():
    # Test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_return_dataset")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    file_names = [os.path.join(folder, "1_some_file.csv"), os.path.join(folder, "1_some_file_1.csv"), os.path.join(folder, "2_some_file.csv"), os.path.join(folder, "2_some_file_1.csv")]
    df_1 = pd.DataFrame({0: [1]})
    df_1.to_csv(file_names[0], index=False)
    pd.DataFrame({0: [1]}).to_csv(file_names[1], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[2], index=False, header=False)
    pd.DataFrame({0: [2]}).to_csv(file_names[3], index=False, header=False)
    pd.DataFrame({"filename": file_names, "label": [1, 1, 2, 2]}).to_csv(os.path.join(folder, "labels.csv"), index=False)

    got_Dataset, got_error = train_model_utils.return_dataset(folder, "testing", 1.0, "top")
    assert got_error is None
    assert isinstance(got_Dataset, TensorDataset)


def test_return_y_transformed():
    # test if the folder does not exist
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "generate_feature_utils_test", "test_return_y_transformed")
    if os.path.exists(folder):
        shutil.rmtree(folder)

    got_y_transformed, got_error = train_model_utils.return_y_transformed(folder)
    assert got_y_transformed is None
    assert got_error == "given_folder_does_not_exist"

    # Test if the folder does not contain a label.csv
    os.makedirs(folder)
    got_y_transformed, got_error = train_model_utils.return_y_transformed(folder)
    assert got_y_transformed is None
    assert got_error == "no_labels_csv_in_given_folder"

    # Test if the file does not contain a label col
    pd.DataFrame({"some_col": [1, 2, 3, 4], "another_col": [2, 3, 4, 5]}).to_csv(os.path.join(folder, "labels.csv"))
    got_y_transformed, got_error = train_model_utils.return_y_transformed(folder)
    assert got_y_transformed is None
    assert got_error == "labels.csv_does_not_contain_label_col"

    # Test if everything works as intended
    pd.DataFrame({"label": [1, 2, 1, 3, 1], "another_col": [2, 3, 4, 5, 6]}).to_csv(os.path.join(folder, "labels.csv"))
    got_y_transformed, got_error = train_model_utils.return_y_transformed(folder)
    assert torch.equal(got_y_transformed, torch.LongTensor([0, 1, 0, 2, 0]).reshape(-1,))
    assert got_error is None


def test_get_n_features():
    # test if the folder does not exist
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_get_n_features")
    if os.path.exists(folder):
        shutil.rmtree(folder)

    got_n_features, got_error = train_model_utils.get_n_features(folder, "top")
    assert got_n_features == 0
    assert got_error ==  "folder_does_not_exist"

    # Test if the folder does not contain all needed files
    os.makedirs(folder)
    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    pd.DataFrame({"label": [1, 2, 1, 3, 1], "another_col": [2, 3, 4, 5, 6]}).to_csv(os.path.join(folder, "labels.csv"))

    got_n_features, got_error = train_model_utils.get_n_features(folder, "top")
    assert got_n_features == 0
    assert got_error ==  "folder_does_not_contain_enough_files"

    # Test if everything works as intended
    pd.DataFrame([[2., 3, 4, 5, 6]]).T.to_csv(os.path.join(folder, "another_file.csv"), index=False, header=None)
    got_n_features, got_error = train_model_utils.get_n_features(folder, "top")
    assert got_n_features == 5
    assert got_error is None


def test_get_out_szs():
    # test if the folder does not exist
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_get_out_szs")
    if os.path.exists(folder):
        shutil.rmtree(folder)

    got_out_szs, got_error = train_model_utils.get_out_szs(folder, "top")
    assert got_out_szs == 0
    assert got_error ==  "folder_does_not_exist"

    # Test if the folder does not contain a labels.csv file
    os.makedirs(folder)
    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    pd.DataFrame([[2., 3, 4, 5, 6]]).to_csv(os.path.join(folder, "another_file.csv"), index=False)
    got_out_szs, got_error = train_model_utils.get_out_szs(folder, "top")
    assert got_out_szs == 0
    assert got_error == "labels.csv_does_not_exist"

    # Test if the labels.csv does not contain a label col
    pd.DataFrame({"another_col": [2., 3, 4, 5, 6]}).to_csv(os.path.join(folder, "labels.csv"), index=False)
    got_out_szs, got_error = train_model_utils.get_out_szs(folder, "top")
    assert got_out_szs == 0
    assert got_error == "labels.csv_does_not_contain_label_col"

    # Test if everything works as intended
    pd.DataFrame({"label": [2., 3, 4, 5, 6]}).to_csv(os.path.join(folder, "labels.csv"), index=False)
    got_out_szs, got_error = train_model_utils.get_out_szs(folder, "top")
    assert got_error is None
    assert got_out_szs == 5


def test_get_n_features_and_out_sz():
    # test if the folder does not exist
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_get_n_features_and_out_sz")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    got_n_features, got_out_szs, got_error = train_model_utils.get_n_features_and_out_sz(folder, "top")
    assert got_n_features == 0
    assert got_out_szs == 0
    assert got_error == "folder_does_not_exist"

    # Test if everything works as intended
    os.makedirs(folder)
    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    pd.DataFrame({"label": [2., 3, 4, 5, 6]}).to_csv(os.path.join(folder, "labels.csv"), index=False)
    pd.DataFrame([[2., 3, 4, 5, 6]]).T.to_csv(os.path.join(folder, "another_file.csv"), index=False, header=None)

    got_n_features, got_out_szs, got_error = train_model_utils.get_n_features_and_out_sz(folder, "top")
    assert got_error is None
    assert got_n_features == 5
    assert got_out_szs == 5


def test_reshape_tensor_to_first_identical_dim():
    # test if everything works as intended
    tensor_1 = torch.from_numpy(pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).to_numpy())
    tensor_2 = torch.from_numpy(pd.DataFrame([1, 2]).to_numpy())
    got = train_model_utils.reshape_tensor_to_first_identical_dim(tensor_1, tensor_2, 2)
    want = torch.from_numpy(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    assert torch.equal(got, want)


def test_valid_first_width_after_flattening():
    # Test if the conv_and_pool_layers contain incorrect entries
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_valid_first_width_after_flattening")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    pd.DataFrame([[1], [2], [3]]*10).T.to_csv(os.path.join(folder, "some_file.csv"), index=False)
    conv_and_pool_layers = ["unidentifiable_layer"]
    width_of_first_layer_after_flattening = 10
    got = train_model_utils.valid_first_width_after_flattening(folder, conv_and_pool_layers, width_of_first_layer_after_flattening)
    assert got == f"received_unidentifiable_layer_{conv_and_pool_layers[0]}"

    # Test if everything works as intended with only convolution  layers
    conv_and_pool_layers = ["conv_1_1_1"]
    width_of_first_layer_after_flattening = 30

    got = train_model_utils.valid_first_width_after_flattening(folder, conv_and_pool_layers, width_of_first_layer_after_flattening)
    assert got is None

    conv_and_pool_layers = ["conv_10_1_1"]
    width_of_first_layer_after_flattening = 300

    got = train_model_utils.valid_first_width_after_flattening(folder, conv_and_pool_layers, width_of_first_layer_after_flattening)
    assert got is None

    conv_and_pool_layers = ["conv_10_1_2"]
    width_of_first_layer_after_flattening = 150

    got = train_model_utils.valid_first_width_after_flattening(folder, conv_and_pool_layers, width_of_first_layer_after_flattening)
    assert got is None

    # test if everything works as intended with multiple conv layers
    conv_and_pool_layers = ["conv_10_1_2", "conv_3_5_2"]
    width_of_first_layer_after_flattening = 18
    got = train_model_utils.valid_first_width_after_flattening(folder, conv_and_pool_layers, width_of_first_layer_after_flattening)
    assert got is None

    # Test if the function predicts the correct output for pooling layers
    conv_and_pool_layers = ["max_pool_2_2"]
    width_of_first_layer_after_flattening = 15
    got = train_model_utils.valid_first_width_after_flattening(folder, conv_and_pool_layers, width_of_first_layer_after_flattening)
    assert got is None

    conv_and_pool_layers = ["max_pool_2_2", "max_pool_2_2"]
    width_of_first_layer_after_flattening = 7
    got = train_model_utils.valid_first_width_after_flattening(folder, conv_and_pool_layers, width_of_first_layer_after_flattening)
    assert got is None

    # test if it correctly predicts when pooling and conv layers are given
    # test if everything works as intended with multiple conv layers
    conv_and_pool_layers = ["conv_10_1_2", "max_pool_2_2", "conv_3_5_2", "max_pool_2_2", ]
    width_of_first_layer_after_flattening = 3
    got = train_model_utils.valid_first_width_after_flattening(folder, conv_and_pool_layers, width_of_first_layer_after_flattening)
    assert got is None


def test_relevant_part_of_truth():
    # test if everything works as intended
    input = torch.arange(0, 20, 1)
    want = torch.arange(0, 10, 1)
    got = train_model_utils.relevant_part_of_truth(10, input)
    assert torch.equal(got, want)


def test_check_needed_and_available_ram():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_check_needed_and_available_ram")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    [pd.DataFrame({"some_col": [100.13]*20}).to_csv(os.path.join(folder, f"{number}_something.csv")) for number in range(100)]

    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    got_available_ram, got_needed_ram, got_error = train_model_utils.check_needed_and_available_ram( folder, "top", 1, "testing_set")
    assert got_error is None

    assert isinstance(got_available_ram, float)
    assert got_available_ram > 0

    assert isinstance(got_needed_ram, float)
    assert got_available_ram > 0


def test_reduce_to_relevant_files_for_hierarchical_model():
    # test when an invalid hierarchical level was given
    got_list, got_error = train_model_utils.reduce_to_relevant_files_for_hierarchical_model([], "invalid_model")
    assert got_error == "invalid_hierarchical_model_given"
    assert got_list == []

    # test for top
    list_of_files = ["somewhere/11_something.csv", "somewhere/12_something.csv", "somewhere/11_something_2.csv", "somewhere/13_something.csv",
                     "somewhere/21_something.csv", "somewhere/22_something.csv", "somewhere/23_something.csv",
                     "somewhere/31_something.csv", "somewhere/1_something_2.csv", "somewhere/2_something.csv", "somewhere/1_something_else.csv"]
    want = ["somewhere/1_something_2.csv", "somewhere/2_something.csv", "somewhere/1_something_else.csv"]
    got_list, got_error = train_model_utils.reduce_to_relevant_files_for_hierarchical_model(list_of_files.copy(), "top")
    assert got_error is None
    assert got_list == want

    # test for lifting
    want = ["somewhere/11_something.csv", "somewhere/12_something.csv", "somewhere/11_something_2.csv", "somewhere/13_something.csv"]

    got_list, got_error = train_model_utils.reduce_to_relevant_files_for_hierarchical_model(list_of_files.copy(), "lifting")
    assert got_error is None
    assert got_list == want

    # test for walking
    want = ["somewhere/21_something.csv", "somewhere/22_something.csv", "somewhere/23_something.csv"]

    got_list, got_error = train_model_utils.reduce_to_relevant_files_for_hierarchical_model(list_of_files.copy(), "walking")
    assert got_error is None
    assert got_list == want


def test_processing_featured_hierarchical_data():
    # Test when the data does not exist in hierarchical featured format
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_processing_featured_hierarchical_data")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(
        log_file, os.path.join(folder, "log.json"))
    assert error is None

    result, got_error = train_model_utils.processing_featured_hierarchical_data(folder)
    assert got_error is None
    assert not result

    # Test when the data does exist in hierarchical featured format
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"balancing_over": {"label_depth": 2}, "generate_features": True}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    result, got_error = train_model_utils.processing_featured_hierarchical_data(folder)
    assert got_error is None
    assert result


def test_if_hierarchical_data_remove_unneeded_files():
    # Test if everything works as intended when the data is not hierarchical
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_if_hierarchical_data_remove_unneeded_files")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"balancing_over": {"label_depth": 2}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    list_of_files = ["somewhere/11_something.csv", "somewhere/12_something.csv",
                     "somewhere/11_something_2.csv", "somewhere/13_something.csv"]

    got_list, got_error = train_model_utils.if_hierarchical_data_remove_unneeded_files(list_of_files, folder, "top")
    assert got_error is None
    assert got_list == list_of_files

    # Test if everything works as intended when the data is hierarchical
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"balancing_over": {"label_depth": 2}, "generate_features": True}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    list_of_files = ["somewhere/11_something.csv", "somewhere/12_something.csv",
                     "somewhere/1_something_2.csv", "somewhere/2_something.csv"]
    want = ["somewhere/1_something_2.csv", "somewhere/2_something.csv"]
    got_list, got_error = train_model_utils.if_hierarchical_data_remove_unneeded_files(list_of_files, folder, "top")
    assert got_error is None
    assert got_list == want

    # Test for another hierarchical submodel

    want = ["somewhere/11_something.csv", "somewhere/12_something.csv"]
    got_list, got_error = train_model_utils.if_hierarchical_data_remove_unneeded_files(list_of_files, folder, "lifting")
    assert got_error is None
    assert got_list == want


def test_correct_out_szs_for_hierarchical_data():
    # Test if everything works as intended for top-level data
    label_series = pd.Series([11, 12, 11, 12, 13, 21, 22, 31, 1, 2, 3, 4, 5])
    assert train_model_utils.correct_out_szs_for_hierarchical_data(label_series, "top") == 5

    # Test if everything works as intended for lifting data
    assert train_model_utils.correct_out_szs_for_hierarchical_data(label_series, "lifting") == 3

    # Test if everything works as intended for walking data
    assert train_model_utils.correct_out_szs_for_hierarchical_data(label_series, "walking") == 2


def test_find_n_features_for_hierarchical_data():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_find_n_features_for_hierarchical_data")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    pd.DataFrame([[1], [2], [3]]).to_csv(os.path.join(folder, "1_something.csv"), index=False, header=None)
    pd.DataFrame([[1], [2], [3], [4]]).to_csv(os.path.join(folder, "11_something.csv"), index=False, header=None)
    pd.DataFrame([[1], [2], [3], [4], [5]]).to_csv(os.path.join(folder, "21_something.csv"), index=False, header=None)

    # test for top
    result, error = train_model_utils.find_n_features_for_hierarchical_data(folder, "top")
    assert error is None
    assert result == 3

    # test for lifting
    result, error = train_model_utils.find_n_features_for_hierarchical_data(folder, "lifting")
    assert error is None
    assert result == 4

    # test for walking
    result, error = train_model_utils.find_n_features_for_hierarchical_data(folder, "walking")
    assert error is None
    assert result == 5


def test_load_needed_params():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_load_needed_params")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # Wrong seed type
    error = file_utils.save_json({"train_model": {"seed": "False"}}, os.path.join(folder, "log.json"))
    assert error is None

    seed, batch_normalization, n_features, out_szs, dropout_rate, layers, error = train_model_utils.load_needed_params(folder, "FFNN")
    assert error == "received_non_int_seed"

    # Wrong batch_normalization type
    error = file_utils.save_json({"train_model": {"seed": 42, "batch_normalization": 42}}, os.path.join(folder, "log.json"))
    assert error is None

    seed, batch_normalization, n_features, out_szs, dropout_rate, layers, error = train_model_utils.load_needed_params(folder, "FFNN")
    assert error ==  "received_non_bool_batch_normalization"

    # Wrong n_features type
    error = file_utils.save_json({"train_model": {"seed": 42, "batch_normalization": True, "n_features": [15]}}, os.path.join(folder, "log.json"))
    assert error is None

    seed, batch_normalization, n_features, out_szs, dropout_rate, layers, error = train_model_utils.load_needed_params(folder, "FFNN")
    assert error == "received_non_int_n_features"

    # Wrong out_szs type
    error = file_utils.save_json({"train_model": {"seed": 42, "batch_normalization": True, "n_features": 15, "out_szs": ">sdf"}}, os.path.join(folder, "log.json"))
    assert error is None

    seed, batch_normalization, n_features, out_szs, dropout_rate, layers, error = train_model_utils.load_needed_params(folder, "FFNN")
    assert error == "received_non_int_out_szs"

    # Wrong dropout_rate type
    error = file_utils.save_json({"train_model": {"seed": 42, "batch_normalization": True,
                                  "n_features": 15, "out_szs": 4, "dropout_rate": "sdsdf"}}, os.path.join(folder, "log.json"))
    assert error is None

    seed, batch_normalization, n_features, out_szs, dropout_rate, layers, error = train_model_utils.load_needed_params(folder, "FFNN")
    assert error == "received_non_float_dropout_rate"

    # Everything is valid
    error = file_utils.save_json({"train_model": {"seed": 42, "batch_normalization": True, "n_features": 15,
                                  "out_szs": 4, "dropout_rate": 0.1, "layer_structure": "80f30"}}, os.path.join(folder, "log.json"))
    assert error is None

    seed, batch_normalization, n_features, out_szs, dropout_rate, layers, error = train_model_utils.load_needed_params(folder, "FFNN")
    assert error is None
    assert 0.1 == dropout_rate

def test_load_model():
    # test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_load_model")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    log_file = {"train_model": {"seed": 42, "batch_normalization": True,
                                "n_features": 15, "out_szs": 4, "dropout_rate": 0.1, "layer_structure": "80f30"}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    # Save a model in the folder be loaded
    layers, error = train_model_utils.build_layers_from_string(log_file.get("train_model").get("layer_structure"), "FFNN")
    assert error is None

    Model = train_model_utils.build_nn_model("FFNN", log_file.get("train_model").get("batch_normalization"), log_file.get("train_model").get("seed"), {})
    model = Model(log_file.get("train_model").get("n_features"), log_file.get("train_model").get("out_szs"), layers, log_file.get("train_model").get("dropout_rate"), log_file.get("train_model").get("batch_normalization"))

    error = train_model_utils.save_model(model, folder)
    assert error is None

    got_model, got_error = train_model_utils.load_model(folder, "FFNN", {})
    assert got_error is None
    assert str(type(got_model)) == "<class 'utils.train_model_utils.create_ffnn_model.<locals>.FeedforwardNetwork'>"


def test_valid_hidden_size():
    # Test when the layerstring cannot be used to build  a valid hiddensize
    got = train_model_utils.valid_hidden_size("not_valid_hidden_sizef45")
    assert got == "invalid_hidden_size_given"

    # test when the min layer is too small
    got = train_model_utils.valid_hidden_size("0f45")
    assert got == "given_hidden_size_too_small"

    # test when one layer width is too big
    got = train_model_utils.valid_hidden_size("30000000f45")
    assert got == "too_much_hidden_units_given"

    # Test when everything is valid
    got = train_model_utils.valid_hidden_size("50f30")
    assert got is None


def test_determine_hidden_size():
    # Test when the layers list does not contain any entries
    got_size, got_error = train_model_utils.determine_hidden_size([])
    assert got_error == "invalid_layers_given"

    # Test when the hidden size is too small
    got_size, got_error = train_model_utils.determine_hidden_size([-100, 150])
    assert got_error == "given_hidden_size_too_small"

    # Test when the hidden size is too big
    got_size, got_error = train_model_utils.determine_hidden_size([1000000000, 150])
    assert got_error == "too_much_hidden_units_given"

    # Test when everything is valid
    got_size, got_error = train_model_utils.determine_hidden_size([100, 150])
    assert got_size == 100
    assert got_error is None


def test_load_model_if_training_was_interrupted():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_load_model_if_training_was_interrupted")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # Test if everything works as intended when the training was interrupted
    log_file = {"train_model": {"seed": 42, "batch_normalization": True, "n_features": 15, "out_szs": 4, "dropout_rate": 0.1,
                                "layer_structure": "80f30", "epochs": 10, "already_trained_epochs": 4, "saving_folder": folder, "type": "FFNN"}}
    error = file_utils.save_json(log_file, os.path.join(folder, "log.json"))
    assert error is None

    # Save a model in the folder be loaded
    layers, error = train_model_utils.build_layers_from_string(log_file.get("train_model").get("layer_structure"), "FFNN")
    assert error is None

    Model = train_model_utils.build_nn_model("FFNN", log_file.get("train_model").get("batch_normalization"), log_file.get("train_model").get("seed"), {})
    model = Model(log_file.get("train_model").get("n_features"), log_file.get("train_model").get("out_szs"), layers, log_file.get("train_model").get("dropout_rate"), log_file.get("train_model").get("batch_normalization"))

    error = train_model_utils.save_model(model, folder)
    assert error is None

    want_model_total_params = sum(p.numel() for p in model.parameters())

    input_model = Model(log_file.get("train_model").get("n_features"), log_file.get("train_model").get("out_szs"), [20, 10], log_file.get("train_model").get("dropout_rate"), log_file.get("train_model").get("batch_normalization"))

    got_model, got_log, got_already_trained_epochs, got_error = train_model_utils.load_model_if_training_was_interrupted(log_file, input_model)
    assert got_error is None
    assert got_already_trained_epochs == 4

    got_total_params = sum(p.numel() for p in got_model.parameters())
    assert got_total_params == want_model_total_params


def test_load_svm_from_pkl():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_load_svm_from_pkl")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # create a simple SVM to save
    x = np.arange(30).reshape(3, 10)
    y = np.array([0, 1, 2]).astype(np.int64)
    model = SVC()
    model.fit(x, y)

    svm_location = os.path.join(folder, "simple_svm.pkl")
    dump(model, svm_location)

    assert os.path.isfile(svm_location)

    got_model = train_model_utils.load_svm_from_pkl(svm_location)
    assert isinstance(got_model, SVC)
    assert np.array_equiv(model.classes_, got_model.classes_)
    assert np.array_equiv(model.support_vectors_, got_model.support_vectors_)


def test_find_correct_svm_folder():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_find_correct_svm_folder")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # Test when the folder does not contain a log file
    got_svm_folder, got_error = train_model_utils.find_correct_svm_folder(folder, "top")
    assert got_error == "no_log_file_found"

    # Create a valid log file
    dicto = {"create_windows": {"data_path": folder}}
    full_filename = os.path.join(folder, "log.json")
    with open(full_filename, 'w') as f:
        json.dump(dicto, f)

    got_svm_folder, got_error = train_model_utils.find_correct_svm_folder(folder, "top")
    assert got_error is None
    want_svm_folder = os.path.join(folder, "train_model", "SVM", "top")
    assert got_svm_folder == want_svm_folder


def test_standardize_before_concatting():
    # test if everything works as intended, when no errors occur
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_standardize_before_concatting")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # Create the log file leading to the standardization file
    dicto = {"create_windows": {"data_path": folder}}
    full_filename = os.path.join(folder, "log.json")
    with open(full_filename, 'w') as f:
        json.dump(dicto, f)

    std_folder = os.path.join(folder, "train_model", "SVM", "lifting")
    if os.path.exists(std_folder):
        shutil.rmtree(std_folder)
    os.makedirs(std_folder)

    # create a df with the standardizing parameters in the wanted location
    standardizing_df_filename = os.path.join(std_folder, "standardizing_std.csv")
    standardizing_df_content = {f"some_col_{x}": [2, 3, 3] for x in range(7)}
    standardizing_df = pd.DataFrame(standardizing_df_content, index=["mean", "scale", "var"])
    standardizing_df.to_csv(standardizing_df_filename)

    # create some dfs_ to standardize
    list_of_dfs = [pd.DataFrame([[5]*7]), pd.DataFrame([[5]*7])]

    hierarchical_model = "lifting"
    got_df, error = train_model_utils.standardize_before_concatting(list_of_dfs, folder, hierarchical_model)
    assert error is None
    want = pd.DataFrame([(5-2)/3]*14)
    want.index = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
    assert_frame_equal(got_df, want)


def test_early_stopping():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "train_model_utils_test", "test_early_stopping")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # test if an empty list was given
    Model = train_model_utils.build_nn_model("FFNN", False, 42, {})
    model = Model(5, 1, [3], 0.0, False)

    got = train_model_utils.early_stopping([],folder, model, True)
    assert got == "list_of_accuracies_empty"

    # test if the current state of the model is not the optimal state
    got = train_model_utils.early_stopping([0.1,0.2,0.005],folder, model, True)
    assert got is None
    assert not os.path.exists(os.path.join(folder, "best_val_model.pt"))

    # Test if everything works as intended when the current state of the model is the best yet
    got = train_model_utils.early_stopping([0.1,0.2,0.3],folder, model, True)

    assert got is None
    assert os.path.isfile(os.path.join(folder, "best_val_model.pt"))
