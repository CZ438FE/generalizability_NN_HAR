import os
import sys
import psutil
import math
import json
import copy
import joblib
import pandas as pd
import numpy as np
from multiprocessing import Pool
import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix
from torch.utils.data import TensorDataset, DataLoader
from config.log import logger
from utils import file_utils, time_utils, plot_utils, data_utils


def valid_table(table: dict):
    """Checks if the arguments in the table are valid
    """
    if table.get("dryrun"):
        logger.warning("Storing results of training is highly recommended")

    # Check the folder of trainings_data
    if not os.path.exists(table.get("training_data")):
        logger.error(f"Given folder with training_data does not exist: {table.get('training_data')}")
        return "training_data_folder_does_not_exist"

    _, error = file_utils.get_files(table.get("training_data"), "csv")
    if error:
        return error

    # Check the folder of validation_data
    if table.get("validation_data") is not None:
        if not os.path.exists(table.get("validation_data")):
            logger.error(f"Given folder with validation data does not exist {table.get('validation_data')}")
            return "validation_data_folder_does_not_exist"

        _, error = file_utils.get_files(table.get('validation_data'), "csv")
        if error:
            return error

    if table.get("validation_data") == table.get("training_data"):
        logger.error(f"Cannot use the same files as trainings-and validation data")
        return "validation_data_folder_identical_to_trainings_data_folder"

    # Check the output date
    # If a output date was given, check if has the correct format (only relevant if there is anything saved there)
    error = time_utils.valid_time_string(table.get("output_date"))
    if error:
        return error

    if table.get("output_date"):
        location_future_log_file = os.path.join(table.get("data_path"), "train_model", table.get("type"), table.get("output_date"), "log.json")
        if os.path.exists(location_future_log_file):
            with open(location_future_log_file) as f:
                log_file = json.load(f)
            if "already_trained_epochs" in log_file.get("train_model").keys():
                if log_file.get("train_model").get("already_trained_epochs") >= table.get("epochs"):

                    logger.error("Tried to train a model after training was already finished")
                    return "training_already_finished"

    if table.get("minibatch_size") < 1:
        logger.error(f"Minibatchsizes < 1 are not allowed. please choose a bigger minibatch_size ")
        return "minibatch_size_too_small"

    if table.get("epochs") < 0:
        logger.error(f"epochs < 0 are not allowed. please choose a bigger epochs ")
        return "epoch_amount_too_small"

    # Check the loss function
    # as there is already a check implemented by the choices, no further checking is needed here

    # Check the optimizer
    # as there is already a check implemented by the choices, no further checking is needed here

    # Check the layer_structure
    error = valid_layerstring(table.get("layer_structure"), table.get("type"), table.get("training_data"))
    if error:
        return error

    # Check the dropout rate
    if table.get("dropout_rate") is not None:
        if not ((table.get("dropout_rate") >= 0) & (table.get("dropout_rate") < 1)):
            logger.error(f"Given dropout rate {table['dropout_rate']} is not within the allowed interval ]0,1[")
            return "dropout_rate_out_of_allowed_boundaries"

    if not ((table.get("learning_rate") > 0) & (table.get("learning_rate") < 1)):
        logger.error(f"Given learning_rate {table['learning_rate']} is not within the allowed interval ]0,1[")
        return "learning_rate_out_of_allowed_boundaries"

    # Check the number of minibatches
    if table.get("number_minibatches") is not None:
        if table.get("number_minibatches") < 1:
            logger.error(f"received a number of minibatches to train for < 1")
            return "nr_minibatches_too_small"

    return None


def valid_layerstring(layerstring: str, network_type: str, folder: str):
    """Checks if the given layer string may be transformed to a valid structure of the network
    """

    if network_type == "FFNN":
        error = valid_feedforward_layers(layerstring.split("f"))
        return error
    if network_type == "CNN":
        conv_and_pool_layers, ff_layers, error = split_layerstring_into_conv_and_ff_layers(layerstring)
        if error:
            return error

        error = valid_conv_and_pool_layers(conv_and_pool_layers)
        if error:
            return error

        error = valid_feedforward_layers(ff_layers)
        if error:
            return error

        error = valid_first_width_after_flattening(
            folder, conv_and_pool_layers, ff_layers[0])
        return error

    if network_type == "RNN":
        error = valid_hidden_size(layerstring)
        return error

    logger.error(f"String structure for other network types not implemented yet")
    return "creating_layerstructure_for_given_type_not_implemented"


def valid_feedforward_layers(layers: list):
    """test if the list of FF layers has valid values
    """
    try:
        layer_widths = [int(x) for x in layers]
    except ValueError:
        logger.error(f"Could not transform given layers {layers} into a valid format")
        return "layerstring_malformed"

    if min(layer_widths) < 1:
        logger.error(f"received layer with invalid width. Please choose higher width")
        return "too_small_width_detected"
    if min(layer_widths) < 5:
        logger.warning(f"Using very thin layers is typically not that useful. received hidden layer with {min(layer_widths)} Units")
    if max(layer_widths) > 2000:
        logger.error(f"Detected layer of unreasonable width: {max(layer_widths)}. Using thiner, yet a longer cascade of layers is probably better suited")
        return "too_great_width_detected"

    return None


def valid_conv_and_pool_layers(conv_and_pool_layers: list):
    """checks if the individual convolution and pooling layers in the list have valid parameters
    """
    for layer in conv_and_pool_layers:
        if "conv" in layer:
            error = valid_convolution_layer(layer)
            if error:
                return error

            continue

        if "pool" in layer:
            error = valid_pooling_layer(layer)
            if error:
                return error

            continue

        logger.error(f"received layer which is neither a convolution nor a pooling layer: {layer}")
        return "received_layer_which_is_neither_conv_nor_pool"

    return None


def valid_pooling_layer(layer: str):
    """tests if the given layer str may be used to build a pooling layer
    """
    if not isinstance(layer, str):
        logger.error(f"Got layer of non-str type : {layer}")
        return "got_layer_of_non_str_dt"

    pooling_str_splitted = layer.split("_")

    if len(pooling_str_splitted) != 4:
        logger.error(f"Invalid format for convolution layer given. Please use format <pooling-method>_pool_<kernel-size>_<stride> \n Got: {layer}")
        return "got_malformed_pool_str"

    try:
        pooling_method = pooling_str_splitted[0]
        kernel_size = int(pooling_str_splitted[2])
        stride = int(pooling_str_splitted[3])
    except ValueError:
        logger.error(f"Invalid format for pooling layer given. Please use format <pooling-method>_pool_<kernel-size>_<stride> \n Got: {layer}")
        return "got_malformed_pool_str"

    valid_pooling_methods = ["max", "mean"]
    if pooling_method not in valid_pooling_methods:
        logger.error(f"received invalid pooling method {pooling_method}. Please choose from {valid_pooling_methods}")
        return "received_invalid_pooling_method"

    error = valid_kernel_size_and_stride(kernel_size, stride)
    if error:
        return error

    return None


def valid_convolution_layer(layer: str):
    """tests, if the given layer may be used to build a convolution layer
    """
    if not isinstance(layer, str):
        logger.error(f"Got layer of non-str type : {layer}")
        return "got_layer_of_non_str_dt"

    convolution_string_splitted = layer.split("_")
    if len(convolution_string_splitted) != 4:
        logger.error(f"Invalid format for convolution layer given. Please use format conv_<nr_filters>_<kernel_size>_<stride> \n Got: {layer}")
        return "got_malformed_conv_str"

    try:
        nr_filters = int(convolution_string_splitted[1])
        kernel_size = int(convolution_string_splitted[2])
        stride = int(convolution_string_splitted[3])
    except ValueError:
        logger.error(f"Invalid format for convolution layer given. Please use format conv_<nr_filters>_<kernel_size>_<stride> \n Got: {layer}")
        return "got_malformed_conv_str"

    error = valid_amount_of_convolution_filters(nr_filters=nr_filters)

    error = valid_kernel_size_and_stride(kernel_size, stride)
    if error:
        return error

    return None


def valid_amount_of_convolution_filters(nr_filters: int):
    """tests if the given amount of convolution filters is valid 
    """
    if nr_filters < 1:
        logger.error(f"received convolution layer with too small amount of filters : {nr_filters}")
        return "received_too_small_amount_of_filters"
    if nr_filters > 50:
        logger.warning(f"Used suspiciously large amount of filters: {nr_filters}")
    if nr_filters > 100:
        logger.error(f"Amount of given filters for convolution layer is too big: {nr_filters}")
        return "received_too_big_amount_of_filters"

    return None


def valid_kernel_size_and_stride(kernel_size: int, stride: int):
    """Checks if the given kernel size and stride are valid
    """
    if kernel_size < 1:
        logger.error(f"Kernel size is too small")
        return "received_too_small_kernel_size"
    if kernel_size > 200:
        logger.warning("received huge kernel size")
    if kernel_size > 500:
        logger.error(f"Kernel size is too big")
        return "received_too_big_kernel_size"

    if stride < 1:
        logger.error(f"stride for is too small")
        return "received_too_small_stride"
    if stride > kernel_size:
        logger.error(f"stride for is bigger than kernel_size")
        return "got_stride_bigger_than_kernel_size"

    return None


def split_layerstring_into_conv_and_ff_layers(layerstring: str):
    """returns a list of all the convolution and pooling layers and a list of all the feedforward layers
    """
    layerstring_splitted = layerstring.split("f")
    conv_and_pool_layers = []
    ff_layers = []
    for layer in layerstring_splitted:
        if "conv" in str(layer) or "pool" in str(layer):
            conv_and_pool_layers.append(layer)
            continue

        try:
            int(layer)
        except ValueError:
            logger.error(f"Could not interpret layer {layer}.  Please use format conv_<nr_filters>_<kernel_size>_<stride>")
            return [], [], f"could_not_interpret_layer_{layer}"
        ff_layers.append(layer)

    if conv_and_pool_layers == []:
        logger.error(f"received CNN Model without any convolution or pooling layer specified")
        return [], [], "no_conv_or_pool_layer_given"

    if ff_layers == []:
        logger.error(f"received CNN Model without any FF layer specified")
        return [], [], "no_ff_layer_given"

    return conv_and_pool_layers, ff_layers, None


def build_layers_from_string(layerstring: str, network_type: str):
    """build a list of layer-widths from the layerstring based on the type of network given
    """
    if network_type == "FFNN":
        return [int(x) for x in layerstring.split("f")], None

    if network_type == "CNN":
        return [x for x in layerstring.split("f")], None

    if network_type == "RNN":
        return [int(x) for x in layerstring.split("f")], None

    logger.error(f"Calculation of the number and structure of other network types implemented yet")
    return [], "building_layers_from_string_not_implemented_for_other_types"


def set_loss_function(loss_function: str):
    """uses the given loss_function_string to return the wanted loss object
    """
    # For Regression tasks
    if loss_function == "MAE":
        return nn.L1Loss(), None
    if loss_function == "MSE":
        return nn.MSELoss(), None
    # For (Multi-class) classification tasks
    if loss_function == "CEL":
        return nn.CrossEntropyLoss(), None
    if loss_function == "NLL":
        return nn.NLLLoss(), None

    logger.error(f"Received Loss function which is not implemented yet")
    return None, "loss_function_not_implemented_yet"


def set_optimizer(optimizer: str, learning_rate: float, model: nn.Module):
    """return an optimizer with the initial learning rate
    """
    if (learning_rate > 5) or (learning_rate < 0):
        logger.error(f"received invalid learning rate if {learning_rate}. Please choose in [0,5]")
        return None, "learning_rate_out_of_allowed_boundaries"

    if learning_rate > 1:
        logger.warning(f"Choosing a high learning rate might prevent the model from learning at all. A learning rate [0,1] is recommended. Got: {learning_rate}")

    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate), None

    if optimizer == "adadelta":
        return torch.optim.adadelta(model.parameters(), lr=learning_rate), None

    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate), None

    logger.error(f"No other optimizers have been implemented yet.")
    return None, "given_optimizer_not_implemented_yet"


def find_feature_columns(df_columns: list):
    """returns a list of feature_colnames of the given df (all cols without time or label)
    """
    return [x for x in df_columns if x not in ["time", "time_start", "time_end", "label"]]


def bring_to_correct_format(y_batch: torch.Tensor):
    """ brings the target values of a given batch into the correct format for the CEL Loss
    """
    return y_batch.to(torch.long).reshape(-1,) - torch.ones(len(y_batch)).to(torch.long)


def transform_predictions_for_confusion_matrix(predictions_tensor: torch.Tensor):
    """As the confusion matrix functionality expects the predictions_tensor in a different format, convert it accordingly 
    """
    return torch.argmax(predictions_tensor, 1).to(torch.int32).reshape(-1,)


def sum_identical_entries(tensor_1: torch.Tensor, tensor_2: torch.Tensor):
    """Returns how often both given tensors have identical values on a position
    """
    return torch.eq(tensor_1, tensor_2).sum()


def save_model(model: nn.Module, saving_folder: str, model_name = "model.pt"):
    """stores the model and the params in the given data_path
    """
    # Save the results in <dp>/train_model/<network_type>/<saving_time>
    full_model_path = os.path.join(saving_folder, model_name)

    torch.save(model.state_dict(), full_model_path)
    if not os.path.exists(full_model_path):
        logger.error(f"Could not find saved model file in location {full_model_path}")
        return "model_does_not_exist_after_saving"

    return None


def display_confusion_matrix(loader: DataLoader, model: nn.Module, data_set: str, store_local: bool, saving_folder: str, verbose=False, frac=1.0, hierarchical_model="top"):
    """Displays the confusion matrix after aggregating the evaluation of all data from the loader
    """

    model = model.eval()

    with torch.no_grad():
        for j, minibatch in enumerate(loader):
            if j/len(loader) > frac:
                break

            y_pred = model.predict(minibatch[0])
            # transform the ground truth for this batch into an appropriate format for confusion matrix
            y_pred_transformed = transform_predictions_for_confusion_matrix(
                y_pred)

            if j == 0:
                y_pred_joined = y_pred_transformed
                y_truth = bring_to_correct_format(minibatch[1])
            else:
                y_pred_joined = torch.cat((y_pred_joined, y_pred_transformed), 0)
                y_truth = torch.cat((y_truth, bring_to_correct_format(minibatch[1])), 0)

        # Create the final confusion matrix of the data
        task = "binary" if len(torch.unique(y_truth)) == 2 else "multiclass"
        conf_mat = ConfusionMatrix(num_classes = len(torch.unique(y_truth)), task = task)

        confusion_matrix = conf_mat(y_pred_joined, y_truth)

        accuracy = round((torch.diagonal(confusion_matrix, 0).sum().item()*100)/(torch.sum(confusion_matrix)).item(), 2)

        if verbose:
            logger.info(f"This is the Confusion Matrix of the {data_set}-data after Training: (n = {torch.sum(confusion_matrix).item()})\n{confusion_matrix}")
        logger.info(f"The final accuracy on {round(frac*100, 2)}% of the {data_set} data is {accuracy}%")

        plot_utils.plot_confusion_matrix(confusion_matrix, store_local, saving_folder, data_set, hierarchical_model)

        return confusion_matrix, accuracy, y_pred_joined


def update_validation_accuracies_and_losses(validation_losses: list, validation_accuracies: list, validationloader, criterion, model: nn.Module, frac=1.0):
    """For each minibatch, the loss and accuracy over (a fraction of) the validation set is calculated
    """
    validation_criterion = copy.deepcopy(criterion)
    validation_model = model.eval()

    with torch.no_grad():
        validation_losses_epoch = []
        validation_accuracies_epoch = []
        for j, validation_minibatch in enumerate(validationloader):
            # Enable skipping a part of the evaluation of the validation_df
            if j/len(validationloader) > frac:
                break

            # Calculate the loss on the validation-data
            y_pred_validation = validation_model.predict(validation_minibatch[0].double())
            # transform the ground truth for this batch into an appropriate format for the loss function
            y_truth_validation = bring_to_correct_format(validation_minibatch[1])

            loss_validation_minbatch = validation_criterion(y_pred_validation, y_truth_validation)
            validation_losses_epoch.append(loss_validation_minbatch)

            # The following is a quick way to estimate the accuracy of the model on the validation data
            y_pred_validation_transformed = transform_predictions_for_confusion_matrix(y_pred_validation)

            accuracy_batch = sum_identical_entries(y_truth_validation, y_pred_validation_transformed).item() / len(validation_minibatch[1])
            validation_accuracies_epoch.append(accuracy_batch)

    average_validation_data_loss = sum(validation_losses_epoch)/len(validation_losses_epoch)
    validation_losses.append(average_validation_data_loss) if isinstance(average_validation_data_loss, float) else validation_losses.append(average_validation_data_loss.item())

    average_validation_accuracy = sum(validation_accuracies_epoch)/(len(validation_accuracies_epoch))
    validation_accuracies.append(average_validation_accuracy)

    validation_model = validation_model.train()
    del validation_model
    validation_criterion

    return validation_accuracies,  validation_losses


def create_ffnn_model(batch_normalization: bool, seed: int):
    """ returns a FFNN model
    """
    torch.manual_seed(seed)

    class FeedforwardNetwork(nn.Module):
        def __init__(self, n_features: int, out_sz: int, layers: list, dropout_rate: float, batch_normalization: bool):
            super().__init__()
            if batch_normalization:
                self.bn_cont = nn.BatchNorm1d(n_features)

            # Create a list of layers with batch-normalization and dropout, if specified
            layerlist = build_fully_connected_layers(n_features, [int(x) for x in layers], batch_normalization, dropout_rate, out_sz)

            self.layers = nn.Sequential(*layerlist)

        def predict(self, x: torch.FloatTensor):
            if batch_normalization:
                x = self.bn_cont(x.float())
            if x.shape[-1] == 1:
                x = x.float().reshape((-1, x.shape[1]))
            return self.layers(x.float())

    return FeedforwardNetwork


def build_nn_model(network_type: str, batch_normalization: bool, seed: int, log_file: dict):
    """returns the class model of the specified type
    """
    if network_type == "FFNN":
        return create_ffnn_model(batch_normalization, seed)
    if network_type == "CNN":
        return create_convolutional_model(batch_normalization, seed)
    return create_rnn_model(batch_normalization, seed, log_file)


def build_fully_connected_layers(n_features: int, layers: list, batch_normalization: bool, dropout_rate: float, out_sz: int):
    """ returns a list of fully connected ReLU layers (with dropout and batch-normalization, if specified)
    """
    n_in = n_features
    layerlist = []
    for layer in layers:
        if "conv" in str(layer) or "pool" in str(layer):
            continue

        layerlist.append(nn.Linear(n_in, int(layer)))
        layerlist.append(nn.ReLU(inplace=True))
        if batch_normalization:
            layerlist.append(nn.BatchNorm1d(int(layer)))
        if dropout_rate != 0:
            layerlist.append(nn.Dropout(dropout_rate))
        n_in = int(layer)

    # when no hidden layers are given, the flattened layer is connected directly to the output neurons
    if layers == []:
        layers = [n_features]
    layerlist.append(nn.Linear(int(layers[-1]), out_sz))
    return layerlist


def create_convolutional_model(batch_normalization: bool, seed: int):
    """ returns a convolutional network
    """
    torch.manual_seed(seed)

    class ConvolutionalNetwork(nn.Module):
        def __init__(self, n_features: int, out_sz: int, layers: list, dropout_rate: float, batch_normalization: bool):
            super().__init__()
            conv_and_pooling_layers, remaining_layers, error = build_conv_and_pool_layers(
                layers, n_features, dropout_rate, batch_normalization)
            if error:
                logger.error(f"Could not build convolution and pooling layers, adjusting the checks in valid_table() is needed")
                return
            self.convolution_and_pooling = nn.Sequential(*conv_and_pooling_layers)

            # After the convolutions and poolings the "n_features" equals the len of the flattened output
            n_features = remaining_layers[0]
            remaining_layers.remove(remaining_layers[0])
            fully_connected_layers = build_fully_connected_layers(n_features, remaining_layers, batch_normalization, dropout_rate, out_sz)
            self.fully_connected_layers = nn.Sequential(*fully_connected_layers)
            # Create the structure of the model from the layers

        def predict(self, x: torch.FloatTensor):
            x = self.convolution_and_pooling(x.float())
            # flatten the data
            x = x.view(-1, x.shape[1]*x.shape[2])
            return self.fully_connected_layers(x.float())

    return ConvolutionalNetwork


def build_conv_and_pool_layers(layers: list, n_features: int, dropout_rate: float, batch_normalization: bool):
    """build a list of convolution and pooling layers based on the given params and returns the remaining layers as a list
    """
    n_in = n_features
    processed_layers = []
    conv_and_pooling_layers = []

    for layer_number, layer in enumerate(layers):
        if "conv" in layer:
            conv_layer, n_in, error = build_convolution_layer(n_in, layer)
            if error:
                return None, None, error
            conv_and_pooling_layers.append(conv_layer)
            conv_and_pooling_layers.append(nn.ReLU(inplace=True))

            if dropout_rate != 0:
                conv_and_pooling_layers.append(nn.Dropout(dropout_rate))

            processed_layers.append(layer)
            continue

        elif "pool" in layer:
            pool_layer, error = build_pooling_layer(layer)
            if error:
                return None, None, error
            conv_and_pooling_layers.append(pool_layer)

            processed_layers.append(layer)
            continue

        break

    [layers.remove(processed_layer) for processed_layer in processed_layers]

    return conv_and_pooling_layers, [int(x) for x in layers], None


def build_convolution_layer(n_in: int, convolution_string: str):
    """returns a 1D convolution layer based on the given string
    """
    if "conv" not in convolution_string:
        logger.error("received malformed convolution_string. Please use format conv_<nr_filters>_<kernel_size>_<stride>")
        return None, None, f"received_malformed_convolution_string_{convolution_string}"

    if convolution_string.count("_") != 3:
        logger.error(f"received malformed convolution_string. Please use format conv_<nr_filters>_<kernel_size>_<stride> \n Got: {convolution_string}")
        return None, None, f"received_malformed_convolution_string_{convolution_string}"

    convolution_string_splitted = convolution_string.split("_")

    try:
        nr_filters = int(convolution_string_splitted[1])
        kernel_size = int(convolution_string_splitted[2])
        stride = int(convolution_string_splitted[3])
    except ValueError:
        logger.error(f"Extraction of needed Information from convolution_string failed. Please use format conv_<nr_filters>_<kernel_size>_<stride> . Got {convolution_string}")
        return None, None, f"information_extraction_failed_from_convolution_string_{convolution_string}"

    return nn.Conv1d(n_in, nr_filters, kernel_size, stride), nr_filters, None


def build_pooling_layer(pooling_str: str):
    """returns a 1D pooling layer based on the given string
    """
    if "pool" not in pooling_str:
        logger.error(f"received invalid pooling string {pooling_str}")
        return None, f"received_malformed_pooling_str_{pooling_str}"

    if pooling_str.count("_") != 3:
        logger.error(f"received malformed pooling_str. Please use format  <pooling-method>_pool_<kernel_size>_<stride>")
        return None, f"received_malformed_pooling_str_{pooling_str}"

    pooling_str_splitted = pooling_str.split("_")

    try:
        pooling_method = pooling_str_splitted[0]
        kernel_size = int(pooling_str_splitted[2])
        stride = int(pooling_str_splitted[3])
    except ValueError:
        logger.error(f"Extraction of needed Information from pooling_str failed. Please use format  <pooling-method>_pool_<kernel_size>_<stride> . Got {pooling_str}")
        return None, f"information_extraction_failed_from_pooling_str_{pooling_str}"

    valid_pooling_methods = ["max", "mean"]
    if pooling_method not in valid_pooling_methods:
        logger.error(f"received invalid pooling method {pooling_method}. Choose from {valid_pooling_methods}")
        return None, "invalid_pooling_method_given"

    if pooling_method == "max":
        return nn.MaxPool1d(kernel_size, stride), None
    return nn.AvgPool1d(kernel_size, stride), None


def Dataset_from_local_drive(folder: str, hierarchical_model: str):
    """returns the dataset based on reading in the data filewise from local storage
    """
    class CustomDataset(TensorDataset):
        def __init__(self, folder):
            all_files, error = file_utils.get_files(folder)
            if error:
                return

            all_files, error = if_hierarchical_data_remove_unneeded_files(all_files, folder, hierarchical_model)
            if error:
                return

            if os.path.join(folder, "labels.csv") in all_files:
                all_files.remove(os.path.join(folder, "labels.csv"))

            labels_df = pd.DataFrame({"filename": all_files})
            labels_df["label"] = labels_df["filename"].astype(str).str.split("/").apply(lambda x: x[-1]).astype(str).str.split("_").apply(lambda x: x[0]).apply(lambda x: x[-1]).astype(int)

            self.labels = labels_df
            self.folder = folder

            hierarchical_data_seen, error = processing_featured_hierarchical_data(folder)
            if error:
                return

            if hierarchical_data_seen:
                svm_model_folder, error = find_correct_svm_folder(folder, hierarchical_model)
                if error:
                    return

                standardizing_df, standardization_type, error = data_utils.prepare_standardization(svm_model_folder, True, hierarchical_model)
                if error:
                    return

                self.standardizing_df = standardizing_df
                self.standardization_type = standardization_type

            self.hierarchical_data_seen = hierarchical_data_seen

            with open(os.path.join(folder, "log.json")) as f:
                log_file = json.load(f)

            self.data_in_channel_format = log_file.get("prepare_dataset").get("grid")

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int):
            path = self.labels.iloc[index, 0]
            label = self.labels.iloc[index, 1]
            df = pd.read_csv(path, dtype=np.double, header=None)

            if self.hierarchical_data_seen:
                df = data_utils.standardize(
                    df[0].T, self.standardization_type, self.standardizing_df)

            if self.data_in_channel_format:
                nr_channels = len(df)
                df = df.T.to_numpy().reshape(nr_channels, -1)
            else:
                df = df.to_numpy()

            return df, label

    return CustomDataset(folder)


def read_prepared_data_file(filename: str):
    """reads a prepared data file 
    """
    if "win" in sys.platform:
        filename = filename.replace("\\","/")
    df = pd.read_csv(filename, dtype=np.float64, header=None)
    label = int(str(filename.split("/")[-1].split("_")[0])[-1])
    return (df, label)


def Dataset_from_RAM(folder: str, epochs_to_train: float, hierarchical_model: str):
    """Creates a dataset existing entirely in the RAM
    """
    all_files, error = file_utils.get_files(folder, "csv")
    if error:
        return None, error

    all_files.remove(os.path.join(folder, "labels.csv"))

    all_files, error = if_hierarchical_data_remove_unneeded_files(all_files, folder, hierarchical_model)
    if error:
        return None, error

    if epochs_to_train < 1:
        all_files = all_files[0:int(epochs_to_train*len(all_files))]

    with Pool() as pool:
        result = pool.map(read_prepared_data_file, all_files)

    hierarchical_data_seen, error = processing_featured_hierarchical_data(folder)
    if error:
        return None, error

    if hierarchical_data_seen:
        joined_df, error = standardize_before_concatting(result, folder, hierarchical_model)
        if error:
            return None, error
    else:
        joined_df = pd.concat([x[0].T for x in result])

    labels = [x[1] for x in result]
    del result

    # create Tensors from the objects
    x = np.stack([joined_df[col].to_numpy() for col in joined_df.columns], 1).astype(np.float64)
    del joined_df
    x = torch.tensor(x, dtype=torch.float)
    y = torch.LongTensor(labels).reshape(-1, 1)

    # When CNN or RNN data is being processed, there might me a reshaping of x needed to bring it to [batchsize,  nr_channels, resampled_timepoints_per_window]
    if x.shape[0] != y.shape[0]:
        nr_channels = len(pd.read_csv(all_files[0], usecols=[0], dtype=np.float64, header=None))
        x = reshape_tensor_to_first_identical_dim(x, y, nr_channels)

    Dataset = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))
    del x
    del y

    return Dataset, None


def return_dataset(folder: str, data_set: str, epochs: float, hierarchical_model: str):
    """uses the prepared folder to create a Dataset Object from it
    """
    # load files into local memory and return a Dataset from them,
    # otherwise warn the user that training will take longer, as the trainings_data needs to be loaded from scratch for each minibatch

    if not os.path.exists(folder):
        logger.error(f"folder does not exist")
        return None, "given_dir_does_not_exist"
    if not os.path.exists(os.path.join(folder, "labels.csv")):
        logger.error(f"labels.csv does not exist in folder {folder}")
        return None, "no_labels_csv_in_given_dir"

    available_RAM_in_GB, needed_ram, error = check_needed_and_available_ram(folder, hierarchical_model, epochs, data_set)
    if error:
        return None, error

    # to prevent filling the RAM completely, it is only loaded when enough RAM is available
    if (available_RAM_in_GB - needed_ram*4) > 2:
        logger.info(f"As reading in the {data_set}-dataset into is RAM possible, therefore starting to read-in files")
        DataSet, error = Dataset_from_RAM(folder, epochs, hierarchical_model)
        if error:
            return None, error

        logger.info(f"Finished reading in {data_set}-Dataset into RAM")
        return DataSet, None

    return Dataset_from_local_drive(folder, hierarchical_model), None


def return_y_transformed(folder: str):
    """returns a version of the target, which is suited for CEL
    """
    if not os.path.exists(folder):
        return None, "given_folder_does_not_exist"
    if not os.path.exists(os.path.join(folder, "labels.csv")):
        return None, "no_labels_csv_in_given_folder"

    try:
        df = pd.read_csv(os.path.join(folder, "labels.csv"), usecols=["label"], dtype=np.int16)
    except ValueError:
        return None, "labels.csv_does_not_contain_label_col"

    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long).reshape(-1, 1)
    return bring_to_correct_format(y), None


def create_loader(folder: str, minibatch_size: int, data_set: str, epochs: float, hierarchical_model: str):
    """returns a Dataloader object which returns the data batch-wise ith the given minibatch_size
    """
    dataset, error = return_dataset(folder, data_set, epochs, hierarchical_model)
    if error:
        return None, error
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=False)

    return loader, None


def get_n_features(folder: str, hierarchical_model: str):
    """returns the nr of features of the data in a given folder
    """
    if not os.path.exists(folder):
        logger.error(f"given folder does not exist_{folder}")
        return 0, "folder_does_not_exist"

    featured_hierarchical_data, error = processing_featured_hierarchical_data(folder)
    if error:
        return 0, error

    if featured_hierarchical_data:
        return find_n_features_for_hierarchical_data(folder, hierarchical_model)

    all_files, error = file_utils.get_files(folder, "csv")
    if error:
        return 0, error

    if len(all_files) < 2:
        logger.error(f"Given folder does not contain the minimum amount of files: {folder}")
        return 0, "folder_does_not_contain_enough_files"

    # Pick any file which is not the labels.csv file
    suitable_file = all_files[0] if not all_files[0].endswith("labels.csv") else all_files[1]
    n_features = len(pd.read_csv(suitable_file, dtype=np.float64, header=None))
    return n_features, None


def get_out_szs(folder: str, hierarchical_model: str):
    """returns the nr of target classes of the data in a given folder
    """
    if not os.path.exists(folder):
        logger.error(f"given folder does not exist_{folder}")
        return 0, "folder_does_not_exist"
    if not os.path.exists(os.path.join(folder, "labels.csv")):
        logger.error(f"labels.csv not found in folder {folder}")
        return 0, "labels.csv_does_not_exist"

    try:
        df = pd.read_csv(os.path.join(folder, "labels.csv"), usecols=["label"], dtype=np.int16)
    except ValueError:
        logger.error(f"labels.csv does not contain a col label")
        return 0, "labels.csv_does_not_contain_label_col"

    out_szs = len(df["label"].unique())

    featured_hierarchical_data, error = processing_featured_hierarchical_data(folder)
    if error:
        return 0, error

    if featured_hierarchical_data:
        out_szs = correct_out_szs_for_hierarchical_data(df["label"], hierarchical_model)

    return out_szs, None


def get_n_features_and_out_sz(folder: str, hierarchical_model: str):
    """ returns the nr of features and the nr of target classes
    """
    n_features, error = get_n_features(folder, hierarchical_model)
    if error:
        return 0, 0, error

    out_szs, error = get_out_szs(folder, hierarchical_model)
    if error:
        return 0, 0, error

    return n_features, out_szs, None


def reshape_tensor_to_first_identical_dim(tensor_1: torch.Tensor, tensor_2: torch.Tensor, nr_channels: int):
    """reshapes tensor 1 to have the same nr of entries in the first dimension as tensor_2 and nr_channels entries in the last dim 
    """
    return tensor_1.reshape(tensor_2.shape[0], nr_channels, -1)


def valid_first_width_after_flattening(folder: str, conv_and_pool_layers: list, width_of_first_layer_after_flattening: int, padding=0):
    """checks if the amount of hidden units in the first layer after all the convolutions and pooling is big enough to accommodate all the needed data
    """
    # get a df to know the shape of an input
    all_files, error = file_utils.get_files(folder, "csv")
    if error:
        return error
    path_to_labels_file = os.path.join(folder, "labels.csv")
    if path_to_labels_file in all_files:
        all_files.remove(path_to_labels_file)

    nr_observations = len(pd.read_csv(all_files[0], nrows=1, header=None, dtype=np.float64).columns)
    all_filters = [1]

    for _, layer in enumerate(conv_and_pool_layers, start=1):
        if "conv" in layer:
            convolution_string_splitted = layer.split("_")
            nr_filters = int(convolution_string_splitted[1])
            all_filters.append(nr_filters)
            kernel_size = int(convolution_string_splitted[2])
            stride = int(convolution_string_splitted[3])
            nr_observations = (math.floor(
                (nr_observations-kernel_size+2*padding)/stride)+1)
            continue

        if "pool" in layer:
            pooling_str_splitted = layer.split("_")
            kernel_size = int(pooling_str_splitted[2])
            stride = int(pooling_str_splitted[3])
            nr_observations = (math.floor(
                (nr_observations-kernel_size+2*padding)/stride)+1)
            continue

        logger.error(f"received layer which is neither convolution or polling layer : {layer}")
        return f"received_unidentifiable_layer_{layer}"

    nr_flattened_units = int(nr_observations*all_filters[-1])

    if nr_flattened_units != int(width_of_first_layer_after_flattening):
        logger.error(f"To contain the flattened data after convolution and pooling {nr_flattened_units} units are needed, but {width_of_first_layer_after_flattening} were given")
        return "wrong_amount_of_hidden_units_after_flattening"

    return None


def relevant_part_of_truth(len_y_pred_joined: int, truth: torch.Tensor):
    """cuts away the part of the data not used, if it was trained for only a part of an epoch
    """
    return torch.split(truth, len_y_pred_joined)[0]


def check_needed_and_available_ram(folder: str, hierarchical_model: str, epochs: float, data_set: str):
    """Checks the needed and available RAM 
    """
    # Check if loading the (part) of the Dataset into RAM is possible
    available_RAM_in_GB = round(psutil.virtual_memory().available/1000000000, 2)
    files_in_dir, error = file_utils.get_files(folder, "csv")
    if error:
        return None, None, error

    files_in_dir, error = if_hierarchical_data_remove_unneeded_files(files_in_dir, folder, hierarchical_model)
    if error:
        return 0., 0., error

    file_sizes = file_utils.count_filesizes_of_list(files_in_dir)
    needed_ram = round(min(1, epochs)*file_sizes, 2)
    logger.info(f"The needed RAM to read-in the (part) of {data_set}-dataset is {needed_ram} GB, while {available_RAM_in_GB} are available")

    return available_RAM_in_GB, needed_ram, None


def reduce_to_relevant_files_for_hierarchical_model(list_of_files: list, hierarchical_model: str):
    """returns a list of files, which are relevant for the chosen hierarchical model
    """
    valid_hierarchical_levels = ["top", "walking", "lifting"]
    if hierarchical_model not in valid_hierarchical_levels:
        logger.error(f"received invalid level {hierarchical_model} Choose from {valid_hierarchical_levels}")
        return [], "invalid_hierarchical_model_given"
    if hierarchical_model == "top":
        valid_prefixes = ("1_", "2_", "3_", "4_")
    elif hierarchical_model == "lifting":
        valid_prefixes = ("11_", "12_", "13_")
    else:
        valid_prefixes = ("21_", "22_", "23_")

    if "win" in sys.platform:
        list_of_files = [file.replace("\\","/") for file in list_of_files]

    return [file for file in list_of_files if file.split("/")[-1].startswith(valid_prefixes)], None


def if_hierarchical_data_remove_unneeded_files(list_of_files: list, folder: str, hierarchical_model: str):
    """checks if the files do contain hierarchical data and removes all unneeded files for chosen model if so
    """
    featured_hierarchical_data, error = processing_featured_hierarchical_data(folder)
    if error:
        return [], error

    if featured_hierarchical_data:
        list_of_files, error = reduce_to_relevant_files_for_hierarchical_model(list_of_files, hierarchical_model)
        if error:
            return list_of_files, error

    return list_of_files, None


def correct_out_szs_for_hierarchical_data(label_series: pd.Series, hierarchical_model: str):
    """returns the correct out_szs based on the given params
    """
    unique_labels = label_series.unique()
    if hierarchical_model == "top":
        return len([x for x in unique_labels if len(str(x)) == 1])
    elif hierarchical_model == "lifting":
        return len([x for x in unique_labels if len(str(x)) == 2 and str(x)[0] == "1"])
    elif hierarchical_model == "walking":
        return len([x for x in unique_labels if len(str(x)) == 2 and str(x)[0] == "2"])
    logger.error(f"received invalid hierarchical_model {hierarchical_model}")
    return 0


def processing_featured_hierarchical_data(folder: str):
    """reads a log_file in folder to check if the data seen is in featured and hierarchical form
    """
    log_file_location = os.path.join(folder, "log.json")
    if not os.path.exists(log_file_location):
        logger.error(f"Log file not found in folder {folder}")
        return True, "log_file_not_found"
    with open(log_file_location) as f:
        log_file = json.load(f)

    if len([key for key in log_file if "balancing" in key]) != 1:
        logger.error(f"found log file which does not contain information regarding balancing")
        return True, "log_file_does_not_contain_information_regarding_balancing"

    balancing_type = [key for key in log_file if "balancing" in key][0]

    return "generate_features" in log_file.keys() and log_file[balancing_type]["label_depth"] == 2, None


def find_n_features_for_hierarchical_data(folder: str, hierarchical_model: str):
    """ returns the correct amount of features for the given hierarchical_model
    """
    all_files, error = file_utils.get_files(folder)
    if error:
        return 0, error

    reduced_files, error = reduce_to_relevant_files_for_hierarchical_model(
        all_files, hierarchical_model)
    if error:
        return 0, error

    return len(pd.read_csv(reduced_files[0], header=None, dtype=np.float64)), None


def create_rnn_model(batch_normalization: bool, seed: int, log_file={}):
    """returns a Recurrent Neural Network
    """
    torch.manual_seed(seed)

    class RecurrentNetwork(nn.Module):
        def __init__(self, n_features: int, out_sz: int, layers: list, dropout_rate: float, batch_normalization: bool):
            super().__init__()
            self.n_features = n_features

            # Because of the way pytorch is implemented, adding multiple LSTM-Layers to the network is only manually possible, see
            # https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3
            # and https://discuss.pytorch.org/t/error-optimizer-got-an-empty-parameter-list/1501/8 comment from Novak

            hidden_size, _ = determine_hidden_size(layers)
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(n_features, hidden_size)

            # Add the fully-connected layer(s):
            layers.remove(layers[0])
            layerlist = build_fully_connected_layers(
                hidden_size, layers, batch_normalization, dropout_rate, out_sz)
            self.linear = nn.Sequential(*layerlist)

            self.nr_timestamps_seen = int(log_file.get("create_windows").get("window_length") / log_file.get("create_windows").get("resampling_rate"))

            # Initialize h0 (short-term memory) and c0 (long-term memory):
            self.hidden = (torch.zeros(1, self.nr_timestamps_seen, hidden_size).double(),
                           torch.zeros(1, self.nr_timestamps_seen, hidden_size).double())

        def predict(self, x: torch.FloatTensor):
            batchsize = x.shape[0]

            x = x.view(batchsize, -1, self.n_features).double()

            lstm_out, self.hidden = self.lstm(x, self.hidden)

            pred = self.linear(lstm_out)

            return pred[:, self.nr_timestamps_seen-1]

    return RecurrentNetwork


def valid_hidden_size(layerstring: str):
    """"Checks if the given hidden size is valid
    """
    try:
        all_layers = [int(x) for x in layerstring.split("f")]
        hidden_size = int(layerstring[0])
    except ValueError:
        logger.error(f" Cannot convert given layerstring {layerstring} to a valid int")
        return "invalid_hidden_size_given"

    if min(all_layers) < 1:
        logger.error(f"Given hidden layer size {hidden_size} is too small")
        return "given_hidden_size_too_small"
    if max(all_layers) > 5000:
        logger.error(f"Given amount of hidden units is too big: {hidden_size}")
        return "too_much_hidden_units_given"

    if max(all_layers) > 500:
        logger.warning(f"received large amount of hidden units: {hidden_size}")

    return None


def determine_hidden_size(layers: list):
    """returns the hidden size
    """
    if len(layers) < 1:
        logger.error(f"received layers with suspicious length : {layers}")
        return 0, "invalid_layers_given"

    hidden_size = layers[0]
    if hidden_size <= 0:
        logger.error(f"Given hidden layer size {hidden_size} is too small")
        return 0, "given_hidden_size_too_small"
    if hidden_size > 5000:
        logger.error(f"Given amount of hidden units is too big: {hidden_size}")
        return 0, "too_much_hidden_units_given"

    return hidden_size, None


def load_needed_params(model_folder: str, network_type: str):
    """reads the log file in the given folder to return the needed params for initializing the model
    """
    with open(os.path.join(model_folder, "log.json")) as f:
        model_log_file = json.load(f)

    seed = model_log_file.get("train_model").get("seed")
    if not isinstance(seed, int):
        logger.error(f"The read-in seed value from log is not an int")
        return None, None, None, None, None, None, "received_non_int_seed"

    batch_normalization = model_log_file.get("train_model").get("batch_normalization")
    if not isinstance(batch_normalization, bool):
        logger.error(f"The read-in batch_normalization value from log is not a bool")
        return None, None, None, None, None, None, "received_non_bool_batch_normalization"

    n_features = model_log_file.get("train_model").get("n_features")
    if not isinstance(n_features, int):
        logger.error(f"The read-in n_features value from log is not an int")
        return None, None, None, None, None, None, "received_non_int_n_features"

    out_szs = model_log_file.get("train_model").get("out_szs")
    if not isinstance(out_szs, int):
        logger.error(f"The read-in out_szs value from log is not an int")
        return None, None, None, None, None, None, "received_non_int_out_szs"

    dropout_rate = model_log_file.get("train_model").get("dropout_rate")
    if not isinstance(dropout_rate, float):
        logger.error(f"The read-in dropout_rate value from log is not an float")
        return None, None, None, None, None, None, "received_non_float_dropout_rate"

    layers, error = build_layers_from_string(model_log_file.get("train_model").get("layer_structure"), network_type)
    if error:
        return None, None, None, None, None, None, error

    return seed, batch_normalization, n_features, out_szs, dropout_rate, layers, error


def load_model(model_folder: str, network_type: str, log_file: dict, model_name = "model.pt"):
    """loads the  model from the given folder
    """

    if network_type not in ["FFNN", "CNN", "RNN", "SVM"]:
        logger.error(f"received not yet implemented model type {network_type}")
        return None, "received_invalid_network_type"

    if network_type == "SVM":
        model, error = load_SVM_model(model_folder)
        return model, error

    seed, batch_normalization, n_features, out_szs, dropout_rate, layers, error = load_needed_params(model_folder, network_type)
    if error:
        return None, error

    Model = build_nn_model(network_type, batch_normalization, seed, log_file)
    model = Model(n_features, out_szs, layers, dropout_rate, batch_normalization)
    model.load_state_dict(torch.load(os.path.join(model_folder, model_name)), strict=False)

    return model, None


def load_SVM_model(model_folder: str):
    """loads the SVM from json format
    """
    all_pkl_files, error = file_utils.get_files(model_folder, ".pkl")
    if error:
        return None, error

    model = load_svm_from_pkl(all_pkl_files[0])

    return model, None


def evaluate_on_validation_data(model: nn.Module, saving_folder: str, validation_losses: list, validation_accuracies: list, validationloader: DataLoader, criterion, frac: float, keep_calc: bool, time_unit: str, time_nr: int, network_type: str, log_file: dict):
    """reports the performance of the model on the validation data 
    """
    error = save_model(model, saving_folder)
    if error:
        logger.error(f"Saving the model failed")
        return [], [], "saving_model_failed"

    validation_accuracies, validation_losses = update_validation_accuracies_and_losses(validation_losses, validation_accuracies, validationloader, criterion, model, frac)
    logger.info(f'The accuracy on {round(frac*100,2)}% of the validation data after {time_unit} {time_nr} is {round(validation_accuracies[-1]*100,2)}%')
    model, error = load_model(saving_folder, network_type, log_file)
    if error:
        return [], [], error

    if not keep_calc:
        validation_accuracies = validation_accuracies[:-1]
        validation_losses = validation_losses[:-1]

    return validation_accuracies, validation_losses, None


def load_svm_from_pkl(pkl_location: str):
    """ returns a sklearn-model from the given location
    """
    model = joblib.load(open(pkl_location, 'rb'))
    # The output classes are changed to three numerical classes
    model.classes_ = np.array([0, 1, 2]).astype(np.int64)

    return model


def load_model_if_training_was_interrupted(log_file: dict, model: nn.Module):
    """tests if the log file does contain information, that prior training was interrupted and loads the model to continue training 
    """
    if "train_model" not in log_file.keys():
        logger.error(f"received  invalid log file")
        return None, None, 0, "invalid_log_file_given"

    if "already_trained_epochs" not in log_file["train_model"].keys():
        return model, log_file, 0, None

    if log_file.get("train_model").get("already_trained_epochs") < log_file.get("train_model").get("epochs"):
        model_saving_folder = log_file.get("train_model").get("saving_folder")
        network_type = log_file.get("train_model").get("type")
        model, error = load_model(model_saving_folder, network_type, log_file)
        if error:
            return None, None, 0, error
        return model, log_file, log_file["train_model"].get("already_trained_epochs"), None

    if log_file.get("train_model").get("already_trained_epochs") == log_file.get("train_model").get("epochs"):
        logger.info(f"Finished Training for all requested Epochs")
        model_saving_folder = log_file.get("train_model").get("saving_folder")
        network_type = log_file.get("train_model").get("type")
        model, error = load_model(model_saving_folder, network_type, log_file)
        return model, log_file, log_file["train_model"].get("already_trained_epochs"), None

    logger.error(f"Figuring out if prior training occurred failed")
    return None, None, 0, "failed_loading_prior_model"


def standardize_before_concatting(list_of_dfs: list, folder: str, hierarchical_model: str):
    """returns a concatted df with the correct standardization applied to the seen features
    """
    # find the correct model folder for the respective svm
    svm_model_folder, error = find_correct_svm_folder(folder, hierarchical_model)
    if error:
        return pd.DataFrame(), error

    standardizing_df, standardization_type, error = data_utils.prepare_standardization(svm_model_folder, True, hierarchical_model)
    if error:
        return pd.DataFrame(), error

    return pd.concat([data_utils.standardize(x[0].T, standardization_type, standardizing_df) for x in list_of_dfs]), None


def find_correct_svm_folder(folder: str, hierarchical_model: str):
    """detects the respective svm folder with the standardization file
    """
    log_position = os.path.join(folder, "log.json")
    if not os.path.exists(log_position):
        logger.error(f"Could not find a log file in the currently handled folder {folder}")
        return "", "no_log_file_found"

    with open(log_position) as f:
        model_log_file = json.load(f)

    datapath = model_log_file.get("create_windows").get("data_path")

    return os.path.join(datapath, "train_model", "SVM", hierarchical_model), None


def early_stopping(list_of_accuracies:list, saving_folder:str, model:nn.Module, store_local:bool):
    """tests if the current state of the model is the best and if so, saves the model
    """
    if not store_local:
        return None
    
    if len(list_of_accuracies) == 0:
        logger.error(f"Received empy list")
        return "list_of_accuracies_empty"

    if list_of_accuracies[-1] == max(list_of_accuracies):
        error = save_model(model, saving_folder, "best_val_model.pt")
        return error
    
    return None

def aggregate_predictions(y_pred: torch.Tensor, nr_wanted_classes: int):
    """aggregates the lowest two classes into a common class
    """
    if len(torch.unique(y_pred)) == nr_wanted_classes:
        return y_pred
    elif len(torch.unique(y_pred)) < nr_wanted_classes:
        logger.error(f"The number of predicted classes is less than the number of wanted classes before aggregating the predictions")
        return torch.tensor([])
    
    return torch.tensor([max([0, value -1]) for value in y_pred.tolist()]) 