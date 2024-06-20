import time
import os
import math
from datetime import datetime
import torch

from config.log import logger
from utils import train_model_utils, file_utils, plot_utils


def add_args(group):
    """ Description of possible arguments for training of models
    """
    group.add_argument(
        '-train_d', '--training_data',
        metavar='',
        dest='training_data',
        required=True,
        type=str,
        help='Specify the folder in which to search for the trainings data'
    )
    group.add_argument(
        '-t', '--type',
        metavar='',
        dest='type',
        type=str,
        choices=['FFNN', 'CNN', 'RNN'],
        default="FFNN",
        help='Specify general structure of nn, choices: [%(choices)s], default: %(default)s'
    )
    group.add_argument(
        '-val_d', '--validation_data',
        metavar='',
        dest='validation_data',
        type=str,
        help='Specify the folder in which to search for the validation data'
    )
    group.add_argument(
        '-hier_m', '--hierarchical_model',
        metavar='',
        dest='hierarchical_model',
        type=str,
        choices=['top', 'lifting', 'walking'],
        default="top",
        help='Specify model to train for featured, hierarchical data, choices: [%(choices)s], default: %(default)s'
    )
    group.add_argument(
        '-m', '--minibatch_size',
        metavar='',
        dest='minibatch_size',
        type=int,
        default=64,
        help='Specify the size of the minibatches with which to train, default: %(default)s'
    )
    group.add_argument(
        '-e', '--epochs',
        metavar='',
        dest='epochs',
        type=float,
        default=1.0,
        help='Specify the amount of epochs with which to train, default: %(default)s'
    )
    group.add_argument(
        '-l', '--loss_function',
        metavar='',
        dest='loss_function',
        choices=['CEL'],
        type=str,
        default="CEL",
        help='Specify the loss function to use, choices: [%(choices)s], default: %(default)s'
    )
    group.add_argument(
        '-op', '--optimizer',
        metavar='',
        dest='optimizer',
        choices=['adam'],
        type=str,
        default="adam",
        help='Specify the optimizer to use, choices: [%(choices)s], default: %(default)s'
    )
    group.add_argument(
        '-d', '--dropout_rate',
        metavar='',
        dest='dropout_rate',
        type=float,
        default=0.0,
        help='Specify the dropout-rate,, default: %(default)s'
    )
    group.add_argument(
        '-ls', '--layer_structure',
        metavar='',
        dest='layer_structure',
        type=str,
        default="250f150f50",
        help='Specify the Amount of Units per Hidden Layer to use, separator is f, default: %(default)s'
    )
    group.add_argument(
        '-b', '--batch_normalization',
        action='store_const',
        const=True,
        default=False,
        help='Set if you want to use batch_normalization, default: %(default)s'
    )
    group.add_argument(
        '-lr', '--learning_rate',
        metavar='',
        dest='learning_rate',
        type=float,
        default=0.001,
        help='Specify the initial learning rate to use, default: %(default)s'
    )
    group.add_argument(
        '-nr_m', '--number_minibatches',
        metavar='',
        dest='number_minibatches',
        type=int,
        help='Specify the number of minibatches to train'
    )
    group.add_argument(
        '-p', '--plot',
        action='store_const',
        const=True,
        default=False,
        help='Set if you want to plot the results'
    )
    group.add_argument(
        '-o', '--output_date',
        metavar='',
        dest='output_date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%dT%H:%M'),
        help='Specify the date of the folder in which to save results, format YYYY-MM-DDThh:mm'
    )


def check_table_args(table):
    """ Sets the default values if not given by user
    """
    if "dryrun" not in table:
        table["dryrun"] = False
    if "type" not in table:
        table["type"] = "FFNN"
    if "validation_data" not in table:
        table["validation_data"] = None
    if "output_date" not in table:
        table["output_date"] = None
    if "minibatch_size" not in table or table["minibatch_size"] is None:
        table["minibatch_size"] = 64
    if "epochs" not in table or table["epochs"] is None:
        table["epochs"] = 1
    if "dropout_rate" not in table:
        table["dropout_rate"] = None
    if "batch_normalization" not in table:
        table["batch_normalization"] = False
    if "hierarchical_model" not in table:
        table["hierarchical_model"] = "top"
    if "number_minibatches" not in table:
        table["number_minibatches"] = None
    return table


def train_model(table: dict):
    """ Does the heavy lifting for the training of the models
    """
    table_as_string = str(table).replace(', ', '\n')
    logger.info(f"These are the given arguments:\n {table_as_string}")

    error = train_model_utils.valid_table(table)
    if error:
        return

    # For better readability the variables are created internally from the table
    data_path: str = table.get("data_path")
    store_local: bool = not table.get("dryrun")

    network_type: str = table.get("type")
    training_data: str = table.get("training_data")
    validation_data = table.get("validation_data")
    hierarchical_model: str = table.get("hierarchical_model")
    output_date = table.get("output_date")

    minibatch_size: int = table.get("minibatch_size")
    epochs: float = table.get("epochs")
    loss_function: str = table.get("loss_function")
    optimizer: str = table.get("optimizer")
    layer_structure: str = table.get("layer_structure")
    dropout_rate: float = table.get("dropout_rate")
    batch_normalization: bool = table.get("batch_normalization")
    learning_rate: float = table.get("learning_rate")
    number_minibatches: int = table.get("number_minibatches")
    plot: bool = table.get("plot")

    service = "train_model"

    # When the results are being stored and the respective folders do not exist yet, create them at <dp>/train_model/<output_date>
    saving_folder = os.path.join(data_path, "train_model", network_type,  output_date)
    table["saving_folder"] = saving_folder
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Set a seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    table["seed"] = seed

    # Keep track of the used params
    validation_data_log_pos = None
    if validation_data:
        validation_data_log_pos = os.path.join(validation_data, "log.json")
    log_file, error = file_utils.initialize_log_file(os.path.join(saving_folder, "log.json"), os.path.join(training_data, "log.json"), service, table, validation_data_log_pos)
    if error:
        logger.error(f"Initializing log file failed: {error}")
        return

    implemented_network_types = ["FFNN", "CNN", "RNN"]
    if network_type not in implemented_network_types:
        logger.error(f"Got invalid network type, please choose from {implemented_network_types}")
        return

    trainloader, error = train_model_utils.create_loader(training_data,  minibatch_size, "training", epochs, hierarchical_model)
    if error:
        return

    # The nr_of_features and the number of target classes are needed for the construction of the Network
    n_features, out_szs, error = train_model_utils.get_n_features_and_out_sz(training_data, hierarchical_model)
    if error:
        return

    table["n_features"] = n_features
    table["out_szs"] = out_szs
    error = file_utils.update_log_file(log_file, table, saving_folder, service)
    if error:
        logger.error(f"saving log file failed: {error}")

    if validation_data:
        validationloader, error = train_model_utils.create_loader(validation_data, minibatch_size, "validation", 1, hierarchical_model)
        if error:
            return

        # Test if the dimensionality of the validation data is equal to that of the trainings-data
        n_features_validation, out_szs_validation, error = train_model_utils.get_n_features_and_out_sz(validation_data, hierarchical_model)
        if error:
            return
        if n_features != n_features_validation:
            logger.error(f"Amount of features seen in validation ({n_features_validation}) and train data ({n_features}) differs")
            return
        if out_szs != out_szs_validation:
            logger.error(f"Amount of target classes seen in validation ({out_szs_validation}) and train data ({out_szs}) differs")
            return

    # Set all the model hyperparameters

    # Contains information regarding the width and used Unit types of all the hidden layers
    layers, error = train_model_utils.build_layers_from_string(layer_structure, network_type)
    if error:
        return

    criterion, error = train_model_utils.set_loss_function(loss_function)
    if error:
        return

    Model = train_model_utils.build_nn_model(network_type, batch_normalization, seed, log_file)

    # initialize the concrete model with the needed structure
    model = Model(n_features, out_szs, layers, dropout_rate, batch_normalization)
    if network_type == "RNN":
        model = model.double()

    # For comparison of different models, the nr of parameters in each model gets stored
    log_file[service]["nr_model_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model, log_file, table["already_trained_epochs"], error = train_model_utils.load_model_if_training_was_interrupted(
        log_file, model)
    if error:
        return

    optim, error = train_model_utils.set_optimizer(optimizer, learning_rate, model)
    if error:
        return

    # Allow for training of parts of epochs (e.g. 0.3 epochs)
    all_trainings_files, error = file_utils.get_files(training_data, "csv")
    if error:
        return

    remaining_epochs_to_train = epochs - table["already_trained_epochs"]

    nr_minibatches_per_epoch = len(all_trainings_files)/minibatch_size
    nr_minibatches_to_train = nr_minibatches_per_epoch * remaining_epochs_to_train
    # The amount of minibatches to train for is either manually given or nr_minibatches_per_epoch * remaining_epochs_to_train
    if number_minibatches:
        nr_minibatches_to_train = min([nr_minibatches_to_train, number_minibatches])
    nr_trained_minibatches = 0
    amount_of_minibatches_to_train_before_validating = 1000
    table["amount_of_minibatches_to_train_before_validating"] = amount_of_minibatches_to_train_before_validating

    # initialize objects for the storing of loss and accuracy per minibatch
    losses = []
    train_accuracies = []

    validation_losses = []
    validation_accuracies = []

    start_time = time.time()

    logger.info("Started training")

    for current_epoch in range(math.ceil(remaining_epochs_to_train)):

        for nr_batch, sample_batched in enumerate(trainloader):
            if nr_trained_minibatches >= nr_minibatches_to_train:
                break

            # classify the examples of the given batch
            y_pred = model.predict(sample_batched[0].double())

            if torch.any(torch.isnan(y_pred)):
                logger.error("Detected NaN values in the predictions")
                return

            # transform the ground truth for this batch into an appropriate format for the loss function
            y_truth = train_model_utils.bring_to_correct_format(sample_batched[1])

            # Calculate the loss / error based on the true values and the estimations
            loss = criterion(y_pred, y_truth)

            # Track the progress in a list
            losses.append(loss) if isinstance(loss, float) else losses.append(loss.item())

            # The format of the predictions needs to be changed for the calculation of the accuracy
            y_pred_transformed = train_model_utils.transform_predictions_for_confusion_matrix(y_pred)

            # Calculate the accuracy and append it to a list for the tracking of the progress of the accuracy
            nr_correct_classifications = train_model_utils.sum_identical_entries(y_truth, y_pred_transformed)
            train_accuracies.append(nr_correct_classifications.item()/len(y_pred_transformed))

            if nr_batch % 160 == 0 or nr_batch == 0:
                if nr_batch == 160 and current_epoch == 0:
                    logger.info(f"Training for {int(nr_minibatches_to_train)} minibatches will take {int((time.time()-start_time)*nr_minibatches_to_train/12000)} minutes")
                logger.info(f'epoch: {table["already_trained_epochs"]+1} minibatch {nr_batch} Trainings-data accuracy: {round(train_accuracies[-1]*100,2)}%')

            if validation_data and nr_batch % amount_of_minibatches_to_train_before_validating == 0:
                validation_accuracies, validation_losses, error = train_model_utils.evaluate_on_validation_data(model, saving_folder, validation_losses, validation_accuracies, validationloader, criterion, 1, True, "Minibatch", nr_trained_minibatches, network_type, log_file)
                if error:
                    return
                
                error = train_model_utils.early_stopping(validation_accuracies, saving_folder, model, store_local)
                if error:
                    return

            # Adjust the weights based on loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            if network_type == "RNN":
                # Detaching the hidden states is necessary, otherwise pytorch fails while trying to backpropagate until the start of training
                model.hidden[0].detach_()
                model.hidden[1].detach_()

            nr_trained_minibatches += 1

        # At the end of each epoch, update the log file, save the model
        if store_local:
            table["already_trained_epochs"] += 1
            error = train_model_utils.save_model(model, saving_folder)
            if error:
                logger.error(f"Saving the model failed")

            error = file_utils.update_log_file(log_file, table, saving_folder, service)
            if error:
                logger.error(f"saving log file failed: {error}")

        if validation_data:
            validation_accuracies, validation_losses, error = train_model_utils.evaluate_on_validation_data(model, saving_folder, validation_losses, validation_accuracies, validationloader, criterion, 1, False, "Epoch", current_epoch+1, network_type, log_file)
            if error:
                return

    logger.info(f"Finished training. Duration: {int((time.time() - start_time)/60)} min")
    
    # If validation_data was given, estimate the accuracy and a confusion matrix for the resulting model on the validation data
    if validation_data:
        _, final_validation_accuracy, _ = train_model_utils.display_confusion_matrix(validationloader, model, "validation", store_local, saving_folder, False, 1.0, hierarchical_model)
        log_file[service]["final_validation_accuracy"] = final_validation_accuracy
        log_file[service]["max_validation_accuracy"] = max(validation_accuracies)

        minibatch_nr_max_validation_acc = (validation_accuracies.index(max(validation_accuracies)))*amount_of_minibatches_to_train_before_validating
        logger.info(f"The best validation accuracy was archived at minibatch nr [{minibatch_nr_max_validation_acc}/{len(validation_accuracies)*amount_of_minibatches_to_train_before_validating}] with {round(max(validation_accuracies)*100, 2)} % ")
        log_file[service]["minibatch_nr_max_validation_acc"] = minibatch_nr_max_validation_acc
        log_file[service]["nr_minibatches_trained"] = math.ceil(nr_minibatches_to_train)

    # Create a classification matrix for the trainings-data
    _, final_training_accuracy, _ = train_model_utils.display_confusion_matrix(trainloader, model, "training", store_local, saving_folder, False, 0.25,  hierarchical_model)
    log_file[service]["final_training_accuracy"] = final_training_accuracy

    # Keep track of the real-world date, when the training finished
    log_file[service]["training_date"] = datetime.now().strftime('%Y-%m-%dT%H-%M')

    # Visualize the results
    if plot:
        plot_utils.plot_accuracies_and_loss_over_batches(train_accuracies, validation_accuracies, losses, validation_losses, store_local, saving_folder, amount_of_minibatches_to_train_before_validating)

    if store_local:
        error = train_model_utils.save_model(model, saving_folder)
        if error:
            logger.error(f"Saving the model failed")

        error = file_utils.update_log_file(log_file, table, saving_folder, service)
        if error:
            logger.error(f"saving log file failed: {error}")
