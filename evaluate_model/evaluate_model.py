import os
import sys
import json
from datetime import datetime

from utils import evaluation_utils, file_utils
from config.log import logger
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def add_args(group):
    """ Description of possible arguments for the evaluation of models
    """
    group.add_argument(
        '-m', '--model',
        metavar='',
        dest='model_folder',
        required=True,
        type=str,
        help='Specify the full absolute path to folder containing the model'
    )
    group.add_argument(
        '-test_d', '--test_data',
        metavar='',
        dest='test_data',
        required=True,
        type=str,
        help='Specify the full absolute path to folder containing the test data'
    )
    group.add_argument(
        '-w', '--weighting',
        metavar='\b',
        dest='weighting',
        type=str,
        choices=['balanced', 'original', 'undersample_majority_classes'],
        default='original',
        help='Choose the weighting of the classes to be applied, default: %(default)s'
    )
    group.add_argument(
        '-o', '--output_date',
        metavar='',
        dest='output_date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%dT%H:%M'),
        help='Specify the date of the folder in which to save the results of the evaluation, format YYYY-MM-DDThh:mm'
    )


def check_table_args(table: dict):
    """ Sets the default values if not given by user
    """
    return table


def evaluate_model(table: dict):
    """ Evaluates the performance of the given model
    """
    error = evaluation_utils.valid_table(table)
    if error:
        return

    # For better readability the variables are created internally from the table
    data_path: str = table.get("data_path")
    store_local: bool = not table.get("dryrun")

    model_folder: str = table.get("model_folder")
    test_data: str = table.get("test_data")
    weighting: str = table.get("weighting")
    output_date: str = table.get("output_date")

    service = "evaluate_model"

    # When the results are being stored and the respective folders do not exist yet, create them at <dp>/evaluate_model/<network_type>/<output_date>
    with open(os.path.join(model_folder, "log.json")) as f:
        model_log_file = json.load(f)
    network_type = model_log_file.get("train_model").get("type")
    saving_folder = os.path.join(data_path, "evaluate_model", network_type, output_date)
    table["saving_folder"] = saving_folder
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Initialize a log file in which the progress of processing, as well as parameters for processing are saved
    log_file, error = file_utils.initialize_log_file(os.path.join(saving_folder, "log.json"), os.path.join(model_folder, "log.json"), service, table, os.path.join(test_data, "log.json"))
    if error:
        logger.error(f"Initializing log file failed: {error}")
        return

    error = evaluation_utils.valid_preprocessing_of_data_for_model(model_folder, test_data)
    if error:
        return

    model, error = evaluation_utils.load_trained_model(model_folder, log_file)
    if error:
        return

    verbose = table.get("log_level") == 'debug'
    confusion_matrix, accuracy, predictions_df, prediction_time, error = evaluation_utils.create_predictions(model, test_data, model_folder, store_local, saving_folder, verbose, 1.0)
    if error:
        return
    
    aggregate_lifting_and_lowering = predictions_df['true_label'].nunique() == 6
    if aggregate_lifting_and_lowering and "Raw_prediction_label_7" in predictions_df.columns:
        predictions_df = evaluation_utils.aggregate_raw_predictions(predictions_df)
    predictions_df = evaluation_utils.add_probability_scores(predictions_df)

    log_file[service]["mean_prediction_time_s"] = prediction_time.mean()
    log_file[service]["median_prediction_time_s"] = prediction_time.median()
    log_file[service]["min_prediction_time_s"] = prediction_time.min()
    log_file[service]["max_prediction_time_s"] = prediction_time.max()

    logger.info(f"The average prediction time in s is {log_file[service]['mean_prediction_time_s']}")

    # Reweighting of classes gets applied only in the last step of the pipeline, i.e. here for DeepFFNet, LSTMNet and ConvNet
    predictions_df.to_csv(os.path.join(saving_folder, "predictions.csv"), index=False)
    if predictions_df['true_label'].nunique() > 3:
        predictions_df = evaluation_utils.reweight_classes(saving_folder, "", "", aggregate_lifting_and_lowering, weighting, log_file)
        if predictions_df.empty:
            return
    resulting_metrics, cohen_k_score, matthews_corr_coeff, error = evaluation_utils.calculate_metrics(predictions_df, model_folder)
    if error:
        return
    log_file[service]["cohen_k_score"] = cohen_k_score
    log_file[service]["matthews_corr_coeff"] = matthews_corr_coeff

    if store_local:
        predictions_df.to_csv(os.path.join(saving_folder, "predictions.csv"), index=False)
        resulting_metrics.to_csv(os.path.join(saving_folder, "resulting_metrics.csv"))

    if store_local:
        if len(log_file[service]["occurred_errors"]) == 0:
            log_file[service]["successfully_processed_files"] = "all"

        error = file_utils.update_log_file(log_file, table, saving_folder, service)
        if error:
            return error
        
