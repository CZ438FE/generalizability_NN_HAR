import os
import json
import pandas as pd
from datetime import datetime

from config.log import logger
from utils import file_utils, evaluation_utils


def add_args(group):
    """ Description of possible arguments for the evaluation of hierarchical models
    """
    group.add_argument(
        '-top', '--top_level_folder',
        metavar='',
        dest='top_level_folder',
        required=True,
        type=str,
        help='Specify the folder containing the results of the evaluation of the top-level classifier'
    )
    group.add_argument(
        '-lift', '--lifting_folder',
        metavar='',
        dest='lifting_folder',
        required=True,
        type=str,
        help='Specify the folder containing the results of the evaluation of the lifting classifier'
    )
    group.add_argument(
        '-walk', '--walking_folder',
        metavar='',
        dest='walking_folder',
        required=True,
        type=str,
        help='Specify the folder containing the results of the evaluation of the walking classifier'
    )
    group.add_argument(
        '-w', '--weighting',
        metavar='\b',
        dest='weighting',
        type=str,
        choices=['balanced', 'original', 'undersample_majority_classes'],
        default='balanced',
        help='Choose the weighting of the classes to be applied, default: %(default)s'
    )
    group.add_argument(
        '-o', '--output_date',
        metavar='',
        dest='output_date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%dT%H:%M'),
        help='Specify the date of the folder in which to save the files with the features, format YYYY-MM-DDThh:mm'
    )


def check_table_args(table):
    """ Sets the default values if not given by user
    """
    if "dryrun" not in table:
        table["dryrun"] = False
    return table


def evaluate_hierarchical(table: dict):
    """ Joins the evaluation of three classifiers for a joined evaluation
    """
    error = evaluation_utils.valid_table_hierarchical(table)
    if error:
        return

   # For better readability the variables are created internally from the table
    data_path: str = table.get("data_path")
    store_local: bool = not table.get("dryrun")
    top_level_folder: str = table.get("top_level_folder")
    lifting_folder: str = table.get("lifting_folder")
    walking_folder: str = table.get("walking_folder")
    weighting: str = table.get("weighting")
    output_date: str = table.get("output_date")
    service = "evaluate_hierarchical"

    # When the results are being stored and the respective folders do not exist yet, create them at <dp>/evaluate_model/<network_type>/<output_date>
    with open(os.path.join(top_level_folder, "log.json")) as f:
        model_log_file = json.load(f)
    network_type = model_log_file.get("train_model").get("type")
    saving_folder = os.path.join(data_path, "evaluate_model", network_type, output_date)
    table["saving_folder"] = saving_folder
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Initialize a log file in which the progress of evaluation
    log_file, error = file_utils.initialize_log_file(os.path.join(saving_folder, "log.json"), os.path.join(top_level_folder, "log.json"), service, table, os.path.join(lifting_folder, "log.json"), os.path.join(walking_folder, "log.json"))
    if error:
        logger.error(f"Initializing log file failed: {error}")
        return

    aggregated_lifting_lowering = pd.read_csv(os.path.join(lifting_folder, "predictions.csv"), usecols=["true_label"])['true_label'].nunique() == 2 

    predictions_df = evaluation_utils.reweight_classes(top_level_folder, lifting_folder, walking_folder, aggregated_lifting_lowering, weighting, log_file)
    if predictions_df.empty:
        return
    if aggregated_lifting_lowering and predictions_df['prediction'].nunique() == 7:
        predictions_df['prediction'] = [1 if elem == 0 else elem for elem in predictions_df['prediction']]
    if predictions_df['prediction'].nunique() == predictions_df['true_label'].nunique() and predictions_df['prediction'].max() +1 == predictions_df['true_label'].max():
        predictions_df['prediction'] +=1

    if store_local:
        predictions_df.to_csv(os.path.join(saving_folder, "predictions.csv"), index=False)
        error = file_utils.update_log_file(log_file, table, saving_folder, service)
        if error:
            return error

    confusion_matrix, accuracy, error = evaluation_utils.create_confusion_matrix(predictions_df, store_local, saving_folder, "top", False)
    if error:
        return

    resulting_metrics, cohen_k_score, matthews_corr_coeff, error = evaluation_utils.calculate_metrics(predictions_df, saving_folder)
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

        error = file_utils.update_log_file(
            log_file, table, saving_folder, service)
        if error:
            return error