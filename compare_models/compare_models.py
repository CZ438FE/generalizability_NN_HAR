import os
import sys
import pandas as pd
from datetime import datetime

from config.log import logger
from utils import file_utils, evaluation_utils

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def add_args(group):
    """ Description of possible arguments for statistical comparison of evaluated models
    """
    group.add_argument(
        '-m1', '--model_folder_1',
        metavar='',
        dest='model_folder_1',
        required=True,
        type=str,
        help='Specify the absolute path to the folder of one evaluated model'
    )
    group.add_argument(
        '-m2', '--model_folder_2',
        metavar='',
        dest='model_folder_2',
        required=True,
        type=str,
        help='Specify the absolute path to the folder of one evaluated model'
    )
    group.add_argument(
        '-m', '--other_model_folders',
        metavar='\b',
        dest='other_model_folders',
        type=str,
        nargs='+',
        help='Specify the absolute path to other folders of evaluated models (needed only for the friedman test)'
    )
    group.add_argument(
        '-a', '--alpha',
        metavar='',
        dest='alpha',
        required=False,
        type=float,
        default=0.01,
        help='Specify the alpha value for the statistical tests, default: %(default)s'
    )
    group.add_argument(
        '-o', '--output_date',
        metavar='',
        dest='output_date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%dT%H:%M'),
        help='Specify the date of the folder in which to save the results, format YYYY-MM-DDThh:mm'
    )


def check_table_args(table):
    """ Sets the default values if not given by user
    """
    return table


def compare_models(table: dict):
    """ Performs statistical tests between two classifiers
    """
    error = evaluation_utils.valid_model_comparison_table(table)
    if error: 
        return
    
    # For better readability the variables are created internally from the table
    data_path: str = table.get("data_path")
    store_local: bool = not table.get("dryrun")
    model_folder_1: str = table.get("model_folder_1")
    model_folder_2: str = table.get("model_folder_2")
    other_model_folders: list = table.get("other_model_folders")

    alpha: float = table.get("alpha")
    output_date = table.get("output_date")
    service = "compare_models"


    # Create the folder in which the results may be saved
    saving_folder = os.path.join(data_path, service, output_date)
    table["saving_folder"] = saving_folder
    if store_local and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    log_file, error = file_utils.initialize_log_file(os.path.join(saving_folder, "log.json"), os.path.join(model_folder_1, "log.json"), service, table, os.path.join(model_folder_2, "log.json"))
    if error:
        logger.error(f"Initializing log file failed: {error}")
        return
    
    # Load the predictions df
    predictions_1 = pd.read_csv(os.path.join(model_folder_1, "predictions.csv"))
    predictions_2 = pd.read_csv(os.path.join(model_folder_2, "predictions.csv"))

    # Perform all tests
    result = evaluation_utils.perform_statistical_model_comparison(predictions_1, predictions_2, other_model_folders, alpha)
    log_file[service]["result"] = result


    # Save the results
    if store_local:
        file_utils.save_json(result, os.path.join(saving_folder, "results.json"))

        if len(log_file[service]["occurred_errors"]) == 0:
            log_file[service]["successfully_processed_files"] = "all"
        error = file_utils.update_log_file(log_file, table, saving_folder, service)
        if error:
            return error