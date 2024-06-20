import os
import sys
from datetime import datetime

from utils import file_utils, classwise_evaluation_utils
from config.log import logger
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def add_args(group):
    """ Description of possible arguments for class-wise evaluations of models
    """
    group.add_argument(
        '-top', '--top_level_classifier_folder',
        metavar='',
        dest='top_level_classifier_folder',
        required=True,
        type=str,
        help='Specify the absolute path to the folder of the evaluated top-level-classifier'
    )
    group.add_argument(
        '-lift', '--lifting_classifier_folder',
        metavar='',
        dest='lifting_classifier_folder',
        required=False,
        type=str,
        help='Specify the absolute path to the folder of the evaluated lifting-classifier'
    )
    group.add_argument(
        '-walk', '--walking_classifier_folder',
        metavar='',
        dest='walking_classifier_folder',
        required=False,
        type=str,
        help='Specify the absolute path to the folder of the evaluated walking-classifier'
    )
    group.add_argument(
        '-o', '--output_date',
        metavar='',
        dest='output_date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%dT%H:%M'),
        help='Specify the date of the folder in which to save the plots, format YYYY-MM-DDThh:mm'
    )


def check_table_args(table):
    """ Sets the default values if not given by user
    """
    if "dryrun" not in table:
        table["dryrun"] = False
    if table.get("lifting_classifier_folder") is None:
        table["lifting_classifier_folder"] = ''
    if table.get("walking_classifier_folder") is None:
        table["walking_classifier_folder"] = ''
    return table


def evaluate_classwise(table: dict):
    """ Plots the class-wise ROC curves of a classifier
    """
    error = classwise_evaluation_utils.valid_table(table)
    if error: 
        return
    
    # For better readability the variables are created internally from the table
    data_path: str = table.get("data_path")
    store_local: bool = not table.get("dryrun")
    top_level_classifier_folder: str = table.get("top_level_classifier_folder")
    lifting_classifier_folder: str = table.get("lifting_classifier_folder")
    walking_classifier_folder: str = table.get("walking_classifier_folder")
    output_date = table.get("output_date")
    service = "evaluate_classwise"

    # Create the folder in which the results may be saved
    saving_folder = os.path.join(data_path, service, output_date)
    table["saving_folder"] = saving_folder
    if store_local and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    lifting_log_location = os.path.join(lifting_classifier_folder, "log.json") if lifting_classifier_folder else ""
    walking_log_location = os.path.join(walking_classifier_folder, "log.json") if walking_classifier_folder else ""

    log_file, error = file_utils.initialize_log_file(os.path.join(saving_folder, "log.json"), os.path.join(top_level_classifier_folder, "log.json"), service, table, lifting_log_location, walking_log_location)
    if error:
        logger.error(f"Initializing log file failed: {error}")
        return
    
    results = classwise_evaluation_utils.evaluate_classwise(top_level_classifier_folder, lifting_classifier_folder, walking_classifier_folder, store_local, saving_folder)    
    log_file[service]["results"] = results


    if store_local:
        file_utils.save_json(results, os.path.join(saving_folder, "class_wise_evaluations.json"))

        if len(log_file[service]["occurred_errors"]) == 0:
            log_file[service]["successfully_processed_files"] = "all"
        error = file_utils.update_log_file(log_file, table, saving_folder, service)
        if error:
            return error