import os
import time
import json
from datetime import datetime
from multiprocessing import Pool

from config.log import logger
from utils import file_utils, data_utils, generate_feature_utils


def add_args(group):
    """ Description of possible arguments for the generating of features
    """
    group.add_argument(
        '-i', '--input_date',
        metavar='',
        dest='input_date',
        required=True,
        type=str,
        help='Specify the date to identify for which windowed data to generate features for, format YYYY-MM-DDThh:mm'
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
    if "output_date" not in table:
        table["output_date"] = None
    return table


def generate_features(table: dict):
    """ Generates features for the windows
    """
    error = generate_feature_utils.valid_table(table)
    if error:
        return

   # For better readability the variables are created internally from the table
    data_path = table.get("data_path")
    store_local = not table.get("dryrun")
    input_date = table.get("input_date")
    output_date = table.get("output_date")
    service = "generate_features"

    # When the results are being stored and the respective folders do not exist yet, create them at <dp>/features/<output_date>
    table["saving_time"] = output_date
    saving_folder = os.path.join(data_path, "features", output_date)
    table["saving_folder"] = saving_folder
    if store_local and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    all_window_files, error = file_utils.get_files(os.path.join(data_path, "windows", input_date), ".csv")
    if error:
        return

    # Initialize a log file in which the progress of processing, as well as parameters for processing are saved
    log_file, error = file_utils.initialize_log_file(os.path.join(saving_folder, "log.json"), os.path.join(data_path, "windows", input_date, "log.json"), service, table)
    if error:
        logger.error(f"Initializing log file failed: {error}")
        return

    all_window_files, error = file_utils.reduce_files_to_handle(all_window_files, saving_folder, service)
    if error or all_window_files == []:
        return

    # Calculation may be speed up, when the files within a chunk have similar size
    all_window_files = file_utils.sort_by_filesizes(all_window_files)

    # As processing all the data might take some time, an estimation is given
    _, error = file_utils.estimate_needed_time(len(all_window_files), service)
    if error:
        return
    time_before = time.time()

    feature_functions_dict, error = generate_feature_utils.feature_utils.feature_functions_dict(data_path)
    if error:
        return

    with open(os.path.join(data_path, "anonymization_file.json")) as f:
        anonymize_file: dict = json.load(f)

    list_of_file_dicts, error = generate_feature_utils.create_list_of_file_dicts(all_window_files, table, feature_functions_dict, anonymize_file)
    if error:
        return

    # Analogue to the creation of windows, multiprocessing is used, which is being done chunkwise
    session_chunks = data_utils.split_files_to_handle_into_chunks(list_of_file_dicts, 96)

    logger.info(f"Finished first processing steps, Starting with generating of features")

    for chunk_number, chunk in enumerate(session_chunks, start=1):
        # Process the files in the chunk with all available CPU cores
        with Pool() as pool:
            result = pool.map(generate_feature_utils.generate_features_for_file, chunk)

        # track current progress / success of the processing
        log_file = file_utils.create_log_from_result_strings(result, os.path.join(saving_folder, "log.json"), service)

        logger.info(f"[{chunk_number}/{max([len(session_chunks), chunk_number])}] chunks finished. {len(log_file[service]['successfully_processed_files'])} files successfully processed")

        # If errors were found, warn the user and save the log file for data investigation purposes
        if len(log_file[service]["occurred_errors"]) != 0:
            logger.warning(f"These errors occurred: {log_file[service]['occurred_errors'].keys()}")
            logger.info(f"For debugging purposes the log file gets saved: Contains errors & respective files")

            error = file_utils.update_log_file(log_file, table, saving_folder, service)
            if error:
                logger.error(f"saving log file failed: {error}")

        if store_local:
            error = file_utils.update_log_file(log_file, table, saving_folder, service)
            if error:
                logger.error(f"saving log file failed: {error}")

    duration = round((time.time() - time_before)/60, 2)
    logger.info(f"Finished creating windows. Duration: {duration} min for {len(all_window_files)} files")
