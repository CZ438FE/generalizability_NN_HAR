import os
import sys
import time
import json
from datetime import datetime
from multiprocessing import Pool

from utils import file_utils, data_utils, create_windows_utils
from config.log import logger
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def add_args(group):
    """ Description of possible arguments for creation of windows
    """
    group.add_argument(
        '-w', '--window_length',
        metavar='',
        dest='window_length',
        default=1000,
        type=int,
        help='Specify window length in milliseconds, default: %(default)s'
    )
    group.add_argument(
        '-s', '--step_length',
        metavar='',
        dest='step_length',
        default=500,
        type=str,
        help='Specify step length in milliseconds, default: %(default)s'
    )
    group.add_argument(
        '-m', '--method',
        metavar='',
        dest='method',
        type=str,
        choices=['ffill', 'linear'],
        default="ffill",
        help='Specify the filling method to use for the data, choices: [%(choices)s], default: %(default)s '
    )
    group.add_argument(
        '-rs', '--resampling_rate',
        metavar='',
        dest='resampling_rate',
        default=10,
        type=int,
        help='Specify the resampling rate in ms, default: %(default)s'
    )
    group.add_argument(
        '-f', '--flatten',
        action='store_const',
        const=True,
        default=False,
        help='Set if you want to flatten the windows, default: %(default)s'
    )
    group.add_argument(
        '-n', '--normalize',
        metavar='',
        dest='normalize',
        type=str,
        choices=[None, 'max', "z_normalize"],
        default=None,
        help='Specify the normalization method to use for the data, choices: [%(choices)s], default: %(default)s'
    )
    group.add_argument(
        '-o', '--output_date',
        metavar='',
        dest='output_date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%dT%H:%M'),
        help='Specify the date of the folder in which to save created windows, format YYYY-MM-DDThh:mm'
    )


def check_table_args(table):
    """ Sets the default values if not given by user
    """
    if "dryrun" not in table:
        table["dryrun"] = False
    if "window_length" not in table:
        table["window_length"] = 1000
    if "step_length" not in table:
        table["step_length"] = 500
    if "resampling_rate" not in table:
        table["resampling_rate"] = 10
    if "output_date" not in table:
        table["output_date"] = None
    if "flatten" not in table:
        table["flatten"] = False
    if "method" not in table:
        table["flatten"] = "ffill"
    if "normalize" not in table:
        table["normalize"] = None
    return table


def create_windows(table: dict):
    """ Cuts labeled data into specified windows
    """
    error = create_windows_utils.valid_table(table)
    if error:
        return

    # For better readability the variables are created internally from the table
    data_path: str = table.get("data_path")
    store_local: bool = not table.get("dryrun")

    window_length: int = table.get("window_length")
    step_length: int = table.get("step_length")
    resampling_rate: int = table.get("resampling_rate")
    output_date: str = table.get("output_date")
    flatten: bool = table.get("flatten")
    filling_method: str = table.get("method")
    normalize: str = table.get("normalize")
    service = "create_windows"

    labeled_files, error = file_utils.get_files(os.path.join(data_path, "labeled_data"), ".csv")
    if error:
        return

    # Keep the information which dataset was processed in the log-file
    table["seen_data_set"] = file_utils.detect_dataset(labeled_files)

    # When the results are being stored and the respective folders do not exist yet, create them at <dp>/windows/<output_date>
    saving_folder = os.path.join(data_path, "windows", output_date)
    table["saving_folder"] = saving_folder
    if store_local and not os.path.exists(saving_folder):
        # Create the folder for the saving of the processed data
        os.makedirs(saving_folder)

    with open(os.path.join(data_path, "anonymization_file.json")) as f:
        anonymize_file: dict = json.load(f)

    needed_columns = anonymize_file.get("create_windows").get("needed_columns")
    data_columns = anonymize_file.get("create_windows").get("data_columns")
    if not set(data_columns).issubset(set(needed_columns)):
        logger.error(f"The given data_columns are not a true subset of the columns of the needed_columns")
        return

    # Initialize a log file in which the progress of processing, as well as parameters for processing are saved
    log_file, error = file_utils.initialize_log_file(os.path.join(saving_folder, "log.json"), os.path.join(data_path, "labeled_data", "log.json"), service, table)
    if error:
        logger.error(f"Initializing log file failed: {error}")
        return

    labeled_files, error = file_utils.reduce_files_to_handle(labeled_files, saving_folder, service)
    if error or labeled_files == []:
        return

    # Calculation may be speed up, when the files within a chunk have similar size
    labeled_files = file_utils.sort_by_filesizes(labeled_files)

    normalization_df, error = create_windows_utils.return_normalization_df(normalize, anonymize_file)
    if error:
        return

    # Each core needs all the params for the processing of the file
    list_of_file_dicts, error = create_windows_utils.create_list_of_file_dicts(labeled_files, needed_columns, window_length, step_length, filling_method, resampling_rate, data_columns, flatten, store_local, saving_folder, normalize, normalization_df)
    if error:
        return

    # Splitting into chunks enables for quick stop of execution, when errors occur
    file_chunks = data_utils.split_files_to_handle_into_chunks(list_of_file_dicts, 24)
    if len(file_chunks) > os.cpu_count():
        logger.info(f"Smaller files are processed first, therefore a slowing of the processing of chunks is to be expected")

    # As processing all the data might take some time, an estimation is given
    _, error = file_utils.estimate_needed_time(len(labeled_files), service)
    if error:
        return
    logger.info(f"Beginning to create windows for {len(labeled_files)} files")

    time_before = time.time()

    for chunk_number, chunk in enumerate(file_chunks, start=1):
        # Process the files in the chunk with all available CPU cores
        with Pool() as pool:
            result = pool.map(create_windows_utils.create_windows_for_file, chunk)

        # track current progress / success of the processing
        log_file = file_utils.create_log_from_result_strings(result, os.path.join(saving_folder, "log.json"), service)

        logger.info(f"[{chunk_number}/{max([len(file_chunks), chunk_number])}] chunks finished. {len(log_file[service]['successfully_processed_files'])} files successfully processed")

        # If errors occurred, warn the user and save the log file for data investigation purposes
        if len(log_file[service]["occurred_errors"]) != 0:

            logger.warning(f"These errors occurred: {log_file[service]['occurred_errors'].keys()}")
            logger.info(f"For debugging purposes the log file gets saved: Contains errors & respective files")

            error = file_utils.update_log_file(log_file, table, saving_folder, service)
            if error:
                logger.error(f"saving log file failed: {error}")
                return

        if store_local:
            if chunk_number == len(file_chunks) and len(log_file[service]["occurred_errors"]) == 0:
                log_file[service]["successfully_processed_files"] = "all"

            error = file_utils.update_log_file(log_file, table, saving_folder, service)
            if error:
                logger.error(f"saving log file failed: {error}")

    duration = round((time.time() - time_before)/60, 2)
    logger.info(f"Finished creating windows. Duration: {duration} min for {len(labeled_files)} files")
