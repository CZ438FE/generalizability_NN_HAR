import os
import sys
import time
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

from utils import preprocess_utils, file_utils, plot_utils, data_utils
from config.log import logger

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def add_args(group):
    """ Description of possible arguments for preprocessing of data
    """
    group.add_argument(
        '-i', '--input_data',
        metavar='',
        dest='input_data',
        required=True,
        type=str,
        help='Specify the absolute path to the folder in which the data to preprocess exists'
    )
    group.add_argument(
        '-m', '--method',
        metavar='',
        dest='method',
        type=str,
        choices=['reduce_data', 'balancing_over', 'balancing_evaluation', 'prepare_dataset'],
        required=True,
        help='Specify which preprocessing method you want to apply to the data, choices: [%(choices)s]'
    )
    group.add_argument(
        '-g', '--granularity',
        metavar='',
        dest='granularity',
        type=str,
        choices=['top', 'mid'],
        default="mid",
        help='Specify under which granularity to save the label column of the given files, choices: [%(choices)s]'
    )
    group.add_argument(
        '--grid',
        action='store_const',
        const=True,
        default=False,
        help='Set if you want to prepare the data for training a CNN or RNN (only has affect when method prepare_dataset is chosen)'
    )
    group.add_argument(
        '-p', '--plot',
        action='store_const',
        const=True,
        default=False,
        help='Set if you want to plot the results'
    )
    group.add_argument(
        '-classes_agg', '--classes_to_aggregate',
        metavar='\b',
        dest='classes_to_aggregate',
        type=str,
        nargs='+',
        required=False,
        default=[],
        help='The classes (labels) to be integrated into a new common class'
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
    if "dryrun" not in table:
        table["dryrun"] = False
    if "output_date" not in table:
        table["output_date"] = None
    if "grid" not in table:
        table["grid"] = False
    if "plot" not in table:
        table["plot"] = False
    if table.get("classes_to_aggregate") is None:
        table['classes_to_aggregate'] = []
    return table


def preprocess(table: dict):
    """ Brings Data into correct format for training of models
    """
    error = preprocess_utils.valid_table(table)
    if error:
        return

    data_path: str = table.get("data_path")
    store_local: bool = not table.get("dryrun")
    input_data_folder: str = table.get("input_data")
    method: str = table.get("method")
    output_date = table.get("output_date")
    granularity: str = table.get("granularity")
    grid: bool = table.get("grid")
    classes_to_aggregate : list = table.get("classes_to_aggregate")
    plot: bool = table.get("plot")
    # When the results are being stored and the respective folders do not exist yet, create them at <dp>/preprocess/<method>/<output_date>
    saving_folder = os.path.join(data_path, "preprocess", method, output_date)
    table["saving_folder"] = saving_folder
    if store_local and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    log_file, error = file_utils.initialize_log_file(os.path.join(saving_folder, "log.json"), os.path.join(input_data_folder,  "log.json"), method, table)
    if error:
        logger.error(f"Initializing log file failed: {error}")
        return

    # Knowing the label depth is needed for appropriate reduction of features and correct balancing
    label_depth, error = preprocess_utils.appropriate_label_depth(granularity)
    if error:
        return
    table["label_depth"] = label_depth

    if method == "reduce_data":
        reduce_data(input_data_folder, saving_folder, method, label_depth, data_path, table, store_local, plot, granularity)
    elif "balancing" in method:
        balance_data(input_data_folder, label_depth, saving_folder, method, plot, table, store_local, log_file, classes_to_aggregate)
    elif method == "prepare_dataset":
        prepare_dataset(log_file, table, input_data_folder, saving_folder, method, label_depth, store_local, grid)
    else:
        logger.error(f"Received invalid method {method}. No preprocessing performed")
        return


def reduce_data(input_data_folder: str, saving_folder: str, method: str, label_depth: int, data_path: str, table: dict, store_local: bool, plot: bool, granularity: str):
    """Reduces the set of generated features to the set of features which is required by the respective model
    """
    all_unreduced_files, error = file_utils.get_files(input_data_folder, ".csv")
    if error:
        return

    all_unreduced_files, error = file_utils.reduce_files_to_handle(all_unreduced_files, saving_folder, method)
    if error or all_unreduced_files == []:
        return

    # Different models have different subsets of features available to them, so the data for these models needs to be reduced accordingly
    needed_columns, error = preprocess_utils.return_svm_features(label_depth, data_path)
    if error:
        return
    table["needed_columns"] = needed_columns

    list_of_file_dicts = preprocess_utils.create_list_of_file_dicts(all_unreduced_files, table)

    # multiprocessing is used, which is being done chunk-wise
    file_chunks = data_utils.split_files_to_handle_into_chunks(list_of_file_dicts, 768)

    _, error = file_utils.estimate_needed_time(len(all_unreduced_files), method)
    if error:
        return
    time_before = time.time()

    for chunk_number, chunk in enumerate(file_chunks, start=1):
        with Pool() as pool:
            result = pool.map(preprocess_utils.reduce_data_of_file, chunk)

        # track current progress / success of the processing
        log_file = file_utils.create_log_from_result_strings(result, os.path.join(saving_folder, "log.json"), method)

        logger.info(f"[{chunk_number}/{max([len(file_chunks), chunk_number])}] chunks finished. {len(log_file[method]['successfully_processed_files'])} files successfully processed")

        # If errors were found, warn the user and save the log file for data investigation purposes
        if len(log_file[method]["occurred_errors"]) != 0:
            logger.warning(f"These errors occurred: {log_file[method]['occurred_errors'].keys()}")
            logger.info(f"For debugging purposes the log file gets saved: Contains errors & respective files")

            error = file_utils.update_log_file(log_file, table, saving_folder, method)
            if error:
                logger.error(f"saving log file failed: {error}")

        if store_local:
            if chunk_number == len(file_chunks) and len(log_file[method]["occurred_errors"]) == 0:
                log_file[method]["successfully_processed_files"] = "all"

            error = file_utils.update_log_file(log_file, table, saving_folder, method)
            if error:
                logger.error(f"saving log file failed: {error}")

    duration = round((time.time() - time_before)/60, 2)
    logger.info(f"Finished reducing data. Duration: {duration} min for {len(all_unreduced_files)} files")

    if plot:
        labels_overview_dict = preprocess_utils.count_label_distribution(all_unreduced_files, label_depth)
        if not labels_overview_dict:
            logger.error(f"Counting the labels failed. Investigation needed")
            return
        plot_utils.show_label_distribution(granularity, labels_overview_dict["classes_count"], saving_folder, store_local)


def oversample(labels_overview_dict: dict, label_depth: int, method: str, plot: bool, table: dict, saving_folder: str, store_local: bool, log_file: dict):
    """Applies balancing via oversampling according to given parameters
    """
    # Remove unneeded label_classes
    labels_overview_dict["classes_count"], removed_keys = preprocess_utils.remove_labels_to_skip(labels_overview_dict["classes_count"])
    labels_overview_dict["classes_files"], _ = preprocess_utils.remove_labels_to_skip(labels_overview_dict["classes_files"])
    if len(removed_keys) != 0:
        logger.info(f"As the keys {removed_keys} are not used in the further processing, they are dropped from the oversampling")

    logger.info(f"This amount of samples exists for each class: {labels_overview_dict.get('classes_count')}")

    hierarchical = "reduce_data" in log_file.keys() and label_depth > 1
    needed_samples = preprocess_utils.calculate_needed_extra_samples(labels_overview_dict, method, hierarchical)

    logger.info(f"This amount of samples is needed for each class: {needed_samples}")
    log_file[method]["needed_samples"] = needed_samples

    # Calculate the proportion of true data on all data after oversampling would be
    prop_original_data_dict = preprocess_utils.calculate_prop_original_data_dict(needed_samples, labels_overview_dict)
    if not prop_original_data_dict:
        return

    #  Warn the user, when at least one class contains significantly less data than other classes
    dangerous_amount_of_resampling_detected = preprocess_utils.dangerously_much_oversampling_needed(prop_original_data_dict)
    if dangerous_amount_of_resampling_detected is None:
        return

    if plot:
        plot_utils.plot_original_samples_on_data_after_resampling(prop_original_data_dict, saving_folder, store_local)

    logger.info("Starting with naive oversampling")

    seed = 42
    table["seed"] = seed

    keys_with_params = preprocess_utils.key_with_params(needed_samples, labels_overview_dict, label_depth, store_local, saving_folder, seed)

    # remove a key if all the files have been processed already
    keys_with_params = preprocess_utils.remove_processed_key_with_params(keys_with_params, log_file, method)
    if not keys_with_params:
        logger.info("All files have been oversampled already")
        log_file[method]["successfully_processed_files"] = "all"
        error = file_utils.update_log_file(log_file, table, saving_folder, method)

    time_before = time.time()

    for key_with_params_nr, key_with_params in enumerate(keys_with_params, start=1):
        result = preprocess_utils.perform_naive_oversampling(key_with_params)

        # Create a log which contains information regarding the current progress / success of the processing
        log_file = file_utils.create_log_from_result_strings(result, os.path.join(saving_folder, "log.json"), method)

        # If errors were found, warn the user and save the log file for data investigation purposes
        if len(log_file[method]["occurred_errors"]) != 0:
            logger.warning(f"These errors occurred: {log_file[method]['occurred_errors'].keys()}")
            logger.info(f"For debugging purposes the log file gets saved: Contains errors & respective files")
            # Update the log file
            error = file_utils.update_log_file(log_file, table, saving_folder, method)
            if error:
                logger.error(f"saving log file failed: {error}")

        # Save the log file
        if store_local:
            if key_with_params_nr == len(keys_with_params) and len(log_file[method]["occurred_errors"]) == 0:
                log_file[method]["successfully_processed_files"] = "all"

            error = file_utils.update_log_file(log_file, table, saving_folder, method)
            if error:
                logger.error(f"saving log file failed: {error}")

    duration = round((time.time() - time_before)/60, 2)
    logger.info(f"Finished oversampling data. Duration: {duration} min")


def balance_data(input_data_folder: str, label_depth: int, saving_folder: str, method: str, plot: bool, table: dict, store_local: bool, log_file: dict, classes_to_aggregate: list):
    """Apply the chosen balancing type to the dataset
    """
    # Ensure that each input file contains only one label
    has_various_levels, error = file_utils.contains_files_with_various_levels(input_data_folder, "label")
    if error:
        return
    if has_various_levels:
        error = file_utils.debundle_files(input_data_folder, "label")
        if error:
            return

    all_unbalanced_files, error = file_utils.get_files(input_data_folder, ".csv")
    if error:
        return

    # For processing the information is needed, which files contain which label and how the overall label_distribution is
    labels_overview_dict = preprocess_utils.count_label_distribution(all_unbalanced_files, label_depth)


    all_unbalanced_files, error = file_utils.reduce_files_to_handle(all_unbalanced_files, saving_folder, method)
    if error or all_unbalanced_files == []:
        return

    # Give an estimation of the needed time
    _, error = file_utils.estimate_needed_time(len(all_unbalanced_files), method)
    if error:
        return

    if method == "balancing_over":
        oversample(labels_overview_dict, label_depth, method, plot, table, saving_folder, store_local, log_file)
    elif method =="balancing_evaluation":
        balance_for_evaluation(labels_overview_dict, saving_folder, store_local, log_file, table, label_depth, classes_to_aggregate)


def prepare_dataset(log_file: dict, table: dict, input_data_folder: str, saving_folder: str, method: str, label_depth: int, store_local: bool, grid: bool):
    """Prepares the balanced data for easy batch-wise accessing for model training and evaluation
    """
    # Data preparation depends on prior processing steps
    windowed = file_utils.processing_windowed_data(log_file)
    featured = file_utils.processing_featured_data(log_file)
    flattened, error = file_utils.processing_flattened_data(log_file)
    if error:
        return

    if windowed:
        seed = 42
        table["seed"] = seed

        if featured and not flattened:
            preprocess_utils.handle_data_for_feature_based_ffnn(input_data_folder, saving_folder, method, log_file, label_depth, table, store_local)

        if not featured and flattened:
            preprocess_utils.prepare_dataset_for_flattened_data(input_data_folder, saving_folder, method, log_file, label_depth, table, store_local, grid)

    else:
        logger.info(f"Preparing a dataset can only be applied to data which exists in form of windows")


def balance_for_evaluation(labels_overview_dict: dict, saving_folder: str, store_local: bool, log_file: dict, table: dict, label_depth: int, classes_to_aggregate: list):
    """Keeps all samples and aggregates labels if specified (the balancing for the results is being done in evaluate_model)
    """
    samples_to_keep = labels_overview_dict['classes_count']
    files_to_keep = {}

    labels_to_skip = ["13", "14", "24", "4", "41", "42"]
    for key in labels_to_skip:
        if key in samples_to_keep:
            del samples_to_keep[key]

    if classes_to_aggregate and not set(classes_to_aggregate).issubset(set(labels_overview_dict['classes_files'].keys())):
        logger.error(f"The set of labels to aggregate {classes_to_aggregate} is not a subset of the available labels {labels_overview_dict['classes_files'].keys()}")
        return 

    for label in samples_to_keep:
        files_to_keep[label] = labels_overview_dict['classes_files'][label]

        all_data_of_label = pd.concat([file_utils.read_csv_safely(file) for file in files_to_keep[label]])
        all_window_uuids_of_label = [file.split("/")[-1] for file in files_to_keep[label]]
        # For hierarchical classification both the row containing the data for the top-level data and tne mid-level data need to contain the window_uuid
        divisor = len(all_data_of_label) / len(all_window_uuids_of_label) 
        if divisor != 1:
            filled_window_uuids_of_label = []
            [filled_window_uuids_of_label.extend([elem]*int(divisor)) for elem in all_window_uuids_of_label]
            all_window_uuids_of_label = filled_window_uuids_of_label

        all_data_of_label["window_uuid"] = all_window_uuids_of_label
        if label in classes_to_aggregate:
            all_data_of_label['label'] = int(classes_to_aggregate[0])

        all_data_of_label, error = preprocess_utils.adjust_granularity_of_label(all_data_of_label, "label", label_depth)
        if error:
            return

        # Save each balanced_df in an own file
        if store_local:
            error = file_utils.save_df_with_key(f"{label}.csv", saving_folder, all_data_of_label)
            if error:
                logger.error("Failed while trying to save balanced df")
                return

    log_file['balancing_evaluation']['successfully_processed_files'] = "all"
    file_utils.update_log_file(log_file, table, saving_folder, "balancing_evaluation")

