import os
import json
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from collections.abc import Iterable

from config.log import logger


def get_files(complete_path: str, file_type="csv"):
    """ returns all file paths of a specified type of folder and all subfolder
    """
    if not os.path.exists(complete_path):
        logger.error(f"The given filepath does not exist")
        return [], "given_dir_does_not_exist"

    all_files = []
    for path, _, files in os.walk(complete_path):
        for filename in files:
            if not file_type:
                all_files.append(os.path.join(path, filename))
            elif filename.endswith(file_type):
                all_files.append(os.path.join(path, filename))

    if all_files == []:
        logger.error(f"Given dir {complete_path} does not contain any data of type {file_type} ")
        return [], "no_data_in_dir"

    return all_files, None


def gather_sessions_in_dict(list_of_window_files: list):
    """ returns a dict which maps each session to a list of all the windows from this session
    """
    sessions_dict = {}
    for window in list_of_window_files:
        session_name = window.split("/")[-1].split("(")[0]
        if session_name not in sessions_dict.keys():
            sessions_dict[session_name] = [window]
            continue
        sessions_dict[session_name].append(window)
    return sessions_dict


def contains_files_with_various_levels(folder_to_check: str, colname: str, max_files_to_check=10):
    """Iterates over the files in a given folder, returns True if at least one file contains various differing entries in given column 
    """
    # get all the files to iterate over
    all_files, error = get_files(folder_to_check, "csv")
    if error:
        return False, error

    for number, file in enumerate(all_files):
        # Check only the first few files, as multiple levels would occur there
        if number > max_files_to_check:
            return False, None

        if not os.path.exists(file):
            return False, "file_does_not_exist"

        try:
            df = pd.read_csv(file, usecols=[colname])
        except ValueError:
            logger.error(f"Could not find column {colname} in the columns of given file {file}")
            return False, f"column_{colname}_missing_from_file_{file}"

        if len(df[colname].unique()) > 1:
            return True, None

    return False, None


def debundle_files(path_to_files: str, colname: str):
    """For balancing files with a only one label are needed, therefore files of various levels are transformed into files of only one level 
    """
    logger.info(f"As the given data did contain files with various label levels, debundeling files was started")
    all_files_to_debundle, error = get_files(path_to_files, "csv")
    if error:
        return error

    for file in all_files_to_debundle:
        df = pd.read_csv(file)
        if colname not in df.columns:
            logger.error(f"No column {colname} found in file {file}")
            return f"column_{colname}_missing_from_file_{file}".replace('\\','/')

        all_levels = df[colname].unique()
        # If the file contains only one level of the column in question, no splitting is needed
        if len(all_levels) == 1:
            continue
        # Otherwise save the individual parts as own files
        for level in all_levels:
            df_to_save = df[df[colname] == level]
            # The new filename is simply the old filename with the level_ inserted
            error = save_df_with_key(
                file, path_to_files, df_to_save, str(level))
            if error:
                return error

        logger.debug(f"Created all subfiles for file {file}")

        # And delete the original file with various levels
        os.remove(file)
        logger.debug(f"Deleted unbundled file {file}")
    return None


def save_df_with_key(file: str, saving_folder: str, df: pd.DataFrame, key=None):
    """Save the given df in the saving_folder (filename begins with key_ , if key was given given)
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Got object df of wrong datatype. Expected pd.DataFrame, got {type(df)}")
        return "got_df_of_non_pdDataFrame_type"

    filename = file.replace('\\','/').split("/")[-1]
    if key is not None:
        filename = key + "_" + filename
    filepath = os.path.join(saving_folder, filename).replace("\\","/")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"]).reset_index(drop=True)

    df.to_csv(filepath, index=False)

    if not os.path.exists(filepath):
        return "file_does_not_exist_after_saving"

    return None


def save_json(json_formatted_object: dict, full_filename: str):
    """Saves an object  in json-format into given location
    """
    dir_name = os.path.dirname(full_filename)
    if not os.path.exists(dir_name):
        logger.warning(f"The given saving path for the json did not exist. Created it: {dir_name}")
        os.makedirs(dir_name)

    with open(full_filename, 'w') as f:
        json.dump(json_formatted_object, f)

    if not os.path.exists(full_filename):
        logger.error(f"Saving the file failed. Could not find {full_filename}")
        return "file_not_found_after_saving"

    return None


def estimate_needed_time(amount_of_files_to_process: int, service_or_method: str, available_cores=os.cpu_count(), size_of_files_to_process_in_GB=0):
    """Gives an estimate regarding the needed time for the processing
    """
    if service_or_method not in ["create_windows", "generate_features", "balancing_over", "balancing_evaluation", "reduce_data", "prepare_dataset", "evaluate_model"]:
        logger.error(f"Cannot estimate needed time for service {service_or_method}")
        return "", "invalid_service_or_method_given"

    if service_or_method == "create_windows":
        estimated_dur_min = round(amount_of_files_to_process / (available_cores*15))
    if service_or_method == "generate_features":
        estimated_dur_min = round(amount_of_files_to_process / (available_cores*70))
    if service_or_method == "balancing_over" or service_or_method == "balancing_evaluation":
        estimated_dur_min = round(amount_of_files_to_process / (available_cores*900))
    if service_or_method == "reduce_data":
        estimated_dur_min = round(amount_of_files_to_process / (available_cores*2100))
    if service_or_method == "prepare_dataset":
        estimated_dur_min = round(size_of_files_to_process_in_GB * 20 / available_cores)
    if service_or_method == "evaluate_model":
        estimated_dur_min = round(size_of_files_to_process_in_GB * 292 / available_cores)

    # Convert the time to hours if needed
    if estimated_dur_min < 60:
        estimated_dur = str(estimated_dur_min) + " min"
    if estimated_dur_min > 60:
        estimated_dur = str(round(estimated_dur_min/60, 2)) + " h"

    logger.info(f"The estimated time to execute {service_or_method} for  these {amount_of_files_to_process} files is {estimated_dur}")

    # this return is only used for testing purposes
    return estimated_dur_min, None


def reduce_files_to_handle(list_of_files_to_handle: list, target_location: str, service: str, verbose=True):
    """Reads in a log file and removes all the already processed files from the list of files to handle 
    """
    # Test if a log file exists in the target location (the folder to save into)
    if not os.path.exists(os.path.join(target_location, "log.json")):
        logger.error(f"No log file found at {target_location}")
        return list_of_files_to_handle, "no_log_in_target_location"

    with open(os.path.join(target_location, "log.json")) as f:
        log_file = json.load(f)

    # Stop executing, when the log does not contain any information regarding processed files
    if service not in log_file.keys():
        return [], f"log_file_does_not_contain_{service}_as_key"

    if "successfully_processed_files" not in log_file[service].keys():
        logger.error("The given log file does not contain any information regarding successfully_processed_files")
        return [], "log_file_does_not_contain_information_regarding_processed_files"

    if verbose:
        logger.info(f"Before removing files which were already processed, {len(list_of_files_to_handle)} files were found do handle")

    # Skip processing, when all has been finished already:
    if log_file[service]["successfully_processed_files"] == "all":
        logger.info("All files have been processed")
        return [], None

    # Remove the already processed files from the files_to_handle
    for file in log_file[service]["successfully_processed_files"]:
        try:
            list_of_files_to_handle.remove(file)
        except ValueError:
            logger.error(f"Already successfully processed file which is not in the list_of_files_to_handle now. Is the same database being processed?")
            return [], "successfully_processed_file_which_is_not_in_the_files_to_handle"

    # Return a unique list of the files to handle
    list_of_files_to_handle = list(set(list_of_files_to_handle))

    if verbose:
        logger.info(f"After removing files which were already processed, {len(list_of_files_to_handle)} are left to handle")

    return list_of_files_to_handle, None


def gather_filesizes_in_df(list_of_files: list):
    """returns a df with the files and their respective sizes
    """
    return pd.DataFrame({"file": list_of_files, "size": [os.stat(file).st_size / 1024 for file in list_of_files]})


def sort_by_filesizes(list_of_files: list, ascending=True):
    """returns the list of files ordered by their filesize
    """
    df_filesizes = gather_filesizes_in_df(list_of_files).sort_values("size", ascending=ascending)
    return [x for x in df_filesizes["file"]]


def detect_dataset(list_of_files: list):
    """uses the filenames to detect which dataset is currently handled
    """
    earliest_validation_session = 1664575200
    earliest_test_session = 1673514051

    earliest_observed_unix_time = min([create_unix_time_from_filename(file) for file in list_of_files])
    seen_dataset = "test_data"
    if earliest_observed_unix_time <= earliest_test_session:
        seen_dataset = "validation_data"
    if earliest_observed_unix_time <= earliest_validation_session:
        seen_dataset = "training_data"

    return seen_dataset


def create_unix_time_from_filename(file: str):
    """splits a full filename and converts the seen date to unix in seconds
    """
    return int(time.mktime(datetime.strptime(file.split("/")[-1].split("T")[0].split("_")[-1], "%Y-%m-%d").timetuple()))


def initialize_log_file(future_location: str, prior_location: str, service: str, table: dict, second_prior_location=None, third_prior_location = None):
    """initializes a log file: Reads in existing if it does exist, otherwise create it from the prior
    """
    # If the file exists in the target/future location, simply read and return it
    if os.path.exists(future_location):
        return read_and_check_existing_log_file(future_location, table, service)

    if os.path.exists(prior_location):
        log_file, error = bring_prior_existing_log_file_to_future_location(prior_location, service, table, future_location)
        if error:
            return {}, error

        if second_prior_location is not None:
            # When a second prior location has been given, save these parameters (contain e.g. the parameters for the test data)
            log_file = append_second_location(log_file, second_prior_location, service)
        if third_prior_location is not None:
            log_file = append_third_location(log_file, third_prior_location, service)

        log_file, error = valid_prior_preprocessing_steps_seen(service, log_file, table)
        if error:
            return {}, error

        return log_file, None

    # The service create_windows cannot find a log file
    if service == "create_windows":
        return initialize_empty_log_file(service, table, future_location)

    return {}, "log_file_neither_in_target_nor_prior_location"


def create_log_from_result_strings(list_of_result_strings: list, path_to_existing_log_file: str, service: str, separator="+++"):
    """Splits up the result strings to give information regarding the files which needed further investigation and those handled successfully. saves log if specified
    """
    log_file = {service: {
        "successfully_processed_files": [],
        "occurred_errors": {},
        "saved_files": 0}}

    if os.path.exists(path_to_existing_log_file):
        with open(path_to_existing_log_file) as f:
            log_file = json.load(f)

    list_of_result_strings = unpack_list_of_result_strings_if_needed(
        list_of_result_strings)

    for result_string in list_of_result_strings:

        result_string_splitted = result_string.split(separator)

        file = result_string_splitted[0]
        success = result_string_splitted[1] == "True"
        error = None if result_string_splitted[2] == "None" else result_string_splitted[2]
        files_saved = int(result_string_splitted[3])
        if success:
            log_file[service]["successfully_processed_files"].append(file)
            log_file[service]["saved_files"] += files_saved
            continue
        if error not in log_file[service]["occurred_errors"].keys():
            log_file[service]["occurred_errors"][error] = [file]
            continue
        log_file[service]["occurred_errors"][error].append(file)

    if len(log_file[service]["occurred_errors"]) != 0:
        for key in log_file[service]["occurred_errors"].keys():
            logger.warning(f"error {key} occurred {len(log_file[service]['occurred_errors'][key])} times")

    return log_file


def processing_flattened_data(log_file: dict):
    """ Returns True if the data of the given log file was flattened while creating windows
    """
    if "create_windows" not in log_file.keys():
        logger.error("create_windows not found in the keys")
        return False, "create_windows _not_found_in_the_keys"
    if "flatten" not in log_file["create_windows"].keys():
        logger.error("create_windows does not contain information regarding flattening")
        return False, "key_flatten_not_found_in_key_create_windows"

    return log_file["create_windows"]["flatten"], None


def processing_featured_data(log_file: dict):
    """ Returns True if the data of the given log file exists in featured form
    """
    return "generate_features" in log_file.keys()


def read_csv_safely(file: str, chunksize=1024, header=0):
    """reads in the file chunkwise
    """
    df_chunks = pd.read_csv(file, chunksize=chunksize, header=header)
    df = pd.concat([sub_df for sub_df in df_chunks])

    if "label" in df.columns:
        df["label"] = df["label"].astype(np.int16)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"]).reset_index(drop=True)

    return df


def count_filesizes_of_dir(folder: str):
    """ returns the filesizes in GB of all the files in given directory
    """
    if not os.path.exists(folder):
        logger.error(f"Given folder {folder} does not exist, counting filesizes not possible")
        return 0.0, "given_dir_does_not_exist"
    return sum([os.path.getsize(f) for f in os.listdir(folder) if os.path.isfile(f)]) / (1024**3), None


def save_csv_from_dict(dict_with_params: dict):
    """saves the df from given dict as the given filename
    """
    df = dict_with_params.get("df")
    if df is None:
        return "no_df_for_saving_given"
    if not isinstance(df, pd.DataFrame):
        return "got_df_of_non_pdDataframe_type"

    filename = dict_with_params.get("filename")
    if filename is None:
        return "no_filename_for_saving_given"

    df.to_csv(filename, index=False, header=dict_with_params.get("save_column_names"))

    if not os.path.exists(filename):
        return "file_does_not_exist_after_saving"

    return None


def save_csvs_labelwise(df: pd.DataFrame, folder_to_store_to: str, grid=False):
    """saves an individual csv file for all the seen label classes in the df
    """
    if not os.path.exists(folder_to_store_to):
        os.makedirs(folder_to_store_to)

    # Split the df into sub_dfs, each containing only a single label
    list_of_dfs = [df[df["label"] == label] for label in list(set(df["label"]))]

    # create a list of dicts, each containing the saving place and the filename
    list_dicts = [{"df": sub_df.reset_index(drop=True),
                   "filename": os.path.join(folder_to_store_to,  str(sub_df.reset_index(drop=True)["label"][0]) + f"_prepared.csv"),
                   "conv": grid,
                   "save_column_names": True} for sub_df in list_of_dfs]


    # use multiple cores to speed up the saving process
    with Pool() as pool:
        result = pool.map(save_csv_from_dict, list_dicts)

    occurred_errors = [x for x in result if x is not None]
    if occurred_errors != []:
        logger.error(f"Encountered these errors while saving df: {occurred_errors}")
        return occurred_errors

    return None


def remove_outdated_processed_files(log_file: dict, current_processing_step: str):
    """delete list of successfully_processed_files, if these simply contain all files from the respective input folder
    """
    methods_to_check = []
    if current_processing_step == "generate_features":
        methods_to_check = ["create_windows"]
    if current_processing_step == "reduce_data":
        methods_to_check = ["create_windows", "generate_features"]
    if current_processing_step == "balancing_over":
        methods_to_check = ["create_windows",
                            "generate_features", "reduce_data"]
    if current_processing_step == "prepare_large_dataset":
        methods_to_check = ["create_windows", "generate_features", "reduce_data", "balancing_over"]

    for method_to_check in methods_to_check:
        # Not every log file must contain all possible prior steps, therefore skipping must be possible
        if method_to_check not in log_file.keys():
            continue

        if log_file[method_to_check]["successfully_processed_files"] == "all":
            continue

        if method_to_check == "create_windows":
            files_in_input_folder, error = get_files(os.path.join(log_file[method_to_check]["data_path"], "labeled_data"), "csv")
            if error:
                return {}, error
            # When the the amount of input files matches the amount of successfully_processed_files, simply safe that all files were processed
            if len(files_in_input_folder) == len(log_file[method_to_check]["successfully_processed_files"]):
                log_file[method_to_check]["successfully_processed_files"] = "all"

        if method_to_check == "generate_features":
            # As the data from balancing may or may not be in featured form, skipping is allowed
            if method_to_check not in log_file.keys():
                continue

            if "saving_folder" in log_file["create_windows"].keys():
                input_folder = log_file["create_windows"]["saving_folder"]
            else:
                input_folder = os.path.join(log_file["create_windows"]["data_path"], "windows", log_file["create_windows"]["output_date"])
            files_in_input_folder, error = get_files(input_folder, "csv")
            if error:
                return {}, error

            if len(files_in_input_folder) == len(log_file[method_to_check]["successfully_processed_files"]):
                log_file[method_to_check]["successfully_processed_files"] = "all"

        if method_to_check == "reduce_data":
            # As the data from balancing may or may not be in featured form, skipping is allowed
            if method_to_check not in log_file.keys():
                continue

            if "saving_folder" in log_file["generate_features"].keys():
                input_folder = log_file["generate_features"]["saving_folder"]
            else:
                input_folder = os.path.join(log_file["generate_features"]["data_path"], "features", log_file["generate_features"]["saving_time"])

            files_in_input_folder, error = get_files(input_folder, "csv")
            if error:
                return {}, error

            if len(files_in_input_folder) == len(log_file[method_to_check]["successfully_processed_files"]):
                log_file[method_to_check]["successfully_processed_files"] = "all"

        if method_to_check == "balancing_over":
            max_saved_files = resulting_files_per_granularity_level(log_file, current_processing_step)
            if log_file[method_to_check]["saved_files"] == max_saved_files and len(log_file[method_to_check]["occurred_errors"]) == 0:
                log_file[method_to_check]["successfully_processed_files"] = "all"

    return log_file, None


def resulting_files_per_granularity_level(log_file: dict, current_processing_step: str):
    """returns the amount of resulting files after saving one file per label
    """
    if processing_featured_data(log_file):
        granularity = log_file[current_processing_step]["granularity"]
        return {"top": 3, "mid": 6}[granularity]

    return -1


def update_log_file(log_file: dict, table: dict, saving_folder: str, processing_step: str):
    """updates the log file in the saving folder with the params from the table
    """
    # Add the currently used params to the log file
    for key in table.keys():
        log_file[processing_step][key] = table[key]

    log_file, error = remove_outdated_processed_files(log_file, processing_step)
    if error:
        return error

    error = save_json(log_file, os.path.join(saving_folder, "log.json"))
    if error:
        return error

    return None


def processing_windowed_data(log_file: dict):
    """ Returns True if the data of the given log file exists in windowed form
    """
    return "create_windows" in log_file.keys()


def get_unique_levels_from_folder(folder: str, col_name: str, data_type, file_type: str):
    """returns a list of unique values of the column of all files in folder
    """
    all_files, error = get_files(folder, file_type)
    if error:
        return None, error

    unique_levels = np.ndarray((0))

    for file in all_files:
        unique_levels = np.append(unique_levels, pd.read_csv(file, usecols=[col_name], dtype={col_name: data_type})[col_name].unique())

    return [int(x) for x in np.unique(unique_levels).astype(int)], None


def count_filesizes_of_list(list_of_files: list):
    """returns the sum of the filesizes in GB
    """
    return round(sum([os.stat(x).st_size for x in list_of_files])/1000000000, 2)


def list_of_maxsized_filechunks(list_of_files: list, maxsize_in_GB: float):
    """returns a list of chunks of files, where the files of each chunk are smaller than the maxsiz_in_GB
    """
    # Check if all the files exist
    non_existing_files = [file for file in list_of_files if not os.path.exists(file)]
    if non_existing_files:
        logger.error(f"Got {len(non_existing_files)} files to process which do not exist")
        return [non_existing_files[0]], "got_non_existing_files_too_process"

    # Count the size of all these files
    joined_size_all_files = count_filesizes_of_list(list_of_files)
    if joined_size_all_files < maxsize_in_GB:
        return [list_of_files], None

    # calculate the number of needed chunks
    nr_chunks = math.ceil(joined_size_all_files/maxsize_in_GB)

    # Calculate the chunksize
    chunksize = math.ceil(len(list_of_files)/nr_chunks)
    
    # Return the list of chunks, last chunk may contain less files
    return [list_of_files[i:i + chunksize] for i in range(0, len(list_of_files), chunksize)], None


def processing_oversampled_data(log_file: dict):
    """ Returns True if the data of the given log file exists was oversampled
    """
    return "balancing_over" in log_file.keys()


def processing_balanced_data(log_file: dict):
    return processing_oversampled_data(log_file) or "balancing_evaluation" in log_file.keys()


def unpack_list_of_result_strings_if_needed(list_of_result_strings: list):
    """As the list_of_result_strings may contain lists with the result strings depacking might be needed
    """
    if not list_of_result_strings:
        return []
    if isinstance(list_of_result_strings[0], str):
        return list_of_result_strings
    return flatten_nested_lists(list_of_result_strings)


def flatten_create_generator(xs: list):
    """flattens any nested list to a list not containing any other lists
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_nested_lists(x)
        else:
            yield x


def flatten_nested_lists(nested_list: list):
    """ returns a list with depth 1
    """
    return [y for y in flatten_create_generator(nested_list)]


def read_and_check_existing_log_file(file_location: str, table: dict, service: str):
    """ Return file after checking if the found parameters are identical to the currently seen ones
    """
    with open(file_location) as f:
        log_file: dict = json.load(f)

    # Check if the earlier given params match with the currently given params in the table
    for key in table.keys():
        # When training was interrupted, none of the keys might exist, therefore they need to be created
        if key not in log_file[service].keys():
            log_file[service][key] = table.get(key)
            continue
        # if the log file was saved, the keys exist and the values can be compared
        if log_file[service][key] != table.get(key):
            logger.warning(f"Found mismatch between parameters: Current Iteration got parameter {table.get(key)} for {key}, parameter of earlier execution was {log_file[service][key]} ")
            return {}, "parameters_of_current_iteration_not_identical_to_previous"

    return log_file, None


def initialize_empty_log_file(service: str, table: dict, future_location: str):
    """initializes a new, empty log file and saves it at the future_location
    """
    log_file = {service: {}}

    # Initialize the params
    log_file[service] = {"successfully_processed_files": [],
                         "occurred_errors": {},
                         "saved_files": 0}

    # Append the current parameters
    for key in table.keys():
        log_file[service][key] = table.get(key)

    error = save_json(log_file, future_location)
    if error:
        return {}, "saving_log_in_future_location_failed"

    return log_file, None


def bring_prior_existing_log_file_to_future_location(prior_location: str, service: str, table: dict, future_location: str):
    """copies the existing log file from the prior location with the new params of the table to the future_location 
    """
    with open(prior_location) as f:
        log_file = json.load(f)

    # Append needed variables for the specific service (might be equal for all services, depends on future implementation)
    log_file[service] = {"successfully_processed_files": [],
                         "occurred_errors": {},
                         "saved_files": 0}

    # Append the current parameters
    for key in table.keys():
        log_file[service][key] = table.get(key)

    error = save_json(log_file, future_location)
    if error:
        logger.error(f"Saving the log file failed")
        return {}, "saving_log_in_future_location_failed"

    return log_file, None


def append_second_location(log_file: dict, second_prior_location: str, service: str):
    """appends the log file of the second location under <service>_second_location
    """
    with open(second_prior_location) as f:
        second_log_file = json.load(f)
    log_file[f"{service}_second_location"] = second_log_file
    return log_file


def append_third_location(log_file: dict, third_prior_location: str, service: str):
    """appends the log file of the third location under <service>_third_location
    """
    with open(third_prior_location) as f:
        second_log_file = json.load(f)
    log_file[f"{service}_third_location"] = second_log_file
    return log_file


def valid_prior_preprocessing_steps_seen(service: str, log_file: dict, table: dict):
    """checks if the necessary prior steps have been performed
    """
    if service == "prepare_dataset":
        if "balancing_over" not in log_file.keys() and "balancing_evaluation" not in log_file.keys():
            logger.error("Tried to prepare dataset for training without balancing the data beforehand")
            return {}, "tried_preparing_dataset_without_balancing before"

        used_balancing = [key for key in log_file.keys() if "balancing" in key][0] 

        if log_file.get(used_balancing).get("granularity") != table.get("granularity"):
            logger.error(f"Tried preparing dataset with differing granularity ")
            return {}, f"tried_preparing_dataset_without_differing_granularity_level_chose_{log_file.get(used_balancing).get('granularity')}_for_balancing_received_{table.get('granularity')}"

    return log_file, None


def valid_anonymize_file_found(datapath: str):
    """Checks if the anonymize file does contain all the necessary information
    """
    if not os.path.exists(datapath):
        logger.error(f"Given Datapath {datapath} does not exist")
        return "datapath_does_not_exist"

    if not os.path.exists(os.path.join(datapath, "anonymization_file.json")):
        logger.error(f"Given Datapath  {datapath} does not contain a anonymization_file.json")
        return "anonymization_file.json_does_not_exist"

    with open(os.path.join(datapath, "anonymization_file.json")) as f:
        anonymize_file: dict = json.load(f)

    if not set(["create_windows", "svm_features", "feature_utils"]).issubset(set(anonymize_file.keys())):
        logger.error("Not all required keys were found in the anonymization_file.json")
        return "not_all_required_keys_found"

    if not set(["needed_columns", "data_columns"]).issubset(set(anonymize_file.get("create_windows").keys())):
        logger.error("Key create_windows does not contain all necessary subkeys")
        return "create_windows_does_not_contain_all_necessary_subkeys"

    if not set(["top", "lifting", "walking"]).issubset(set(anonymize_file.get("svm_features").keys())):
        logger.error("Key svm_features does not contain all necessary subkeys")
        return "svm_features_does_not_contain_all_necessary_subkeys"

    if not set(["accel_cols", "differences_cols", "needed_cols", "on_single_column", "magn_feat_name"]).issubset(set(anonymize_file.get("feature_utils").keys())):
        logger.error("Key feature_utils does not contain all necessary subkeys")
        return "feature_utils_does_not_contain_all_necessary_subkeys"

    for key in anonymize_file.keys():
        for subkey in anonymize_file.get(key).keys():
            if not isinstance(anonymize_file.get(key).get(subkey), list):
                logger. error("Found object in anonymization_file.json which is not a list")
                return "non_list_object_found"

    return None
