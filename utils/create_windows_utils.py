import os
import sys
import pandas as pd
import numpy as np

from config.log import logger
from utils import time_utils, data_utils, file_utils


def prepare_time_col(df: pd.DataFrame, file: str):
    """Brings sci notation to ints, Sorting time col, bringing to same precision, Conversion to Unix, validity checks 
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected df of datatype pd.DataFrame, got {type(df)} for file {file}")
        return pd.DataFrame(), f"expected_pdDataFrame_got_{type(df)}"

    if not "time" in df.columns:
        logger.error(f"Malformed df given: Could not find a col time in file {file}")
        return pd.DataFrame(), "time_col_missing"

    # Remove all rows with missing values in the time col, as it is not known when the seen data occurred
    df = df[df["time"].notna()]

    # Time col might be in sci notation, therefore transformation might be needed
    if isinstance(df["time"][0], str):
        df, error = time_utils.transform_sci_not(df, "time")
        if error:
            logger.error(f"Given error occurred at file {file}")
            return pd.DataFrame(), error

    if not isinstance(df["time"][0], np.int64):
        error = 'got_col_time_of_nonint_dt'
        logger.error(f"Got time col of non int dt")
        return pd.DataFrame(), error

    df = df.sort_values("time", ascending=True).reset_index(drop=True)

    # Enforce identical precision in the time column
    if not data_utils.has_identical_length(df, "time"):
        logger.info(f"Got df with varying degrees of precision in the time col for file {file} . Transform time col")
        df, error = time_utils.bring_col_to_greatest_len(df, "time")
        if error:
            logger.error(f"Given error {error} occurred at file {file}")
            return pd.DataFrame(), error

    # Transfer the time column into ms in UNIX time
    df, error = time_utils.transform_time_unit(df, "ms", "time")
    if error:
        logger.error(f"Transforming the time column failed for file {file}.")
        return pd.DataFrame(), error

    error = time_utils.error_in_time_col(df, "time")
    if error:
        logger.error(f"Found inconsistencies in time col within file {file}")
        return pd.DataFrame(), error

    # Test if the conversion of the time column was successful
    if not data_utils.has_identical_length(df, "time"):
        logger.error(f"Df has varying degrees of precision after transforming time col in file {file}")
        return pd.DataFrame(), "varying_precision_even_after_transforming_time_col"

    return df, None


def create_list_of_windows(df: pd.DataFrame, window_length: int, step_length: int, file: str):
    """Creates a list of boundaries for the windows to create from the df
    """
    if "time" not in df.columns:
        logger.error(f"no time col found in given df")
        return [], "no_time_col_found"
    # Stores the starting and endpoints of the windows
    windows = []

    # Create a starting- and an endtime as the limits for the windows to be produced
    start_of_window = df["time"][0]
    last_datapoint = df["time"][len(df)-1]

    # Calculate the end of the first window
    end_of_current_window = start_of_window + window_length

    # Create a list containing the start- and endvalues of the windows to save
    while end_of_current_window <= last_datapoint:

        windows.append([int(start_of_window), int(end_of_current_window)])
        start_of_window += step_length
        end_of_current_window += step_length

    seen_time_in_df = last_datapoint - start_of_window

    if windows == [] and seen_time_in_df > window_length:
        # If the file is long enough to generate at least one window, but the list of windows to generate is empty, investigation is needed
        logger.error(f"Could not generate any windows a file. Investigate file {file}")
        return windows, "generating_list_of_windows_failed"

    return windows, None


def handle_window(df: pd.DataFrame, window: list, df_filled: pd.DataFrame, data_columns: list, flatten: bool, file: str, window_length: int, resampling_rate: int, store_local: bool, saving_folder: str, windows_saved_current_file: int):
    """Saves the (flattened) window if it does contain unfilled data, updates amount of saved windows and return an error message
    """
    if not isinstance(df, pd.DataFrame):
        return None, windows_saved_current_file, "given_df_no_pdDataFrame"

    if not isinstance(df_filled, pd.DataFrame):
        return None, windows_saved_current_file, "given_df_filled_no_pdDataFrame"

    if "time" not in df.columns:
        logger.error(f"no time col found in given df")
        return None, windows_saved_current_file, "no_time_col_found"

    if "label" not in df.columns:
        logger.error(f"no time col found in given df")
        return None, windows_saved_current_file, "no_label_col_found"

    if "time" not in df_filled.columns:
        logger.error(f"no time col found in given df")
        return None, windows_saved_current_file, "no_time_col_found_in_df_filled"

    if "label" not in df_filled.columns:
        logger.error(f"no time col found in given df")
        return None, windows_saved_current_file, "no_label_col_found_in_df_filled"

    if not isinstance(window[0], int) or not isinstance(window[1], int):
        logger.error(f"got wrong datatypes for the window_boundaries: expected int, got {[type(x) for x in window]}")
        return None, windows_saved_current_file, "window_boundaries_not_ints"

    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Only save a windows, if real data has been recorded within this period
    # select the period which could be saved from the not-filled-df
    first_time_index = data_utils.find_first_index_bigger_x_in_sorted_list([x for x in df["time"]], window[0]-1)
    last_time_index = data_utils.find_first_index_bigger_x_in_sorted_list([x for x in df["time"]], window[1])
    df_to_save = df.iloc[first_time_index:last_time_index,:].reset_index(drop=True)

    # Check if there are real, non-filled datapoints within this time, if not skip saving of this file
    err = data_utils.only_nan(df_to_save, data_columns)
    if err:
        logger.debug(f"Got a window with no real data within the df, skipped saving")
        return False, windows_saved_current_file, None

    # Else Take the df to save from the respective part of the prepared filled_df
    # This is done to preserve as much information as possible

    # As the original df and the df_filled may contain differing rows and row numbers, a recalculation is needed
    first_time_index = data_utils.find_first_index_bigger_x_in_sorted_list([x for x in df_filled["time"]], window[0]-1)
    last_time_index = data_utils.find_first_index_bigger_x_in_sorted_list([x for x in df_filled["time"]], window[1])
    df_to_save = df_filled.iloc[first_time_index:last_time_index, :].reset_index(drop=True)

    if flatten:
        df_to_save, error = data_utils.flatten(df_to_save, file, window_length, resampling_rate)
        if error:
            return None, windows_saved_current_file, error

    if store_local:
        if "win" in sys.platform:
            file = file.replace("/", "\\").replace("\\","/")
        new_filename = file.split("/")[-1].replace(".csv", f"_{windows_saved_current_file +1}.csv")
        full_path_to_new_file = os.path.join(saving_folder, new_filename)
        if "win" in sys.platform:
            full_path_to_new_file = full_path_to_new_file.replace("/", "\\").replace("\\","/")

        df_to_save.to_csv(full_path_to_new_file, index=False)

        if not os.path.exists(os.path.join(saving_folder, new_filename)):
            return None, windows_saved_current_file, "saving_file_failed"

        windows_saved_current_file += 1

    return True, windows_saved_current_file, None


def create_windows_for_file(file_with_attributes: dict):
    """ Converts time col, cuts df into windows of window_size and step_length, flattens and saves 
    """
    file = file_with_attributes.get("file")
    needed_columns = file_with_attributes.get("needed_columns")
    window_length = file_with_attributes.get("window_length")
    step_length = file_with_attributes.get("step_length")
    filling_method = file_with_attributes.get("filling_method")
    resampling_rate = file_with_attributes.get("resampling_rate")
    data_columns = file_with_attributes.get("data_columns")
    flatten = file_with_attributes.get("flatten")
    store_local = file_with_attributes.get("store_local")
    saving_folder = file_with_attributes.get("saving_folder")
    normalize = file_with_attributes.get("normalize")
    normalization_df = file_with_attributes.get("normalization_df")

    if not os.path.exists(file):
        return data_utils.create_result_string(file, [1], 0, "file_not_found")

    df = pd.read_csv(file, usecols=needed_columns)[needed_columns]

    # Check if the df contains all the needed columns
    if list(df.columns) != needed_columns:
        logger.error(f"Columns of file do not match the needed columns. Please investigate {file}")
        return data_utils.create_result_string(file, [1], 0, "found_col_do_not_match_needed_col")

    # Prepare the time col: Sorting time col, bringing to same precision, Conversion to Unix, validity checks
    df, error = prepare_time_col(df, file)
    if error:
        return data_utils.create_result_string(file, [1], 0, error)

    if normalize == "z_normalize":
        df[data_columns] = data_utils.z_normalize_df(df[data_columns].copy(True), normalization_df)
    elif normalize == "max":
        df[data_columns] = data_utils.max_scale_df(df[data_columns], normalization_df)

    # Calculate the boundaries for the windows to create from the df and the given window_size and step_length
    windows, error = create_list_of_windows(df, window_length, step_length, file)
    if error:
        return data_utils.create_result_string(file, [1], 0, error)
    # This Case catches files with a seen duration smaller than one window-length, meaning no window can be build, but it is not an error either
    if windows == []:
        return data_utils.create_result_string(file, [None], 0, "None")

    # To prevent loss of information, ffill the df, as not every column has the same sampling rate and may have no values at all within very small windows
    df_filled, error = data_utils.create_filled_df(df, filling_method)
    if error:
        return data_utils.create_result_string(file, [1], 0, error)

    # Resample the filled_df
    df_filled, error = time_utils.resample_data(df_filled, str(resampling_rate)+"ms")
    if error:
        return data_utils.create_result_string(file, [1], 0, error)

    # Handle all the windows for the current file: initialize objects regarding success, errors and successfully saved files
    windows_saved_current_file = 0
    window_successes = []
    window_errors = []
    # error_string captures occurred errors, which will be reported
    error_string = None

    for window in windows:
        success, windows_saved_current_file, error = handle_window(df, window, df_filled, data_columns, flatten, file, window_length, resampling_rate, store_local, saving_folder, windows_saved_current_file)
        # From the list containing the successes and the errors the result_string is calculated
        window_successes.append(success)
        window_errors.append(error)
        if isinstance(error, str):
            error_string = error

    logger.debug(f"Finished processing {windows_saved_current_file} windows for the file {file}")

    return data_utils.create_result_string(file, window_errors, windows_saved_current_file, error_string)


def valid_attributes(labeled_files: list, needed_columns: list, window_length: int, step_length: int, filling_method: str, resampling_rate: int, data_columns: list, flatten: bool, store_local: bool, saving_folder: str, normalize: str):
    """checks if all the given arguments are valid
    """
    if not isinstance(labeled_files, list):
        logger.error(f"Got labeled_files {labeled_files} which is not a list, investigation needed")
        return "given_labeled_files_not_a_list"

    if sum([isinstance(x, str) for x in labeled_files]) != len(labeled_files):
        logger.error(f"Got a file which is not a string, investigation needed")
        return "got_file_of_nonstr_type"

    if not labeled_files:
        logger.error(f"Got empty list of labeled_files {labeled_files}, investigation needed")
        return "labeled_files_empty"

    if not isinstance(needed_columns, list):
        logger.error(f"Got needed_columns {needed_columns} which is not a list, investigation needed")
        return "needed_columns_not_a_list"

    if not needed_columns:
        logger.error(f"Got empty list of needed columns {needed_columns}, investigation needed")
        return "needed_columns_empty"

    if not isinstance(window_length, int):
        logger.error(f"Given window length is not an int: {window_length, type(window_length)}, investigation needed")
        return "non_int_window_length"

    if window_length < 10:
        logger.error(f"Received too small window length: {window_length}, investigation needed")
        return "window_length_too_small"

    if step_length < 2:
        logger.error(f"Received too small step_length: {step_length}, investigation needed")
        return "step_length_too_small"

    if step_length > window_length:
        logger.error(f"Given window_length {window_length} is smaller than given step_length {step_length} Not allowed to avoid loss of data")
        return "step_length_bigger_window_length"

    valid_filling_methods = ["ffill", "linear"]
    if filling_method not in valid_filling_methods:
        logger.error(f"Given filling_method {filling_method} not yet implemented choose from  {valid_filling_methods}")
        return "filling_method_not_implemented"

    if not isinstance(resampling_rate, int):
        logger.error(f"Gor resampling_rate of wrong data-type, Expected int, got {type(resampling_rate)} ")
        return "got_resampling_rate_of_nonint_type"

    if resampling_rate < 2:
        logger.error(f"Given resampling_rate {resampling_rate} too small ")
        return "resampling_rate_too_small"

    if resampling_rate > 200:
        logger.error(f"Given resampling_rate {resampling_rate} too big ")
        return "resampling_rate_too_big"

    if not set(data_columns).issubset(needed_columns):
        logger.error("The data columns are not a true subset of the needed cols")
        return "data_col_not_subset_of_needed_cols"

    if not isinstance(flatten, bool):
        logger.error(f"Expected type bool for flatten, got {type(flatten)}")
        return "got_flatten_of_nonbool_type"

    if not isinstance(store_local, bool):
        logger.error(f"Expected type bool for store_local, got {type(store_local)}")
        return "got_store_local_of_nonbool_type"

    if not isinstance(saving_folder, str):
        logger.error(f"Expected type str for store_local, got {type(saving_folder)}")
        return "got_saving_folder_of_nonstr_type"

    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    valid_normalization_levels = [None, "max", "z_normalize"]
    if normalize not in valid_normalization_levels:
        logger.error(f"Received invalid normalization level {normalize}")
        return "invalid_normalization_level_given"

    return None


def create_list_of_file_dicts(labeled_files: list, needed_columns: list, window_length: int, step_length: int, filling_method: str, resampling_rate: int, data_columns: list, flatten: bool, store_local: bool, saving_folder: str, normalize: str, normalization_df: pd.DataFrame):
    """ Creates a list with dicts of the file and parameters, with which to handle this file
    """
    error = valid_attributes(labeled_files, needed_columns, window_length, step_length, filling_method,
                             resampling_rate, data_columns, flatten, store_local, saving_folder, normalize)
    if error:
        return [], error

    list_of_file_dicts = []
    for file in labeled_files:
        list_of_file_dicts.append(
            {"file": file,
             "needed_columns": needed_columns,
             "window_length": window_length,
             "step_length": step_length,
             "filling_method": filling_method,
             "resampling_rate": resampling_rate,
             "data_columns": data_columns,
             "flatten": flatten,
             "store_local": store_local,
             "saving_folder": saving_folder,
             "normalize": normalize,
             "normalization_df": normalization_df})

    return list_of_file_dicts, None


def valid_table(table: dict):
    """Tests if the given table arguments are valid
    """
    # Check if the folder with the data_path exists
    if not os.path.exists(table.get("data_path")):
        logger.error(f"Could not find given data_path {table.get('data_path')}")
        return "dp_does_not_exist"

    # test if valid parameters for the anonymization_file were given
    error = file_utils.valid_anonymize_file_found(table.get("data_path"))
    if error:
        return error

    # Check if there is a folder with the labeled data in the given data_path
    if not os.path.exists(os.path.join(table.get("data_path"), "labeled_data")):
        logger.error(f"Could not find required folder labeled_data in {table.get('data_path')}")
        return "dir_labeled_data_not_subfolder_of_dp"

    # Check if a valid value for store_local was given
    if not isinstance(table.get("dryrun"), bool):
        logger.error(f"Wrong datatype for dryrun given. Expected bool, got  {type(table.get('dryrun'))}")
        return "got_dryrun_of_nonbool_datatype"

    # Check if the windows lengths and step lengths are correct
    error = data_utils.valid_window_and_step_length(table.get("window_length"), table.get("step_length"))
    if error:
        return error

    # check if a valid resampling_rate was given
    if not isinstance(table.get("resampling_rate"), int):
        logger.error(f"Wrong datatype for resampling_rate given. Expected int, got  {type(table.get('resampling_rate'))}")
        return "got_resampling_rate_of_nonint_dataytpe"

    if table.get("resampling_rate") > 200:
        logger.error(f"Given resampling_rate {table.get('resampling_rate')} too big ")
        return "resampling_rate_too_big"

    if table.get('resampling_rate') < 2:
        logger.error(f"Given resampling_rate {table.get('resampling_rate')} too small ")
        return "resampling_rate_too_small"

    if table.get("resampling_rate") > table.get("window_length"):
        logger.error(f"Given resampling_rate {table.get('resampling_rate')} is bigger than specified window_length {table.get('window_length')}")
        return "resampling_rate_bigger_than_window_size"

    # test if a valid value for flatten was given
    if not isinstance(table.get('flatten'), bool):
        logger.error(f"Expected type bool for flatten, got {type(table.get('flatten'))}")
        return "got_flatten_of_nonbool_type"

    # If a time was given as the folder name to save the windows into, check if the date in in the correct format
    error = time_utils.valid_time_string(table.get("output_date"))
    if error:
        return error

    # Test if there are valid files in the folder
    _, error = file_utils.get_files(os.path.join(table.get("data_path"), "labeled_data"), "csv")
    if error:
        return error

    valid_normalization_levels = [None, "max", "z_normalize"]
    if table.get("normalize") not in valid_normalization_levels:
        logger.error(f"Received invalid normalization level {table.get('normalize')}. Valid levels are {valid_normalization_levels}")
        return "invalid_normalization_level_given"

    return None


def return_normalization_df(normalize: str, anonymize_file: dict):
    """returns a valid df for normalization based on the given normalize value
    """
    if not normalize:
        return pd.DataFrame(), None
    if normalize == "z_normalize":
        normalization_df = pd.DataFrame(columns=anonymize_file["create_windows"]["data_columns"])
        normalization_df.loc[0] = anonymize_file["create_windows"]["z_normalize_mean"]
        normalization_df.loc[1] = anonymize_file["create_windows"]["z_normalize_std"]
        return normalization_df, None
    if normalize == "max":
        normalization_df = pd.DataFrame(columns=anonymize_file["create_windows"]["data_columns"])
        normalization_df.loc[0] = anonymize_file["create_windows"]["max_normalize_max"]
        return normalization_df, None

    logger.error(f"Could not return valid normalization df with unknown normalization type {normalize}")
    return pd.DataFrame(), "initializing_normalization_df_failed"
