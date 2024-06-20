import os
import sys
import json
import pandas as pd
import numpy as np
from config.log import logger

from utils import time_utils, file_utils, data_utils, feature_utils


def valid_table(table: dict):
    """Tests if the given table arguments are valid
    """
    # Check if the folder with the data_path exists
    if not os.path.exists(table.get("data_path")):
        logger.error(f"Could not find folder datapath {table.get('data_path')}")
        return "dp_does_not_exist"

    # test if valid parameters for the anonymization_file were given
    error = file_utils.valid_anonymize_file_found(table.get("data_path"))
    if error:
        return error

    # Check if a valid value for store_local was given
    if not isinstance(table.get("dryrun"), bool):
        logger.error(f"Wrong datatype for dryrun given. Expected bool, got  {type(table.get('dryrun'))}")
        return "got_dryrun_of_nonbool_datatype"

    # Check the input date
    if table.get("input_date") is None:
        return "got_input_date_of_Nonetype"
    error = time_utils.valid_time_string(table.get("input_date"))
    if error:
        logger.error(f"Invalid input_date given: {table.get('input_date')} . Please use format YYYY-MM-DDThh:mm. error: {error}")
        return "got_input_date_of_invalid_format"

    # Test if there are valid files in the input folder
    windowed_files, error = file_utils.get_files(os.path.join(table.get("data_path"), "windows", table.get('input_date')), "csv")
    if error:
        logger.error(f"The folder {os.path.join(table.get('data_path'), 'windows', table.get('input_date'))} does not contain any data in csv form")
        return error

    # Test if there is a log file in the input folder and if the params of the file match the ones needed for generating features
    log_filepath = os.path.join(table.get("data_path"), "windows", table.get('input_date'), "log.json")
    if os.path.exists(log_filepath):
        with open(log_filepath) as f:
            log_file = json.load(f)
        # Test if input data was flattened
        if log_file.get("create_windows").get("flatten"):
            logger.error(f"The input data was flattened, which is not supported for the generating of features")
            return "generating_features_for_flattened_data_not_allowed"
        if log_file.get("create_windows").get("method") != "ffill":
            logger.error(f"Generating of features is only valid with filling method ffill, but {log_file.get('method')} found in the log file of the input data")
            return "generating_features_for_interpolated_data_not_allowed"

    # If a output_date was given as the folder name to save the windows into, check if the date in in the correct format
    error = time_utils.valid_time_string(table.get("output_date"))
    if error:
        return error

    return None


def generate_features_for_file(file_with_attributes: dict):
    """generates the features for one window / one session depending on given arguments
    """
    # Create local variables from the given dict
    file = file_with_attributes.get("file")
    store_local = file_with_attributes.get("store_local")
    feature_functions_dict = file_with_attributes.get("feature_functions_dict")
    anonymize_file = file_with_attributes.get("anonymize_file")

    # Initialize a return string, containing the information that generating features succeeded and nothing was saved (overwritten when errors or saving occur)
    return_string = data_utils.create_result_string(file, [], 0, "None")

    # Start a counter containing  the information how many files have been processed
    processed_windows = 0

    # read the file and resample (and ffill if NA in df)
    df_resampled, error_result_string, label, window_uuids = read_prepare_and_ffill_file(file, processed_windows)
    if error_result_string:
        return error_result_string

    # Create all the columns, eg. diffs, gradients, e.g.
    df_extended, error = feature_utils.expand_df(df_resampled, anonymize_file)
    if error:
        return data_utils.create_result_string(file, [1], 0, error)

    # Remove NA
    df_extended = df_extended.dropna(how='any')

    # Create a df containing the boundaries of the seen df
    df_boundaries, error = feature_utils.create_df_boundaries(df_extended, np.int64(label), file)
    if error:
        logger.error(f"Generation of features not possible without a valid df_boundaries. Error occurred at file {file}")
        return data_utils.create_result_string(file, [1], processed_windows, error)

    # Generate features for each window
    df_features, error = feature_utils.generate_features(feature_functions_dict, df_extended, df_boundaries)
    if error or df_features.empty:
        logger.error(f"feature data frame empty for file {file}")
        return data_utils.create_result_string(file, [1], processed_windows, "df_empty_after_generating_features")


    # If saving is chosen save the individual window
    if store_local:
        return_string, error, processed_windows = save_generated_features(df_features, file, file_with_attributes, processed_windows)
        # When an error occurred, further processing is stopped to enable direct investigation
        if error:
            return return_string

    # After all has finished, the result may be published via the return_string
    return return_string


def save_generated_features(df: pd.DataFrame, file: str, table: dict, processed_windows:int):
    """saves the generated dfs using the arguments from the table, returns an result_string, an error and the amount of processed_windows
    """
    # Test if all the needed entries exist in the table dict
    if not set(["store_local", "data_path", "saving_time"]).issubset(set([x for x in table.keys()])):
        error_message = "not_all_needed_keys_in_table_for_saving"
        return data_utils.create_result_string(file, [1], processed_windows, error_message), error_message, processed_windows

    # When bundeling is given, the filename of the session is used
    if "win" in sys.platform:
        file = file.replace("\\","/")
    file_name = file.split("/")[-1]

    # Save the window / bundle of windows
    saving_path = os.path.join(table.get("data_path"), "features", table.get("saving_time"), file_name)
    if not os.path.exists(os.path.join(table.get("data_path"), "features", table.get("saving_time"))):
        os.makedirs(os.path.join(table.get("data_path"), "features", table.get("saving_time")))
    df.to_csv(saving_path, index=False)

    # Return an error if the file does not exist after saving
    if not os.path.exists(saving_path):
        error_message = "saving_file_failed"
        return data_utils.create_result_string(file, [1], processed_windows, error_message), error_message, processed_windows

    # return that the df was processed successfully:
    processed_windows += 1
    error_message = "None"

    return data_utils.create_result_string(file, [None], processed_windows, error_message), error_message, processed_windows


def read_prepare_and_ffill_file(file: str, processed_windows: int):
    """ Returns the resampled and ffilled df, the error-result_string, the label and the seen window-uuids
    """
    if not os.path.exists(file):
        error_message = "file_does_not_exist"
        return pd.DataFrame(), data_utils.create_result_string(file, [1], processed_windows, error_message), 0, []

    df_window = pd.read_csv(file)

    if "label" not in df_window.columns:
        error_message = "no_label_col_found"
        return pd.DataFrame(), data_utils.create_result_string(file, [1], processed_windows, error_message), 0, []

    if df_window.shape[0] in [0, 1]:
        error_message = "received_df_of_too_small_size"
        return pd.DataFrame(), data_utils.create_result_string(file, [1], processed_windows, error_message), 0, []

    # Extract the label and window_uuid from the seen df
    label = df_window['label'].dropna()[0]
    df_window = df_window.drop(columns=['label'])
    window_uuids = []
    if "window_uuid" in df_window.columns:
        window_uuids = df_window ['window_uuid'].to_list()

    df_resampled, error = time_utils.resample_data(df_window)
    if error:
        return pd.DataFrame(), data_utils.create_result_string(file, [1], 0, error), 0, []

    # As generate_features is used only for a comparison between window- and feature-based NN and SVMs,
    # expanding the potential ways of filling this is not needed

    # Get rid of nan values through forward-filling, if needed
    if df_resampled.isna().any().any():
        df_resampled = df_resampled.ffill().dropna(how='any')

    return df_resampled, None, label, window_uuids


def create_list_of_file_dicts(all_window_files:list, table: dict, feature_functions: dict, anonymize_file: dict):
    """ Creates a list with dicts of the file_name for one file for each session and parameters for processing 
    """
    error = valid_table(table)
    if error:
        return [], error

    list_of_file_dicts = [{"file": file,
                           "data_path": table.get("data_path"),
                           "input_date": table.get("input_date"),
                           "saving_time": table.get("saving_time"),
                           "store_local": not table.get("dryrun"),
                           "feature_functions_dict": feature_functions,
                           "anonymize_file": anonymize_file} for file in all_window_files]

    return list_of_file_dicts, None
