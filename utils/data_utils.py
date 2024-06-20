import copy
import bisect
import math
import pandas as pd
import numpy as np

from config.log import logger
from utils import file_utils


def has_identical_length(df: pd.DataFrame, colname: str):
    """Checks if all the values in given col have identical length
    """
    return len(str(df[colname].max())) == len(str(df[colname].min()))


def only_nan(df: pd.DataFrame, cols: list):
    """Checks if the df has only NaN in the given columns
    """
    df_to_test = df[cols].copy(True)
    if df_to_test.isnull().sum().sum() == len(df)*len(cols):
        return "contains_only_nan"
    return None


def valid_window_and_step_length(window_length: int, step_length: int, min_window_length=1, min_step_length=1):
    """ Checks if the given window_length and step_length are valid
    """
    # Check if a window length was given
    if window_length is None:
        logger.error("Cannot create features with no window_length given")
        return "window_length_is_None"
    # Check if a step length was given
    if step_length is None:
        logger.error(f"Cannot create features with no step_length given")
        return "step_length_is_None"
    # Check if the window length is valid
    if window_length < min_window_length:
        logger.error(f"Cannot create windows of a length shorter than the minimum window length {min_window_length}. Choose bigger windows size")
        return f"given_window_length_smaller_{min_window_length}"
    # Check if the step length is valid
    if step_length < min_step_length:
        logger.error(f"Cannot create windows by using a shorter step length than the minimum step length {min_step_length}. Choose bigger step size")
        return f"given_step_length_smaller_{min_step_length}"
    # To prevent loss of information, using a step length which is greater than the window size is not permitted
    if step_length > window_length:
        logger.error(f"Given step length {step_length} is greater than specified window_length. This is not allowed, to prevent loss of information. Reduce the step_length")
        return "step_length_greater_window_length"

    return None


def create_filled_df(df: pd.DataFrame, filling_method: str):
    """Creates a filled version of the df
    """
    valid_filling_methods = ['ffill', 'linear']
    if filling_method not in valid_filling_methods:
        logger.error(f"The given filling method {filling_method} is not implemented/valid. Choose from {valid_filling_methods}")
        return pd.DataFrame(), f"invalid_filling_method_{filling_method}_given"

    if not isinstance(df, pd.DataFrame):
        logger.error(f"Got invalid datatype for the df. Expected Pandas Df, got {type(df)}")
        return pd.DataFrame(), f"expected_pdDataFrame_got_{type(df)}"

    if "time" not in df.columns:
        logger.error("Df without a time column given, constructing of df_filled not possible")
        return pd.DataFrame(), "time_col_missing"

    if "label" not in df.columns:
        logger.error("Df without a label column given, constructing df_filled not possible")
        return pd.DataFrame(), "label_col_missing"

    if df["label"].nunique() > 1:
        logger.error("Got df with multiple label levels")
        return pd.DataFrame(), "Received_df_with_multiple_labels"

    # To make sure the original df is not modified, a true copy of the df is generated and manipulated
    df_to_be_filled = copy.deepcopy(df)

    # As no processing is needed for the label col, remove it
    label = int(df_to_be_filled["label"][0])
    df_to_be_filled = df_to_be_filled.drop(columns=["label"], axis=1)

    # Make sure the df is sorted by the time col
    df_to_be_filled = df_to_be_filled.sort_values("time", ascending=True).reset_index(drop=True)

    # Get the range of timepoints with which to fill
    earliest_timepoint = int(df_to_be_filled["time"][0])
    last_timepoint = int(df_to_be_filled["time"][len(df_to_be_filled)-1])

    # Handle the filling

    # For the real observations no filler is needed, therefore the needed timepoints are gathered by the diff between all possible and the existing
    step_length_in_ms = 1
    all_potential_timepoints = {x for x in range(earliest_timepoint, last_timepoint, step_length_in_ms)}
    all_seen_timepoints = {x for x in df_to_be_filled["time"]}

    all_timepoints_to_add = list(all_potential_timepoints.difference(all_seen_timepoints))

    # Create a df with the new timepoints and nan in the other cols
    df_to_concat = pd.DataFrame({"time": all_timepoints_to_add})
    for col in df_to_be_filled.columns:
        if col == "time":
            continue
        df_to_concat[col] = np.nan

    df_filled = pd.concat([df_to_be_filled, df_to_concat])

    # Sort the df, as the filler rows got appended at the end, and reset the index
    df_filled = df_filled.sort_values("time", ascending=True).reset_index(drop=True).astype({"time": np.int64})

    # within the file, filling is needed, as the filler rows were initiated empty
    if filling_method == "ffill":
        df_filled = df_filled.ffill().bfill()
    elif filling_method == "linear":
        df_filled = df_filled.interpolate("linear", 0)
    else:
        logger.error(f"Received filling method {filling_method}, which is implemented yet")
        return pd.DataFrame(), f"invalid_filling_method_{filling_method}_given"

    # Append the label
    df_filled["label"] = label

    return df_filled, None


def flatten(df_to_flatten: pd.DataFrame, file: str, window_length=1000, resampling_length_ms=10, remove_time_entries=True):
    """returns a flattened version of the df, meaning all the rows get appended to one row shape(5,5) -> shape(1,25)
    """
    # Check if the label and time col are in the df
    if not "time" in df_to_flatten.columns or "label" not in df_to_flatten.columns:
        logger.error(f"The given df does not contain a time a label column. Investigation needed for file {file}")
        return pd.DataFrame(), "time_or_label_not_in_df_to_flatten"

    # Test if the amount of seen columns matches the expected amount
    if len(df_to_flatten.columns) != 11:
        logger.error(f"Received wrong amount of columns. Expected 11, got {len(df_to_flatten.columns)} at file {file} ")
        return pd.DataFrame(), f"wrong_col_amount_expected_11_got_{len(df_to_flatten.columns)}"

    # Depending on the input, the df may contain one additional row, which might be discarded
    if len(df_to_flatten) == (int(window_length/resampling_length_ms) + 1):
        df_to_flatten = df_to_flatten.iloc[1:, :]

    # If the len of the df_to_flatten still does not match the expected length, investigation is needed
    if len(df_to_flatten) != (window_length/resampling_length_ms):
        logger.error(f"Received df with the wrong len. Expected a len of {(window_length/resampling_length_ms)}, got len {len(df_to_flatten)}. file {file}")
        return pd.DataFrame(), f"wrong_len_of_df_expected_{(window_length/resampling_length_ms)}_got_{len(df_to_flatten)}"

    # Remove the label, as it needs to be in the resulting df only once
    label = int(df_to_flatten.iloc[0, df_to_flatten.columns.get_loc("label")])
    df_to_flatten = df_to_flatten.drop(columns=["label"])

    # When the time columns should not be flattened, it is removed from the df_to_flatten beforehand
    if remove_time_entries:
        df_to_flatten = df_to_flatten.drop(columns=["time"])

    # Flatten the df to a one-dim df
    df_flattened = pd.DataFrame(df_to_flatten.values.flatten()).T

    # creating the colnames is necessary for ensuring correct appending of dfs later on
    # Create the colnames, of format <feature_name>_<timepoint_in_df >
    flattened_colnames = []
    for iterations in range(len(df_to_flatten)):
        for colname in df_to_flatten.columns:
            if colname == "label":
                continue
            if remove_time_entries and colname == "time":
                continue
            flattened_colnames.append(f"{colname}_{str(iterations)}")

    df_flattened.columns = flattened_colnames

    df_flattened["label"] = label

    return df_flattened, None


def split_files_to_handle_into_chunks(files_to_handle_list: list, chunksize=24):
    """Splits a List of files to handle into a list of chunks
    """
    return [files_to_handle_list[i:i + chunksize] for i in range(0, len(files_to_handle_list), chunksize)]


def find_first_index_bigger_x_in_sorted_list(sorted_list: list, x: int):
    """returns the position of the first value in the sorted (ascending) list which is bigger than the given value
    """
    return bisect.bisect(sorted_list, x)


def create_result_string(file: str, window_errors: list, windows_saved_current_file: int, error_string: str, separator="+++"):
    """Creates a string which captures, if the processing of this file was successful, how many windows were saved (and an error message, if one occurred)
    """
    if window_errors.count(None) == len(window_errors):
        return (file + f"{separator}True{separator}None{separator}{str(windows_saved_current_file)}").replace("\\","/")

    return (file + f"{separator}False{separator}{error_string}{separator}{str(windows_saved_current_file)}").replace("\\","/")


def downsize_label(label_series: pd.Series, label_depth: int):
    """checks if the label column is valid and brings it to the specified label_depth
    """
    valid_label_depths = [1, 2, 3]
    if label_depth not in valid_label_depths:
        logger.error(f"Received invalid label_depth {label_depth}. Choose from {valid_label_depths}")
        return None, "invalid_label_depth_given"

    # Test if there are labels in series with a precision smaller than the needed
    if len(str(label_series.min())) < label_depth:
        logger.error(f"Received label column with a granularity which is smaller than the needed label_depth {label_depth}. Investigation needed")
        return None, "granularity_of_label_col_smaller_than_label_depth"

    # return a cutted version of the label column with label depth 2
    cutted_label = np.where(label_series.to_numpy() < 100, label_series.to_numpy(), label_series.to_numpy()/10).astype(int)
    if label_depth == 1:
        cutted_label = np.divide(cutted_label, 10).astype(int)
    return cutted_label, None


def prepare_label(label_series: pd.Series, label_depth: int, label_mapping=None):
    """ Prepares the label by cutting away unwanted precision and mapping old_label_nr to range(1, nr_labels +1)
    """
    valid_label_depths = [1, 2, 3]
    if label_depth not in valid_label_depths:
        logger.error(f"The given label_depth {label_depth} is not in the valid_label_depths {valid_label_depths}")
        return [], "invalid_label_depth_given"

    downsized_label, error = downsize_label(label_series, label_depth)
    if error:
        return [], error

    # Additionally the labels need to be remade: made to ints starting from 1
    # Find the amount of labels
    nr_all_labels = np.unique(downsized_label)

    # Make a dict which maps label nrs to new_nr: eg. {"11":1,"12":2,"15":"3","21":4}
    if label_mapping is None:
        label_mapping = create_label_remapping(nr_all_labels)

    # create a list of the new labels from the label_mapping
    for key in label_mapping.keys():
        downsized_label = np.where(
            downsized_label == key, label_mapping[key], downsized_label)

    return downsized_label, None


def label_class_mean(df: pd.DataFrame, col: str, label_col: str):
    """Returns a dict mapping from the encountered labels to the classwise means for this col
    """
    if df[col].isna().all():
        logger.error(f"Cannot calculate mean with only NA values")
        return {}, "label_class_mean_cannot_be_calculated_for_row_with_only_NAN"
    if label_col not in df.columns:
        logger.error(f"Cannot calculate labelwise mean without label col {label_col}")
        return {}, "label_col_not_found"

    # Quite much of the quality of the data would be lost by simply imputing the overall mean of the feature
    # a more sophisticated way is to impute the mean of the feature of the seen label class,
    # as this preserves differences in classes
    label_classes = df[label_col].unique()
    dict_label_to_mean = {}
    for label in label_classes:
        # Calculate the mean for this label
        mean = df.loc[df[label_col] == label][col].mean(skipna=True)
        dict_label_to_mean[label] = mean

    return dict_label_to_mean, None


def naive_imputation(df: pd.DataFrame, log_file: dict):
    """Implements a naive imputation to prevent dropping rows from df and messing up label distribution
    """
    if not df.isna().any().any():
        return df, None

    list_of_dfs, error = create_list_of_dfs_for_imputation(df, log_file)
    if error:
        return pd.DataFrame(), error


    label_idx = df.columns.get_loc("label")
    for df in list_of_dfs:
        for col in df.columns:
            # Rise an error when there are NaN in the label col
            if col in ["label", "label_top", "label_mid"] and df[col].isna().any():
                logger.error(f"found nan in label col {col}")
                return pd.DataFrame(), "found_nan_values_in_label_col"
            if col in ["label", "label_top", "label_mid"]:
                continue

            # Replace nan for this col, if there are nan in this col
            if df[col].isna().any():
                label_col, error = detect_label_col(df, log_file)
                if error:
                    return pd.DataFrame(), error

                # calculate a class-wise mean for this row
                col_means, error = label_class_mean(df, col, label_col)
                if error:
                    return pd.DataFrame(), error

                col_without_nas = [col_means[df.iloc[number, label_idx]] if math.isnan(x) else x for number, x in enumerate(df[col])]
                df[col] = col_without_nas

        if df.isna().any().any():
            return pd.DataFrame(), "nan_values_found_after_imputation"

    joined_df, error = join_list_of_sub_dfs(list_of_dfs)
    if error:
        return pd.DataFrame(), error

    return joined_df, None


def naively_impute_if_needed(all_data_df: pd.DataFrame, log_file: dict):
    """Checks if there are NaN in data and imputes naively, if the amount to impute is small
    """

    unexpected_na_values_found, error = unexpected_na_values_seen(all_data_df, log_file)
    if error:
        return pd.DataFrame(), error

    if unexpected_na_values_found:
        logger.info("Detected nan values, beginning with naive imputation")
        all_data_df, error = naive_imputation(all_data_df, log_file)
        if error:
            return pd.DataFrame(), error

        logger.info(f"Finished Imputation of NA values")

    return all_data_df, None


def create_label_remapping(labels: list):
    """returns a dict mapping old label_nr to new (e.g. 11 -> 1, 12 -> 2, 13 -> 3)
    """
    unique_sorted_labels = sorted(list(set(labels)))
    label_remapping = {}
    counter = 1
    for label in unique_sorted_labels:
        label_remapping[label] = counter
        counter += 1
    return label_remapping


def max_scale_df(df: pd.DataFrame, normalization_df: pd.DataFrame):
    """preforms max_scaling on the df
    """
    return pd.DataFrame(np.divide(df.to_numpy(), normalization_df.to_numpy()))


def unexpected_na_values_seen(df: pd.DataFrame, log_file: dict):
    """ checks the given df for unexpected NA values (only these need imputation)
    """
    if "generate_features" not in log_file.keys():
        return df.isna().any().any(), None

    if "reduce_data" not in log_file.keys():
        logger.error(f"processing featured data without reducing the data is not allowed")
        return True, "received_featured_but_unreduced_data"

    if log_file["reduce_data"]["label_depth"] == 1:
        return df.isna().any().any(), None

    result, error = unexpected_na_in_hierarchical_features(df, log_file)

    return result, error


def unexpected_na_in_hierarchical_features(df: pd.DataFrame, log_file: dict):
    """Checks the df for hierarchical featured data if unexpected NaN values were found 
    """
    list_of_dfs, error = create_list_of_dfs_for_imputation(df, log_file)
    if error:
        return True, error

    for individual_df in list_of_dfs:
        if individual_df.isna().any().any():
            # Naive Imputation is not possible, when the amount to impute is too big
            nr_of_nan_values = individual_df.isna().sum().sum()
            proportion_of_nan_values = nr_of_nan_values / (len(individual_df) * len(individual_df.columns))

            if proportion_of_nan_values > 0.1:
                logger.error(f"The amount of data to impute is too big: {proportion_of_nan_values*100} % of all available data")
                return True, "too_much_nan_values_or_naive_imputation"

            return True, None

    return False, None


def create_top_and_mid_df(df: pd.DataFrame, log_file: dict):
    """returns a df containing only rows meant for top-level classification and a df containing rows for mid-level class.
    """
    df["index_col"] = df.index

    if "generate_features" not in log_file.keys():
        logger.error("creating sub_dfs is only valid for featured data")
        return pd.DataFrame(), pd.DataFrame(), "received_unfeatured_data"

    if "reduce_data" not in log_file.keys():
        logger.error("processing featured data without reducing the data is not allowed")
        return pd.DataFrame(), pd.DataFrame(), "received_featured_but_unreduced_data"

    if "needed_columns" not in log_file["reduce_data"].keys():
        logger.error("log_file[reduce_data] does not contain the needed cols")
        return pd.DataFrame(), pd.DataFrame(), "needed_cols_missing_from_log_file"

    internal_copy_df = df.copy(True)

    # The resting data gets separated, as this data is meant for the top-level
    df_resting_data = internal_copy_df[internal_copy_df["label"] == 31]

    internal_copy_df = internal_copy_df[df["label"] != 31]

    top_level_df = create_top_level_df(internal_copy_df, log_file, df_resting_data)

    mid_level_df = create_mid_level_df(df, internal_copy_df, log_file, top_level_df, df_resting_data)

    return top_level_df, mid_level_df, None


def create_list_of_dfs_for_imputation(df: pd.DataFrame, log_file: dict):
    """ As the hierarchical data will be imputed after separating, this function provides the list of df(s) to impute
    """
    if "generate_features" not in log_file.keys():
        return [df], None

    if "reduce_data" not in log_file.keys():
        logger.error("Imputation is not allowed for unreduced data")
        return [], "received_unreduced_data"

    top_level_df, mid_level_df, error = create_top_and_mid_df(df, log_file)
    if error:
        return [], error

    list_of_dfs, error = split_joined_mid_level_df(mid_level_df, log_file)
    if error:
        return [], error

    list_of_dfs.append(top_level_df)

    return list_of_dfs, None


def join_list_of_sub_dfs(list_of_subdfs: list):
    """returns a single df containing the joined df
    """
    if len(list_of_subdfs) == 1:
        return list_of_subdfs[0], None

    for df in list_of_subdfs:
        if not isinstance(df, pd.DataFrame):
            logger.error("detected element of non pd.Dataframe type")
            return pd.DataFrame(), "got_non_pdDataframe_dt"

    return pd.concat(list_of_subdfs).sort_index(), None


def prepare_label_for_featured_data(df: pd.DataFrame, label_depth: int):
    """prepares the df for hierarchical classification
    """
    valid_label_depths = [1, 2]
    if label_depth not in valid_label_depths:
        logger.error("Received invalid label_depths")
        return pd.DataFrame(), "invalid_label_depth_given"

    if label_depth == 1:
        df["label"], error = prepare_label(df["label"], label_depth)
        if error:
            return pd.DataFrame(), error

        return df, None

    df, error = prepare_labels_hierarchical_classification(df, label_depth)
    if error:
        return pd.DataFrame(), error

    if "top_level_marker" in df.columns:
        df = df.drop(columns=["top_level_marker"])

    return df, None


def prepare_labels_hierarchical_classification(df: pd.DataFrame, label_depth: int):
    """ initializes additional cols containing the label col in the needed transformations
    """
    df["index_col"] = df.index

    if label_depth < 2:
        logger.error("Cannot prepare for hierarchical classification, when top-level classification is requested")
        return pd.DataFrame(), "received_too_small_label_depth"

    if len(str(df["label"].min())) != len(str(df["label"].max())):
        logger.error("Cannot prepare_label for hierarchical class for unequal label precision")
        return pd.DataFrame(), "unequal_label_precision_found"

    if len(str(df["label"].min())) != 2:
        logger.error("Cannot prepare_label for hierarchical class. for precision != 2")
        return pd.DataFrame(), "wrong_label_precision_found"

    prepared_label, error = prepare_label(df["label"], 1)
    if error:
        return pd.DataFrame(), error
    df["label_top"] = prepared_label

    more_general_activities = df["label_top"].unique()
    list_of_sub_dfs = []

    for more_general_activity in more_general_activities:
        sub_df = df[df["label_top"] == more_general_activity].copy()
        label_remapping = create_label_remapping(
            sorted([x for x in sub_df["label"].unique()]))

        mid_labels = sub_df["label"].copy()

        for key in label_remapping.keys():
            mid_labels = np.where(
                mid_labels == key, label_remapping[key], mid_labels)

        sub_df["label_mid"] = mid_labels
        list_of_sub_dfs.append(sub_df)

    joined_df, error = join_list_of_sub_dfs(list_of_sub_dfs)

    joined_df = joined_df.sort_values("index_col").reset_index(drop=True).drop(columns=["index_col"])

    return joined_df, error


def rowwise_proportion_of_na_smaller_than_boundary(df: pd.DataFrame, proportion: float) -> np.ndarray:
    """returns a np.ndarray containing as bool if the proportion of NaN values in row is smaller than given boundary
    """
    return np.sum(np.isnan(df.to_numpy()), 1) < proportion*len(df.columns)


def split_joined_mid_level_df(df: pd.DataFrame, log_file: dict):
    """As the mid_level df may contain the data for various models with various input features, splitting into a list of models is needed
    """
    if "label_top" not in df.columns:
        logger.error(f"cannot split by label_top, when label_top not given")
        return [], "label_top_col_not_found"

    list_of_mid_level_dfs = []
    for top_level_label in sorted(df["label_top"].unique()):
        needed_columns = log_file["reduce_data"]["needed_columns"][top_level_label].copy()
        needed_columns.extend(["label_top", "label_mid", "index_col"])
        mid_level_df = df[df["label_top"] == top_level_label][needed_columns].reset_index(drop=True)
        list_of_mid_level_dfs.append(mid_level_df)

    return list_of_mid_level_dfs, None


def detect_label_col(df: pd.DataFrame, log_file: dict):
    """detects the name of the correct label to use for imputation
    """
    if "label_top" not in df.columns:
        return "label", None
    feature_names_top_level = log_file["reduce_data"]["needed_columns"][0]
    if set(feature_names_top_level).issubset(set(df.columns)):
        return "label_top", None

    feature_names_lifting = log_file["reduce_data"]["needed_columns"][1]
    if set(feature_names_lifting).issubset(set(df.columns)):
        return "label_mid", None

    feature_names_walking = log_file["reduce_data"]["needed_columns"][2]
    if set(feature_names_walking).issubset(set(df.columns)):
        return "label_mid", None

    logger.error(f"unable to detect correct label_col_to use")
    return "", "detection_of_label_col_failed"


def create_top_level_df(df: pd.DataFrame, log_file: dict, df_top_level_without_subcategories: pd.DataFrame):
    """returns a df containing only the rows which contain information for the top-level classification
    """
    top_level_features: list = log_file["reduce_data"]["needed_columns"][0].copy()

    top_level_features.extend(["label_top", "label_mid", "index_col"])
    if "window_uuid" in top_level_features:
        top_level_features.remove("window_uuid")


    top_level_df = df[top_level_features].copy(True)
    cols_to_use = [col for col in top_level_df.columns if col != "window_uuid"]
    top_level_df["top_level_marker"] = rowwise_proportion_of_na_smaller_than_boundary(top_level_df[cols_to_use], 0.5)
    top_level_df['window_uuid'] = df["window_uuid"][top_level_df["top_level_marker"].to_list()]
    top_level_df = top_level_df[top_level_df["top_level_marker"]]

    top_level_features.extend(["top_level_marker", "window_uuid"])
    df_top_level_without_subcategories["top_level_marker"] = True
    return pd.concat([top_level_df, df_top_level_without_subcategories[top_level_features]]).reset_index(drop=True)


def create_mid_level_df(df: pd.DataFrame, reduced_copy_df: pd.DataFrame, log_file: dict, top_level_df: pd.DataFrame, df_top_level_without_subcategories: pd.DataFrame):
    """returns a df containing only the rows which contain information for the top-level classification
    """
    # create a list containing all the needed columns
    needed_mid_level_cols = log_file["reduce_data"]["needed_columns"][1].copy()
    needed_mid_level_cols.extend(log_file["reduce_data"]["needed_columns"][2])
    needed_mid_level_cols.extend(["label_top", "label_mid", "index_col"])

    # create a joined version of the df (needed, as a full df is needed for the filtering via indexes later on)
    reduced_copy_df = pd.concat([reduced_copy_df, df_top_level_without_subcategories]).sort_values("index_col").reset_index(drop=True)
    mid_level_df = reduced_copy_df[needed_mid_level_cols].reset_index(drop=True)

    # Calculate the indices for the mid-level data, index by it and set the top-level marker to False
    mid_level_row_indices = sorted(list(set(df["index_col"]).difference(set(top_level_df["index_col"]))))

    mid_level_df = mid_level_df.iloc[mid_level_row_indices, :].reset_index(
        drop=True)
    mid_level_df["top_level_marker"] = False

    return mid_level_df


def z_normalize_df(df: pd.DataFrame, normalization_df: pd.DataFrame):
    """uses the values from the normalization_df to z-transform the df
    """
    for col in df.columns:
        zero_centered_col = df[col].to_numpy() - normalization_df[col][0]
        normalized_col = np.divide(zero_centered_col, normalization_df[col][1])
        df[col] = normalized_col

    return df


def determine_standardization_type(standardizing_df: pd.DataFrame):
    """ reads the df with the standardization to return seen standardization type
    """
    if "Unnamed: 0" in standardizing_df.columns:
        if set(["mean", "scale", "var"]).issubset(set(standardizing_df["Unnamed: 0"])):
            return "normal", None
        if set(["max_abs", "scale"]).issubset(set(standardizing_df["Unnamed: 0"])):
            return "max_abs", None

    if set(["mean", "scale", "var"]).issubset(set(standardizing_df.index)):
        return "normal", None
    if set(["max_abs", "scale"]).issubset(set(standardizing_df.index)):
        return "max_abs", None

    logger.error("Could not determine standardization_type of given standardization file")
    return None, "could_not_determine_standardization_type"


def prepare_standardization_df(model_folder: str, standarization_needed: bool):
    """reads the csv with the standardization params from the model_folder and infers the type of the seen standardization
    """
    if not standarization_needed:
        return pd.DataFrame(), None, None

    standardizing_files, error = file_utils.get_files(model_folder, "_std.csv")
    if error:
        return pd.DataFrame(), None, error

    standardizing_df = pd.read_csv(standardizing_files[0])

    standardization_type, error = determine_standardization_type(
        standardizing_df)
    if error:
        return pd.DataFrame(), None, error

    if "Unnamed: 0" in standardizing_df.columns:
        standardizing_df = standardizing_df.drop("Unnamed: 0", axis=1)

    return standardizing_df, standardization_type, None


def standardize(df_to_standardize: pd.DataFrame, standardization_type: str, standardizing_df: pd.DataFrame):
    """applies the given standardization with the parameters from the standardizing_df to the df_to_standardize
    """
    if standardization_type == "normal":
        return pd.DataFrame(np.divide(df_to_standardize.to_numpy() - standardizing_df.iloc[0, :].to_numpy(), standardizing_df.iloc[1, :].to_numpy()))
    if standardization_type == "max_abs":
        return pd.DataFrame(np.divide(df_to_standardize.to_numpy(), standardizing_df.iloc[0, :].to_numpy()))


def bring_standardizing_df_in_correct_order(standardizing_df: pd.DataFrame, hierarchical_model: str):
    """ brings the columns of the standardizing df into the correct order
    """
    if standardizing_df.empty:
        return standardizing_df, None

    if hierarchical_model == "top":
        return standardizing_df, None

    if hierarchical_model == "lifting":
        standardizing_df.columns = [x for x in range(len(standardizing_df.columns))]
        rearrannged_df = pd.DataFrame(standardizing_df[[0, 4, 1, 2, 3, 5, 6]].to_numpy())
        return rearrannged_df, None

    if hierarchical_model == "walking":
        standardizing_df.columns = [x for x in range(len(standardizing_df.columns))]
        rearrannged_df = pd.DataFrame(standardizing_df[[0, 2, 1, 3, 4, 5, 6, 7]].to_numpy())
        return rearrannged_df, None


def prepare_standardization(model_folder: str, standarization_needed: bool, hierarchical_model: str):
    """reads and orders the df with the information for standardization based on the files in the model folder 
    """
    standardizing_df, standardization_type, error = prepare_standardization_df(model_folder, standarization_needed)
    if error:
        return None, None, error

    # Ensure that the order of the cols is correct before applying the standardization
    standardizing_df, error = bring_standardizing_df_in_correct_order(standardizing_df, hierarchical_model)
    if error:
        return None, None, error

    return standardizing_df, standardization_type, None
