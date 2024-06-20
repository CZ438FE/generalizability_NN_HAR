import math
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.fft import fft

from utils import file_utils
from config.log import logger
module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def expand_df(df: pd.DataFrame, anonymize_file: dict):
    """ extends the given df with differences and gradients
    """
    error = is_valid(df, anonymize_file)
    if error:
        return pd.DataFrame(), error

    # Append the differences to the df
    df_with_diffs, error = extend_with_differences(df, anonymize_file)
    if error:
        logger.error("Computing acceleration and gradients not possible for malformed df")
        return pd.DataFrame(), error

    # Append the overall acceleration
    df_with_diff_and_acc, error = extend_with_acceleration(df_with_diffs, anonymize_file)
    if error:
        return pd.DataFrame(), error

    df_fully_extended = extend_with_gradients(df_with_diff_and_acc)

    return df_fully_extended, error


def extend_with_differences(df: pd.DataFrame, anonymize_file: dict):
    """ returns a df which contains additional columns containing the differences between columns
    """
    differences_cols = anonymize_file.get("feature_utils").get("differences_cols")

    if not set(differences_cols[0:2]).issubset(set(df.columns)):
        logger.error("Given df does not contain columns for the calculation of differences")
        return pd.DataFrame(), "differences_cols_not_found_calculation_of_diff_failed"

    df[differences_cols[2]] = abs(df[differences_cols[0]] - df[differences_cols[1]])
    return df, None


def extend_with_acceleration(df: pd.DataFrame, anonymize_file: dict):
    """ extends given df with a column containing the overall acceleration
    """
    accel_cols = anonymize_file.get("feature_utils").get("accel_cols")
    if not set(accel_cols[0:3]).issubset(set(df.columns)):
        logger.error("Calculation of the overall acceleration is not possible as the needed accel columns are not in the df")
        return pd.DataFrame(), "needed_accel_cols_not_found"

    acc_x = df[accel_cols[0]].to_numpy()
    acc_y = df[accel_cols[1]].to_numpy()
    acc_z = df[accel_cols[2]].to_numpy()

    acc_all = []
    for i, _ in enumerate(acc_x):
        if np.isnan(acc_x[i]) or np.isnan(acc_y[i]) or np.isnan(acc_z[i]):
            acc_all.append(None)
        else:
            acc_all.append(
                math.sqrt(acc_x[i] ** 2 + acc_y[i] ** 2 + acc_z[i] ** 2))

    df[accel_cols[3]] = acc_all

    return df, None


def extend_with_gradients(df: pd.DataFrame):
    """ returns a df which contains the gradients of the given columns
    """
    for column in df.columns:
        if column == 'time':
            continue
        df[f'{column}_grad'] = df[column].diff()
        # set first grad value to 0
        if df[f'{column}_grad'].empty:
            continue
        if math.isnan(df[f'{column}_grad'].iloc[0]):
            df.at[0, f'{column}_grad'] = 0
    return df


def is_valid(df: pd.DataFrame, anonymize_file: dict):
    """ checks if given objects are in the correct format for the generating of features
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Given object is not a dataframe")
        return "got_df_without_dtype_pdDataframe"

    needed_cols = anonymize_file.get("feature_utils").get("needed_cols")

    if not set(needed_cols).issubset(set(df.columns)):
        diff = set(needed_cols).difference(set(df.columns))
        logger.error(f"Some needed columns are missing in the given df: {diff}")
        return "needed_cols_missing"""

    return None


def create_df_boundaries(df: pd.DataFrame, label: np.int64, file: str):
    """ Create a df with label, start- and endtime, and window_uuid as entry
    """
    if not isinstance(label, np.int64):
        logger.error(f"No valid label given")
        return pd.DataFrame(), "given_label_not_int64"
    if "time" not in df.columns:
        logger.error("Could not find a time col in given df")
        return pd.DataFrame(), "time_col_missing"
    if df.shape[0] == 0:
        logger.error(f"Cannot create features for a df without rows")
        return pd.DataFrame(), "df_without_rows_given"

    labels = pd.DataFrame(columns=['time_start', 'time_end', 'label', 'window_uuid'])
    start_idx = int(df.iloc[0, df.columns.get_loc("time")])
    end_idx = int(df.iloc[len(df)-1, df.columns.get_loc("time")])
    labels.loc[0] = ({
        'time_start': start_idx,
        'time_end': end_idx,
        'label': label,
        'window_uuid': file.split("/")[-1]
    })

    return labels, None


def function_on_window(df: pd.DataFrame, df_windows: pd.DataFrame, function, column: str):
    """Applies a specific function for a single variable to each sliding window.
    """

    result = []
    for _, row in df_windows.iterrows():
        if type(function) == str:
            result.append(df[(df["time"] >= row['time_start']) & (
                df["time"]) <= row['time_end']][column].apply(function))
        else:
            result.append(df[(df["time"] >= row['time_start']) & (
                df["time"]) <= row['time_end']][column].pipe(function))
    if len(result) == len(df_windows):
        return result, None
    return [], "mismatch_between_len_of_files"


def median_absolute_deviation(df: pd.Series):
    """Returns a robust measure of the variability: Median of the absolute distances to median
    """
    return abs(df - df.median()).median()


def root_mean_square(df: pd.Series):
    """Returns the root of the mean of squares
    """
    if not isinstance(df, np.float64):
        df.to_numpy()
    return np.sqrt(np.nanmean(df ** 2))


def variation_coefficient(df: pd.Series):
    """returns the coefficient of variation
    """
    mean = df.mean()
    # The following check ensures that the values of the var_coef do not explode when mean -> 0
    if abs(mean) < 0.001:
        if mean < 0:
            mean = -0.001
        else:
            mean = 0.001
    return df.std() / mean


def first_location_of_maximum(df: pd.Series):
    """ returns the location of the first global maximum [0,1]
    """
    if len(df) == 0:
        return 0
    return df.idxmax() / (len(df)-1)


def power_spectral_entropy(df: pd.Series):
    """ returns the power spectral entropy of the given data
    """
    df_clean = df.dropna(how='any').copy()
    y = fft(df_clean.values)
    if len(y) == 0:
        return 0
    psd = (abs(y) ** 2)/len(y)
    if np.sum(psd) == 0:
        return 0
    p = psd / np.sum(psd)
    p[p == 0] = 10**-10
    return -np.sum((p)*np.log10(p))


def magnitude_area(df: pd.Series):
    if len(df) == 0:
        return 0
    return np.sum(abs(df).sum(axis=1)) / len(df)


def feature_functions_dict(datapath: str):
    """returns a dict of all functions to create features from
    """
    error = file_utils.valid_anonymize_file_found(datapath)
    if error:
        return {}, error

    with open(os.path.join(datapath, "anonymization_file.json")) as f:
        anonymize_file: dict = json.load(f)

    magnitude_area_cols: list = anonymize_file.get("feature_utils").get("accel_cols")[0:3]
    magn_feat_name: str = anonymize_file.get("feature_utils").get("magn_feat_name")[0]

    on_single_cols: list = anonymize_file.get("feature_utils").get("on_single_column")
    on_single_cols.extend([median_absolute_deviation, root_mean_square, variation_coefficient])

    feature_dict = {
        'on_single_column': on_single_cols,
        'on_multiple_columns': [
            {
                'function': magnitude_area,
                'columns': magnitude_area_cols,
                'feature_name': magn_feat_name
            }
        ]
    }

    return feature_dict, None


def generate_features(feature_functions: dict, df_extended: pd.DataFrame, df_boundaries: pd.DataFrame):
    """ Generates features for given data
    """
    # Create all features based in a single column
    for column in df_extended.columns:
        # don't calculate any features for the time column
        if column == 'time':
            continue
        for feature_function in feature_functions['on_single_column']:
            if type(feature_function) == str:
                feature_name = feature_function
            else:
                feature_name = feature_function.__name__

            error = None
            df_boundaries[f'{column}_{str(feature_name)}'], error = function_on_window(
                df_extended, df_boundaries, feature_function, column)
            if error:
                return pd.DataFrame(), error

    # Create all features based on multiple columns
    for multi_column_feature in feature_functions['on_multiple_columns']:
        if not set(['function', 'columns', 'feature_name']).issubset(set(multi_column_feature.keys())):
            logger.warning(
                "not processing multi column feature because declarations are missing")
            continue
        multiple_col_feature, error = function_on_window(
            df_extended, df_boundaries, multi_column_feature['function'], multi_column_feature['columns'])
        if error:
            return pd.DataFrame(), error
        df_boundaries[multi_column_feature['feature_name']
                      ] = multiple_col_feature
    return df_boundaries, None
