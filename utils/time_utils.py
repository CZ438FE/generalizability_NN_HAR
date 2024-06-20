import pandas as pd
import numpy as np
from datetime import datetime

from config.log import logger


def transform_time_unit(df: pd.DataFrame, target_unit:str, time_col:str):
    """This function transforms the time column into unix time in ms
    """
    if target_unit not in ["s", "ms"]:
        logger.error(f" Got invalid time unit {target_unit}, therefore no conversion possible")
        return pd.DataFrame(), f"invalid-target_unit_{target_unit}_given"

    # As the time is handled in milliseconds, no transforming is needed if it already is in the correct time
    if target_unit == "s":
        final_len = 10
    elif target_unit == "ms":
        final_len = 13
    # Cannot generate a common factor of multiplication for the transformation, if the data exist in different time-related precision
    identical_length = len(str(df[time_col].max())) == len(str(df[time_col].min()))
    if not identical_length:
        logger.error("Cannot find common factor for multiplication, if input column has varying degrees of precision")
        return pd.DataFrame(), "transforming_failed_multiple_degrees_of_precision_in_time_col"

    # If the time is already formatted to the correct unit, no transformation is needed
    if len(str(df[time_col][0])) == final_len:
        return df, None

    # Otherwise use the difference in lengths to find the correct factor for transformation
    len_diff = final_len - len(str(df[time_col][0]))
    df[time_col] = np.multiply(df[time_col].to_numpy(), 10**len_diff).astype(int)

    identical_length = len(str(df[time_col].max())) == len(str(df[time_col].min()))
    if not identical_length:
        logger.error(f"Something went wrong with the transformation, as the resulting time_col has varying degrees of precision")
        return pd.DataFrame(), "transformed_time_unit_still_various_degrees_of_precision"

    return df, None


def bring_col_to_greatest_len(df: pd.DataFrame, col: str):
    """returns a modified version of the df where every entry in given integer col is of the same cardinality
    """
    if 0 in df[col].unique():
        logger.error("Cannot bring entry 0 to maxlen, returned empty df")
        return pd.DataFrame(), "cannot_bring_0_to_maxlen"
    if not isinstance(df[col][0], np.int64):
        logger.error(f"Adjusting cardinality of col {col}, failed: wrong dtype given")
        return pd.DataFrame(), f"got_col_{col}_of_nonint_dt"

    max_len = len(str(df[col].max()))
    min_len = len(str(df[col].min()))

    while min_len < max_len:
        df[col] = np.where(df[col].to_numpy() * 10 < df[col].max(), df[col].to_numpy() * 10, df[col].to_numpy())
        min_len = len(str(df[col].min()))

    return df, None


def error_in_time_col(df: pd.DataFrame, time_col: str):
    """Checks if the seen time in the time col has realistic values,
    """
    min_time = np.amin(np.divide(df[time_col].to_numpy(), 1000).astype(int))
    max_time = np.amax(np.divide(df[time_col].to_numpy(), 1000).astype(int))

    # Check if the seen data has an unrealistic low starting time
    if datetime.fromtimestamp(min_time).year < 2018:
        logger.error(f"The earliest seen data comes from a year smaller than 2018: {datetime.fromtimestamp(min_time).year} Investigation needed")
        return "earliest_date_seen_from_year_smaller_2018"
    # Check if the seen data has an unrealistic high ending time
    if datetime.fromtimestamp(max_time).year > 2023:
        logger.error(f"The latest seen data comes from a year greater than 2023: {datetime.fromtimestamp(max_time).year} Investigation needed")
        return "latest_date_seen_from_year_greater_2023"
    # Check if the seen data goes over an unrealistic long period of time
    eight_hours_in_s = 8*60*60
    if (max_time-min_time) > eight_hours_in_s:
        logger.error(f"The seen session duration {(max_time-min_time/3600)}h is longer than 8hours. Investigation needed")
        return "seen_duration_longer_eight_hours"

    return None


def transform_sci_not(df: pd.DataFrame, time_col: str):
    """Checks if the time column of the df is in scientific notation and transforms to int
    """
    # When the time col already has the wanted datatype, no conversion is necessary
    if isinstance(df[time_col][0], np.int64):
        return df, None

    # Test if any of the ways of marking scientific notation are in the data
    list_of_scientific_markers = ["E+", "e+"]
    sci_marker = None
    for marker in list_of_scientific_markers:
        if marker in df[time_col][0]:
            sci_marker = marker
            break

    if sci_marker is None:
        logger.error(f"Time column is neither an int, nor in scientific notation, therefore no processing possible")
        return pd.DataFrame(), "time_col_contains_unknown_datatype"

    # Transform the sci time to UNIX and overwrite it
    vect_func = np.vectorize(transform_sci_not_to_unix_e)
    if marker == "E+":
        vect_func = np.vectorize(transform_sci_not_to_unix_E)

    df[time_col] = vect_func(df[time_col].to_numpy())

    return df, None


def transform_sci_not_to_unix_e(time_in_sci_not: str):
    """transforms the time in sci notation into unix time with e+ as separator
    """
    exponent = int(time_in_sci_not.split("e+")[1])
    return round(float(time_in_sci_not.split("e+")[0].replace(",", "."))*(10**exponent))


def transform_sci_not_to_unix_E(time_in_sci_not: str):
    """transforms the time in sci notation into unix time with E+ as separator
    """
    exponent = int(time_in_sci_not.split("E+")[1])
    return round(float(time_in_sci_not.split("E+")[0].replace(",", "."))*(10**exponent))


def resample_data(df: pd.DataFrame, sample_rate='10ms', time_unit='ms'):
    """ Adds a datetime column and resamples the data frame with the sample rate by calculating the mean values.
    """
    if "time" not in df.columns:
        logger.error(f"Cannot resample the df without a time col")
        return pd.DataFrame(), "time_col_not_found"
    if not isinstance(df["time"][0], np.int64):
        logger.error(f"received time col with non-int dt: {type(df['time'][0])}")
        return pd.DataFrame(), "time_col_has_non_int_entries"

    # Check if the resampling rate is in a correct format
    if "ms" not in sample_rate:
        logger.error(f"received invalid resampling rate {sample_rate}")
        return pd.DataFrame(), "ms_not_found_in_given_sampling_rate"
    try:
        resampling_int = int(sample_rate.split("ms")[0])
    except ValueError:
        logger.error(f"Could not generate a valid duration for resampling from {sample_rate}")
        return pd.DataFrame(), f"generating_valid_sampling_dur_from_{sample_rate}_failed"
    if resampling_int < 2:
        logger.error(f"The given resampling rate is too small: {resampling_int}. Must be greater 1")
        return pd.DataFrame(), "sampling_rate_smaller_two_given"
    if resampling_int > 100:
        logger.error(f"The given resampling rate is too big: {resampling_int}. Max: 100")
        return pd.DataFrame(), "sampling_rate_greater_100ms_given"

    df_new = df.copy()

    # Sort the df by the time column
    df_new = df_new.sort_values(by=["time"]).reset_index(drop=True)

    # Resample the df
    df_new['datetime'] = pd.to_datetime(df_new['time'], unit=time_unit)
    df_resampled = df_new.resample(rule=f'{sample_rate}', on='datetime').mean()

    # Drop all the entries with NaN values in the time col, as it is not know when the seen data happened
    df_resampled = df_resampled[df_resampled['time'].notna()]

    # As the resampling might create timepoints which are not integers, convert these to ints
    df_resampled = df_resampled.astype({"time": np.int64}).reset_index(drop=True)

    return df_resampled, None


def valid_time_string(time_string: str):
    """ Tests if a given string has a valid format of YYYY-MM-DDThh:mm
    """
    if time_string is None:
        return None

    if not isinstance(time_string, str):
        logger.error(f"Wrong datatype for the time_string given. Expected str, got {type(time_string)}")
        return "got_non_str_time_string"
    # Check if the string has the wanted length
    if len(time_string) != 16:
        logger.error(f"Got a string of an invalid length, expected 16 , got  {len(time_string)}. Wanted format: YYYY-mm-DDTHH:MM")
        return "got_time_str_of_invalid_len"
    # Check if the string contains the correct amount of separators
    if time_string.count("-") != 2:
        logger.error(f"Expected 2 - as separators in the time_string. got {time_string.count('-')} in string {time_string}")
        return "wrong_amount_of_-_separators_in_time_string"
    if time_string.count("T") != 1:
        logger.error(f"Expected 1 T as separator in the time_string. got {time_string.count('T')}")
        return "wrong_amount_of_T_separators_in_time_string"
    # Check if the time blocks contain valid values
    try:
        year_block = int(time_string[0:4])
        month_block = int(time_string[5:7])
        day_block = int(time_string[8:10])
        hour_block = int(time_string[11:13])
        minute_block = int(time_string[14:16])
    except ValueError:
        logger.error(
            f"Could not generate valid time ints from the time_string {time_string}")
        return "could_not_gather_time_blocks_from_time_string"
    return None
