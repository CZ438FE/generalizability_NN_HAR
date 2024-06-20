import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from utils import time_utils


def test_transform_time_unit():
    # Test if the data is already in the right format
    df = pd.DataFrame({"numbers": [1666092096000, 1666092098000]})
    got_df, got_error = time_utils.transform_time_unit(df, "ms", "numbers")
    assert_frame_equal(got_df, df)
    assert got_error is None

    # Test if the data is too precise
    precise_df = pd.DataFrame({"numbers": [1666092096000000, 1666092098000000]})
    got_df, got_error = time_utils.transform_time_unit(precise_df, "ms", "numbers")
    precise_df["numbers"] = precise_df["numbers"].astype(np.int32)
    assert_frame_equal(got_df, precise_df)
    assert got_error is None

    # Test if the resolution of the data is too low
    unprecise_df = pd.DataFrame({"numbers": [1666, 1666]})
    got_df, got_error = time_utils.transform_time_unit(unprecise_df, "ms", "numbers")
    extrapolated_df = pd.DataFrame({"numbers": [1666000000000, 1666000000000]})
    extrapolated_df["numbers"] = extrapolated_df["numbers"].astype(np.int64)
    assert_frame_equal(got_df, extrapolated_df)
    assert got_error is None


def test_bring_col_to_greatest_len():
    # test if the entries cannot be brought to the same len
    test_df = pd.DataFrame({"numbers": [1, 2, 200], "core": [12, 200, 0]})
    want_df = pd.DataFrame()
    got_df, got_error = time_utils.bring_col_to_greatest_len(test_df, "core")
    assert_frame_equal(got_df, want_df)
    assert got_error == "cannot_bring_0_to_maxlen"

    # test if the col contains wrong datatype
    test_df = pd.DataFrame({"numbers": [1, 2, 200], "core": ["12", "2hgjg00", "dsfsdf0"]})
    want_df = pd.DataFrame()
    got_df, got_error = time_utils.bring_col_to_greatest_len(test_df, "core")
    assert_frame_equal(got_df, want_df)
    assert got_error == "got_col_core_of_nonint_dt"

    # test if everything works as intended with valid input
    test_df = pd.DataFrame({"numbers": [1, 2, 200], "core": [12, 440, 20]})
    want_df = pd.DataFrame({"numbers": [1, 2, 200], "core": [120, 440, 200]})
    got_df, got_error = time_utils.bring_col_to_greatest_len(test_df, "core")
    assert_frame_equal(got_df, want_df)
    assert got_error is None


def test_time_column_valid():
    # Check if it correctly dismisses if the seen date is too small
    test_df = pd.DataFrame({"numbers": [1, 2], "core": [12, 20]})
    assert time_utils.error_in_time_col(test_df, "core") == "earliest_date_seen_from_year_smaller_2018"
    # Check if it correctly dismisses if the seen date is too big
    test_df = pd.DataFrame({"numbers": [1, 2], "core": [2041408633505, 2041408633505]})
    assert time_utils.error_in_time_col(test_df, "core") == "latest_date_seen_from_year_greater_2023"
    # Check if it correctly dismisses if the seen period is too big
    test_df = pd.DataFrame({"numbers": [1, 2], "core": [1641408633505, 1641408633505 + 10*60*60*1000]})
    assert time_utils.error_in_time_col(test_df, "core") == "seen_duration_longer_eight_hours"
    # Check if it correctly accepts if the seen period is valid
    test_df = pd.DataFrame({"numbers": [1, 2], "core": [1641408633505, 1641408633905]})
    assert time_utils.error_in_time_col(test_df, "core") is None


def test_transform_sci_not():
    # test correct returning if the time col does not have scientific notation
    test_df = pd.DataFrame({"numbers": [1, 2], "core": [1641408633505, 1641408633905]})
    got_df, got_error = time_utils.transform_sci_not(test_df, "core")
    assert_frame_equal(test_df, got_df)
    assert got_error is None

    # Test correct processing if the time col contains data which cannot be managed
    test_df = pd.DataFrame({"numbers": [1, 2], "core": ["sdklfhlskdh", "ksdjfnkdfn"]})
    got_df, got_error = time_utils.transform_sci_not(test_df, "core")
    assert_frame_equal(pd.DataFrame(), got_df)
    assert "time_col_contains_unknown_datatype" == got_error

    # Test correct processing if the marker E+ is being used, common exponent
    test_df = pd.DataFrame({"numbers": [1, 2], "core": ["1,657007905832E+018", "1,657007905843E+018"]})
    want_df = pd.DataFrame({"numbers": [1, 2], "core": [1657007905832000000, 1657007905843000000]})
    got_df, got_error = time_utils.transform_sci_not(test_df, "core")
    assert_frame_equal(want_df, got_df)
    assert got_error is None

    # Test correct processing if the marker E+ is being used, common exponent
    test_df = pd.DataFrame({"numbers": [1, 2], "core": ["1,657007905832E+018", "1,657007905843E+018"]})
    want_df = pd.DataFrame({"numbers": [1, 2], "core": [1657007905832000000, 1657007905843000000]})
    got_df, got_error = time_utils.transform_sci_not(test_df, "core")
    assert_frame_equal(want_df, got_df)
    assert got_error is None

def test_resample_data():
    # test when the df has no time col
    test_df = pd.DataFrame({"not_the_time_col": [1667989727000, 1667989727006], "some_col": [12, 14]})
    got_df, got_error = time_utils.resample_data(test_df, "not_a_valid_target_unit")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "time_col_not_found"

    # test when the time col does not contain ints
    test_df = pd.DataFrame({"time": ["1667989727000", "1667989727006"], "some_col": [12, 14]})
    got_df, got_error = time_utils.resample_data(test_df, "time_col_has_non_int_entries")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "time_col_has_non_int_entries"

    # test if invalid sample_rate is given
    test_df = pd.DataFrame({"time": [1667989727000, 1667989727006], "some_col": [12, 14]})
    got_df, got_error = time_utils.resample_data(test_df, "not_a_valid_target_unit")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "ms_not_found_in_given_sampling_rate"

    # test when the given sample rate cannot be deconstructed
    test_df = pd.DataFrame({"time": [1667989727000, 1667989727006], "some_col": [12, 14]})
    got_df, got_error = time_utils.resample_data(test_df, "balbms1561")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "generating_valid_sampling_dur_from_balbms1561_failed"

    # test when the given sample rate is too small
    test_df = pd.DataFrame({"time": [1667989727000, 1667989727006], "some_col": [12, 14]})
    got_df, got_error = time_utils.resample_data(test_df, "1ms")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "sampling_rate_smaller_two_given"

    # test when the given sample rate is too big
    test_df = pd.DataFrame({"time": [1667989727000, 1667989727006], "some_col": [12, 14]})
    got_df, got_error = time_utils.resample_data(test_df, "10000ms")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "sampling_rate_greater_100ms_given"

    # Test if everything works as intended
    test_df = pd.DataFrame({"time": [1667989727000, 1667989727006], "some_col": [12, 14]})
    got_df, got_error = time_utils.resample_data(test_df)
    want = pd.DataFrame({"time": [1667989727003], "some_col": [13.0]})
    assert_frame_equal(got_df, want)
    assert got_error is None


def test_valid_time_string():
    # test if a wrong datatype was given
    time_string_wrong_datatype = ["i am a list, not a string"]
    got = time_utils.valid_time_string(time_string_wrong_datatype)
    assert got == "got_non_str_time_string"

    # test if a wrong len was given:
    time_string_wrong_length = "2000-01-01T00-00-00"
    got = time_utils.valid_time_string(time_string_wrong_length)
    assert got == "got_time_str_of_invalid_len"

    # test if the format is wrong: wrong separators used: _ instead of -
    time_string_wrong_date_seperator = "2000_01_01T00-00"
    got = time_utils.valid_time_string(time_string_wrong_date_seperator)
    assert got == "wrong_amount_of_-_separators_in_time_string"

    # test if the format is wrong: wrong separators used: . insted of T
    time_string_wrong_middle_separator = "2000-01-01.00:00"
    got = time_utils.valid_time_string(time_string_wrong_middle_separator)
    assert got =="wrong_amount_of_T_separators_in_time_string"

    # test if the separators are correct but the contents are mixed up
    time_string_content_mixed = "01-01-2000:00T00"
    got = time_utils.valid_time_string(time_string_content_mixed)
    assert got == "could_not_gather_time_blocks_from_time_string"
    # test if everything is correct
    valid_time_string = "2000-01-01T00:00"
    got = time_utils.valid_time_string(valid_time_string)
    assert got is None
