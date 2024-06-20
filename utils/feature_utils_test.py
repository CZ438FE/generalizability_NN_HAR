import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from numpy import NaN
from pandas.testing import assert_frame_equal

from utils import feature_utils, file_utils
from config.test_config import test_config

def test_expand_df():
    # Test if an empty df gets returned when a invalid df gets inserted
    inval_df = 15
    got_df, got_error = feature_utils.expand_df(inval_df, {})
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "got_df_without_dtype_pdDataframe"

    # Test with fully correct df
    correct_df = pd.DataFrame({"column_a": [1, 1], "column_b": [1, 1], "column_c": [1, 1], "column_d": [50, 0],
                                "column_e": [0, 50], "time": [1000, 1500], "column_f": [70, 75],
                                "column_g": [10, 15], "column_h": [0, 2], "column_i": [0, 3]})
    result_df = pd.DataFrame({"column_a": [1, 1], "column_b": [1, 1], "column_c": [1, 1], "column_d": [50, 0],
                                "column_e": [0, 50], "time": [1000, 1500], "column_f": [70, 75], "column_g": [10, 15],
                                "column_h": [0, 2], "column_i": [0, 3], "column_j": [50, 50],
                                "column_k": [1.732051, 1.732051], "column_a_grad": [0.0, 0.0], "column_b_grad": [0.0, 0.0],
                                "column_c_grad": [0.0, 0.0], "column_d_grad": [0.0, -50.0], "column_e_grad": [0.0, 50.0],
                                "column_f_grad": [0.0, 5.0], "column_g_grad": [0.0, 5.0], "column_h_grad": [0.0, 2.0],
                                "column_i_grad": [0.0, 3.0], "column_j_grad": [0.0, 0.0], "column_k_grad": [0.0, 0.0]})

    got_df, got_error = feature_utils.expand_df(correct_df, {"feature_utils": {"needed_cols": ["column_a", "column_b", "column_c", "column_d", "column_e", "time", "column_f"],
                                                                                "differences_cols": ["column_d", "column_e", "column_j"],
                                                                                "accel_cols": ["column_a", "column_b", "column_c", "column_k"]}})
    assert got_error is None
    assert_frame_equal(got_df, result_df)


def test_extend_with_differences():
    # test if a df with incorrect column names is being given:
    malformed_df = pd.DataFrame({"wrong_colname": [1234, 32]})
    valid_anonymize_file = {"feature_utils": {"differences_cols": ["column_d", "column_e", "column_j"]}}
    got_df, got_error = feature_utils.extend_with_differences(
        malformed_df, valid_anonymize_file)
    assert got_error == "differences_cols_not_found_calculation_of_diff_failed"
    assert_frame_equal(got_df, pd.DataFrame())

    # Test if the differences col gets appended correctly
    valid_df = pd.DataFrame({'column_d': [100, 80], 'column_e': [20, 30]})
    want_df = pd.DataFrame({'column_d': [100, 80], 'column_e': [
                            20, 30], "column_j": [80, 50]})
    got_df, got_error = feature_utils.extend_with_differences(valid_df, valid_anonymize_file)
    assert got_error is None
    assert_frame_equal(got_df, want_df)


def test_extend_with_acceleration():
    # test if acceleration does get computed correctly
    df = pd.DataFrame({
        'column_a': [1, 0, 0, None],
        'column_b': [0, 2, 0, 5],
        'column_c': [0, 0, 3, 0],
    })
    want_df = pd.DataFrame({
        'column_a': [1, 0, 0, None],
        'column_b': [0, 2, 0, 5],
        'column_c': [0, 0, 3, 0],
        'column_zz': [1.0, 2.0, 3.0, None]
    })
    valid_anonymize_file = {"feature_utils": {"accel_cols": [
        "column_a", "column_b", "column_c", "column_zz"]}}
    got_df, got_error = feature_utils.extend_with_acceleration(
        df, valid_anonymize_file)
    assert got_error is None
    assert_frame_equal(got_df, want_df)

    # Test if an empty Dataframe gets returned when the needed columns are missing
    df_not_valid = pd.DataFrame({
        'column_a': [1],
        'column_b': [2],
        'not_column_zz': [3],
    })
    got_df, got_error = feature_utils.extend_with_acceleration(df_not_valid, valid_anonymize_file)
    assert got_error == "needed_accel_cols_not_found"
    assert_frame_equal(got_df, pd.DataFrame())


def test_extend_with_gradients():
    df = pd.DataFrame({
        "time": [10, 20, 30, 40, 50],
        "column_a": [1, 0, 0, 4, 0],
        "column_b": [0, 2, 0, 0, 5],
        "column_c": [0, 0, 3, 0, 0],
        "column_d": [1, 2, 3, 4, 5],
        "column_e": [5, 4, 3, 2, 1],

    })

    want_df = pd.DataFrame({
        "time": [10, 20, 30, 40, 50],
        "column_a": [1, 0, 0, 4, 0],
        "column_b": [0, 2, 0, 0, 5],
        "column_c": [0, 0, 3, 0, 0],
        "column_d": [1, 2, 3, 4, 5],
        "column_e": [5, 4, 3, 2, 1],

        "column_a_grad": [0., -1., 0., 4., -4.],
        "column_b_grad": [0., 2., -2., 0., 5.],
        "column_c_grad": [0., 0., 3., -3., 0.],
        "column_d_grad": [0., 1., 1., 1., 1.],
        "column_e_grad": [0., -1., -1., -1., -1.],
    })
    got_df = feature_utils.extend_with_gradients(df)
    assert_frame_equal(got_df, want_df)


def test_is_valid():
    # test when an wrong datatype is given
    df_wrong_datatype = 17
    assert feature_utils.is_valid(df_wrong_datatype, {}) == "got_df_without_dtype_pdDataframe"

    # test when the column names are incorrect
    df_wrong_colnames = pd.DataFrame({
        "time_afs": [10, 20, 30, 40, 50],
        "wrong_colname": [1, 0, 0, 4, 0],
    })
    assert feature_utils.is_valid(df_wrong_colnames, {"feature_utils": {"needed_cols": ["some_col"]}}) == "needed_cols_missing"

    # test if None gets returned when the df is valid
    valid_df = pd.DataFrame({
        "time": [10, 20, 30, 40, 50],
        "column_a": [1, 0, 0, 4, 0],
        "column_b": [0, 2, 0, 0, 5],
        "column_c": [0, 0, 3, 0, 0],
        "column_g": [1, 0, 0, 4, 0],
        "column_h": [0, 2, 0, 0, 5],
        "column_i": [0, 0, 3, 0, 0],
        "column_d": [1, 2, 3, 4, 5],
        "column_e": [5, 4, 3, 2, 1],
        "column_f": [5, 4, 3, 2, 1],
    })
    got = feature_utils.is_valid(valid_df, {"feature_utils": {"needed_cols": ["time", "column_a", "column_b", "column_c", "column_g", "column_h", "column_i", "column_d", "column_e", "column_f"]}})
    assert got is None


def test_create_df_boundaries():
    # Test if the wrong label is given
    valid_df = pd.DataFrame({"labels": [12, 2, 3, 4], "time": [1, 3, 5, 6]})
    wrong_label = 12.2312
    file = "a test file"
    got_df, got_error = feature_utils.create_df_boundaries(valid_df, wrong_label, file)
    assert got_error == "given_label_not_int64"
    assert_frame_equal(got_df, pd.DataFrame())

    # Test if time is not in the columns of the df
    invalid_df = pd.DataFrame({"labels": [12, 2, 3, 4]})
    got_df, got_error = feature_utils.create_df_boundaries(invalid_df, np.int64(15), file)
    assert got_error == "time_col_missing"
    assert_frame_equal(got_df, pd.DataFrame())

    # Test if the correct df gets returned when the df is valid
    valid_df = pd.DataFrame({"labels": [12, 2, 3, 4], "time": [1, 3, 5, 6]})
    label = np.int64(15)
    got_df, got_error = feature_utils.create_df_boundaries(valid_df, label, file)
    want = pd.DataFrame({"time_start": [1], "time_end": [6], "label": [label], "window_uuid": [file]})
    want = want.astype({"time_start": np.int64, "time_end": np.int64, "label": np.int64, "window_uuid": object})
    assert got_error is None
    assert_frame_equal(got_df, want)


def test_function_on_window():
    df = pd.DataFrame({
        'datetime': [datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 1), datetime(2000, 1, 1, 0, 0, 2), datetime(2000, 1, 1, 0, 0, 3), datetime(2000, 1, 1, 0, 0, 4)],
        'time': [946681200, 946681201, 946681202, 946681203, 946681204],
        'v_1': [-100, 1, None, 10, 500],
        'v_1_NaN': [-100, 1, NaN, 10, 500],
        'v_2': [0.123, 0, NaN, 5, -123]
    })
    df.set_index('datetime', inplace=True)
    df_windows = pd.DataFrame({
        'time_start': [946681200],
        'time_end': [946681204]
    })

    # test string function
    want = [500]
    got_result, got_error = feature_utils.function_on_window(df, df_windows, 'max', 'v_1')
    assert got_result == want
    got_result, got_error = feature_utils.function_on_window(df, df_windows, 'max', 'v_1_NaN')
    assert len(got_result) ==len(want)

    # test defined function
    want = [2.6227506679548855]
    got_result, got_error = feature_utils.function_on_window(df, df_windows, feature_utils.variation_coefficient, 'v_1')
    assert got_result == want
    got_result, got_error = feature_utils.function_on_window(df, df_windows, feature_utils.variation_coefficient, 'v_1_NaN')
    assert len(got_result) == len(want)


def test_median_absolute_deviation():
    df = pd.DataFrame({
        'v_1': [1, 2, None, 3],
        'v_1_NaN': [1, 2, NaN, 3],
    })
    want = 1.0
    got = feature_utils.median_absolute_deviation(df['v_1'])
    assert got == want
    got = feature_utils.median_absolute_deviation(df['v_1_NaN'])
    assert got == want


def test_root_mean_square():
    df = pd.DataFrame({
        'v_1': [3, 3, None, -3],
        'v_1_NaN': [3, 3, NaN, -3],
    })
    want = 3.0
    got = feature_utils.root_mean_square(df['v_1'])
    assert got == want
    got = feature_utils.root_mean_square(df['v_1_NaN'])
    assert got == want


def test_variation_coefficient():
    df = pd.DataFrame({'v_1': [60, 56, 61, 68, 51, 53, 69, 54, NaN]})
    want = 0.11459718708183275
    got = feature_utils.variation_coefficient(df['v_1'])
    assert got == want


def test_first_location_of_maximum():
    df = pd.Series([1., 1., 2., 5., 5., 3., 1., 2., NaN])
    want = 0.375
    got = feature_utils.first_location_of_maximum(df)
    assert got == want


def test_power_spectral_entropy():
    df = pd.DataFrame({'v_1': [-2, 1, NaN, 0, -1, 2]})
    want = 0.518
    got = feature_utils.power_spectral_entropy(df)
    assert round(got, 3) == want

def test_magnitude_area():
    df = pd.DataFrame({'v_1': [-2, 1, NaN, 0, -1, 2]})
    want = 1.0
    got = feature_utils.magnitude_area(df)
    assert round(got, 3) == want


def test_feature_functions_dict():
    # test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "feature_utils_test", "test_feature_functions_dict")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    error = file_utils.save_json({"create_windows": {"needed_columns": [], "data_columns": []},
                                    "svm_features": {"top": [], "lifting": [], "walking": []},
                                    "feature_utils": {"accel_cols": ["col_1", "col_2", "col_3", "col_15"], "differences_cols": [], "needed_cols": [], "on_single_column": ["some_func"], "magn_feat_name": ["some_feature_name"]}},
                                    os.path.join(folder, "anonymization_file.json"))
    assert error is None

    got, error = feature_utils.feature_functions_dict(folder)
    assert error is None

    assert len(got["on_single_column"]) == 4
    assert got["on_multiple_columns"][0]["feature_name"] == "some_feature_name"
