import os
import shutil
import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
import numpy as np

from config.test_config import test_config
from utils import data_utils

def test_has_identical_length():
    # Test if all entries in col have the same len
    test_df = pd.DataFrame({"numbers": [1, 23, 53], "core": [22, 48, 78]})
    assert data_utils.has_identical_length(test_df, "core")

    # Test if the entries in col do not have the same len
    test_df = pd.DataFrame({"numbers": [1, 23, 53], "core": [202, 48, 78]})
    assert not data_utils.has_identical_length(test_df, "core")


def test_only_nan():
    # Test if only 0 in col
    test_df = pd.DataFrame({"numbers": [1, 23, 53], "core": [0, 0, 0]})
    assert data_utils.only_nan(test_df, ["core"]) is None

    # Test if nan in col
    test_df = pd.DataFrame({"numbers": [1, 23, 53], "core": [np.nan, np.nan, np.nan]})
    assert data_utils.only_nan(test_df, ["core"]) == "contains_only_nan"

    # Test if mixed values col
    test_df = pd.DataFrame({"numbers": [1, 23, 53], "core": [12, np.nan, 15]})
    assert data_utils.only_nan(test_df, ["core"]) is None


def test_valid_window_and_step_length():
    # test when no window_length is given
    assert data_utils.valid_window_and_step_length(None, 200) == "window_length_is_None"

    # test when  no step length is given
    assert data_utils.valid_window_and_step_length(1000, None) == "step_length_is_None"

    # test when the window length is too small
    assert data_utils.valid_window_and_step_length(-250, 200) ==  "given_window_length_smaller_1"

    # test when the step length is too small
    assert data_utils.valid_window_and_step_length(2050, -35) == "given_step_length_smaller_1"

    # test when the step length is greater than the window length
    assert data_utils.valid_window_and_step_length(100, 100000) ==  "step_length_greater_window_length"

    # Test when everything is valid
    assert data_utils.valid_window_and_step_length(800, 200) is None


def test_create_filled_df():
    # Test if an invalid filling_method was given
    valid_df = pd.DataFrame({"time": [1000, 1010], "label": [12, 12], "some_col": [10, 12]})
    got_df, got_error = data_utils.create_filled_df(valid_df, "an_invalid_filling_method")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "invalid_filling_method_an_invalid_filling_method_given"

    # Test if a df of an invalid type was given
    valid_df = pd.DataFrame({"time": [1000, 1010], "label": [12, 12], "some_col": [10, 12]})
    got_df, got_error = data_utils.create_filled_df([valid_df], "ffill")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == f"expected_pdDataFrame_got_{type([])}"

    # Test if the label-col is missing from the df
    valid_df = pd.DataFrame({"time": [1000, 1010],  "some_col": [10, 12]})
    got_df, got_error = data_utils.create_filled_df(valid_df, "ffill")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "label_col_missing"

    # Test if the time-col is missing from the df
    valid_df = pd.DataFrame({"label": [12, 12], "some_col": [10, 12]})
    got_df, got_error = data_utils.create_filled_df(valid_df, "ffill")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "time_col_missing"

    # Test if everything works as intended, when the filling method ffill is being used
    valid_df = pd.DataFrame({"time": [1000, 1010], "label": [
                            12, 12], "some_col": [10, 12]})
    got_df, got_error = data_utils.create_filled_df(valid_df, "ffill")
    want_df = pd.DataFrame({"time": [x for x in range(1000, 1011)],
                            "some_col": [10.0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12],
                            "label": [12]*11
                            })
    assert_frame_equal(got_df, want_df)
    assert got_error is None

    # Test if everything works as intended, when the filling method linear is being used
    valid_df = pd.DataFrame({"time": [1000, 1009], "label": [12, 12], "some_col": [10, 19]})
    got_df, got_error = data_utils.create_filled_df(valid_df, "linear")
    want_df = pd.DataFrame({"time": [x for x in range(1000, 1010)],
                            "some_col": [float(x) for x in range(10, 20)],
                            "label": [12]*10
                            })
    assert_frame_equal(got_df, want_df)
    assert got_error is None


def test_flatten():
    # test if the time col is missing
    invalid_df = pd.DataFrame({"label": [11, 11], "some_col": [12, 13]})
    got_df, got_error = data_utils.flatten(invalid_df, "some_file.csv")
    assert got_error == "time_or_label_not_in_df_to_flatten"
    assert_frame_equal(got_df, pd.DataFrame())

    # test if the label col is missing
    invalid_df = pd.DataFrame({"time": [11, 11], "some_col": [12, 13]})
    got_df, got_error = data_utils.flatten(invalid_df, "some_file.csv")
    assert got_error == "time_or_label_not_in_df_to_flatten"
    assert_frame_equal(got_df, pd.DataFrame())

    # test if the wrong amount of cols was given
    invalid_df = pd.DataFrame({"time": [11, 11], "some_col": [12, 13], "label": [12, 12]})
    got_df, got_error = data_utils.flatten(invalid_df, "some_file.csv")
    assert got_error == "wrong_col_amount_expected_11_got_3"
    assert_frame_equal(got_df, pd.DataFrame())

    # test if the function works as intended when the time col is dropped
    valid_df = pd.DataFrame({"time": [11, 12]*50, "label": [10, 10]*50, "some_col_a": [12, 13]*50, "some_col_b": [12, 13]*50, "some_col_c": [12, 13]*50, "some_col_d": [12, 13]*50,
                            "some_col_e": [12, 13]*50, "some_col_f": [12, 13]*50, "some_col_g": [12, 13]*50, "some_col_h": [12, 13]*50, "some_col_i": [12, 13]*50})
    flattened_colnames = []
    for iterations in range(len(valid_df)):
        for colname in valid_df.columns:
            if colname in ["label", "time"]:
                continue
            flattened_colnames.append(f"{colname}_{str(iterations)}")
    entries = [12]*9
    entries.extend([13]*9)
    want_df = pd.DataFrame({"0": entries*50}).T
    want_df.index = pd.RangeIndex(start=0, stop=1, step=1)
    want_df.columns = flattened_colnames
    want_df["label"] = 10

    got_df, got_error = data_utils.flatten(valid_df, "some_file.csv")
    assert got_error is None 
    assert_frame_equal(got_df, want_df)

    # test if the function works as intended when the time col is kept
    valid_df = pd.DataFrame({"time": [11, 12]*50, "label": [10, 10]*50, "some_col_a": [12, 13]*50, "some_col_b": [12, 13]*50, "some_col_c": [12, 13]*50, "some_col_d": [12, 13]*50,
                            "some_col_e": [12, 13]*50, "some_col_f": [12, 13]*50, "some_col_g": [12, 13]*50, "some_col_h": [12, 13]*50, "some_col_i": [12, 13]*50})
    entries = [11]
    entries.extend([12]*9)
    entries.append(12)
    entries.extend([13]*9)

    flattened_colnames = []
    for iterations in range(len(valid_df)):
        for colname in valid_df.columns:
            if colname == "label":
                continue
            flattened_colnames.append(f"{colname}_{str(iterations)}")

    want_df = pd.DataFrame({"0": entries*50}).T
    want_df.index = pd.RangeIndex(start=0, stop=1, step=1)
    want_df.columns = flattened_colnames
    want_df["label"] = 10

    got_df, got_error = data_utils.flatten(
        valid_df, "some_file.csv", remove_time_entries=False)
    assert got_error is None
    assert_frame_equal(got_df, want_df)


def test_split_files_to_handle_into_chunks():
    # test if everything works as intended
    files_to_handle_list = [x for x in range(100)]
    want = [[x for x in range(24)], [x for x in range(24, 48)], [x for x in range(
        48, 72)], [x for x in range(72, 96)], [x for x in range(96, 100)]]
    got = data_utils.split_files_to_handle_into_chunks(
        files_to_handle_list)
    assert got == want


def test_find_first_index_bigger_x_in_sorted_list():
    # Test for ints
    example_list = [1, 2, 3, 4, 4, 5, 7, 12, 13]
    value = 8
    got = data_utils.find_first_index_bigger_x_in_sorted_list(example_list, value)
    assert got == 7
    # test for floats
    example_list = [1.8, 2.2, 3.1, 4.1, 4.5, 5.3, 7.8, 12.1, 13.9]
    value = 3.4
    got = data_utils.find_first_index_bigger_x_in_sorted_list(example_list, value)
    assert got == 3


def test_split_files_to_handle_into_chunks():
    # test if everything works as intended
    list_to_split = [x for x in range(50)]
    got = data_utils.split_files_to_handle_into_chunks(list_to_split, 10)
    want = [[x for x in range(10)], [x for x in range(10, 20)], [x for x in range(20, 30)], [x for x in range(30, 40)], [x for x in range(40, 50)]]
    assert got == want


def test_find_first_index_bigger_x_in_sorted_list():
    # test if everything works as intended
    example_list = [x for x in range(100)]
    got = data_utils.find_first_index_bigger_x_in_sorted_list(example_list, 15)
    assert got == 16


def test_create_result_string():
    # test if No errors occurred during any windows
    file = "somefile.csv"
    got = data_utils.create_result_string(file, [], 0, "i_should_not_appear", )
    want = file + "+++True+++None+++0"
    assert got == want

    # test if errors occurred during any windows
    file = "somefile.csv"
    got = data_utils.create_result_string(file, [14], 10, "i_should_appear", )
    want = file + "+++False+++i_should_appear+++10"
    assert got == want


def test_downsize_label():
    # Test if an invalid label_depth is given
    invalid_label_depth = 45
    valid_label_series = pd.Series([12, 12, 12, 12, 12])
    got_list, got_error = data_utils.downsize_label(valid_label_series, invalid_label_depth)
    assert got_list is None
    assert got_error == "invalid_label_depth_given"

    # Test if the granularity in the seen data is too small
    label_depth = 3
    got_list, got_error = data_utils.downsize_label(valid_label_series, label_depth)
    assert got_list is None
    assert got_error == "granularity_of_label_col_smaller_than_label_depth"

    # Test if downsizing works correctly when everything is valid
    label_depth = 1
    got_list, got_error = data_utils.downsize_label(valid_label_series, label_depth)
    assert_array_equal(got_list, np.ones([5]).reshape(5,))
    assert got_error is None


def test_prepare_label():
    # Test if an invalid label_depth was given
    invalid_label_depth = 14
    valid_label_series = pd.Series([120, 120, 120, 110, 110, 140])
    got_list, got_error = data_utils.prepare_label(
        valid_label_series, invalid_label_depth)
    assert got_list == []
    assert got_error == "invalid_label_depth_given"

    # Test if everything works with valid inputs
    got_list, got_error = data_utils.prepare_label(valid_label_series, 2)
    assert_array_equal(got_list, pd.Series([2, 2, 2, 1, 1, 3]).to_numpy())
    assert got_error is None


def test_label_class_mean():
    # test if there are only NA values in col
    test_df = pd.DataFrame({"only_na": [np.nan, np.nan, np.nan], "one_NA": [1, 1, np.nan], "label": [1, 1, 1]})
    got_dict, got_error = data_utils.label_class_mean(test_df, "only_na", "label")
    assert got_error == "label_class_mean_cannot_be_calculated_for_row_with_only_NAN"
    assert not got_dict

    # test if the df does not contain a label col
    test_df = pd.DataFrame({"only_na": [np.nan, np.nan, np.nan], "one_NA": [1, 1, np.nan], "some_col": [1, 1, 1]})
    got_dict, got_error = data_utils.label_class_mean(test_df, "one_NA", "label")
    assert got_error == "label_col_not_found"
    assert not got_dict

    # test if a robust mean can be calculated
    test_df = pd.DataFrame({"only_na": [np.nan, np.nan, np.nan], "one_NA": [1, 1, np.nan], "label": [1, 1, 1]})
    got_dict, got_error = data_utils.label_class_mean(test_df, "one_NA", "label")
    assert got_dict == {1: 1.}
    assert got_error is None

    # Test if the calculation works when various labels are given
    test_df = pd.DataFrame({"only_na": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], "one_NA": [
                            1, 1, np.nan, 10, 10, np.nan], "label": [1, 1, 1, 2, 2, 2]})
    got_dict, got_error = data_utils.label_class_mean(test_df, "one_NA", "label")
    assert got_dict == {1: 1., 2: 10.}
    assert got_error is None


def test_naive_imputation():
    # test if the label_col has nan
    df = pd.DataFrame({"label": [1, np.nan, 1, 1], "feature_col": [3, 3, np.nan, 3]})
    want_df = pd.DataFrame()
    got_df, got_error = data_utils.naive_imputation(df, {})
    assert_frame_equal(got_df, want_df)
    assert got_error ==  "found_nan_values_in_label_col"

    # test when the label column has only one value
    df = pd.DataFrame({"label": [1, 1, 1, 1], "feature_col": [3, 3, np.nan, 3]})
    want_df = pd.DataFrame({"label": [1, 1, 1, 1], "feature_col": [3., 3., 3., 3.]})
    got_df, got_error = data_utils.naive_imputation(df, {})
    assert_frame_equal(got_df, want_df)
    assert got_error is None

    # Test when the feature column has various levels,
    df = pd.DataFrame({"label": [1, 1, 1, 1, 2, 2, 2], "feature_col": [
                        3, 3, np.nan, 3, 7, np.nan, 7]})
    want_df = pd.DataFrame({"label": [1, 1, 1, 1, 2, 2, 2], "feature_col": [3., 3., 3., 3., 7., 7., 7.]})
    got_df, got_error = data_utils.naive_imputation(df, {})
    assert_frame_equal(got_df, want_df)


def test_naively_impute_if_needed():
    # test when no na values are present
    df = pd.DataFrame({"some_col": [1, 2, 3], "label": [1, 1, 1]})
    got_df, got_error = data_utils.naively_impute_if_needed(df, {})
    assert got_error is None
    assert_frame_equal(got_df, df)

    # Test when the amount if missing data is too big
    df["some_col"] = np.nan
    got_df, got_error = data_utils.naively_impute_if_needed(df, {})
    assert got_error == "label_class_mean_cannot_be_calculated_for_row_with_only_NAN"
    assert_frame_equal(got_df, pd.DataFrame())

    # Test when everything is valid (same df as in function above)
    df = pd.DataFrame({"label": [1, 1, 1, 1, 2, 2, 2], "feature_col": [3, 3, np.nan, 3, 7, np.nan, 7]})
    want_df = pd.DataFrame({"label": [1, 1, 1, 1, 2, 2, 2], "feature_col": [3., 3., 3., 3., 7., 7., 7.]})
    got_df, got_error = data_utils.naive_imputation(df, {})
    assert got_error is None
    assert_frame_equal(got_df, want_df)


def test_create_label_remapping():
    # Test if everything works as intended
    got = data_utils.create_label_remapping([20, 25, 31, 15, 27, 18])
    want = {15: 1, 18: 2, 20: 3, 25: 4, 27: 5, 31: 6}
    assert got == want


def test_max_scale_df():
    df = pd.DataFrame({"some_col": [-80, 15, 94]})
    normalization_df = pd.DataFrame({"some_col": [100]})
    got = data_utils.max_scale_df(df, normalization_df)
    want = pd.DataFrame([[-0.8], [0.15], [0.94]])
    assert_frame_equal(got, want)


def test_unexpected_na_values_seen():
    # Test when the data does not exist in featured format
    df = pd.DataFrame({"some_col": [np.nan]})
    got_na_found, got_error = data_utils.unexpected_na_values_seen(df, {})
    assert got_error is None
    assert got_na_found

    # test when the data exists in featured format, but was not reduced
    got_na_found, got_error = data_utils.unexpected_na_values_seen(
        df, {"generate_features": {}})
    assert got_error == "received_featured_but_unreduced_data"
    assert got_na_found

    # test when the data exists in featured format, but only top-level classification was requested
    got_na_found, got_error = data_utils.unexpected_na_values_seen(df, {"generate_features": {}, "reduce_data": {"label_depth": 1}})
    assert got_error is None
    assert got_na_found


def test_rowwise_proportion_of_na_smaller_than_boundary():
    df = pd.DataFrame({"some_col": [1, 2, 3], "another_col": [
                        2, 3, 4], "col_with_nan": [np.nan, 4, 5]})
    got = data_utils.rowwise_proportion_of_na_smaller_than_boundary(df, 0.30).tolist()
    want = [False, True, True]
    assert got == want


def test_create_top_level_df():
    # test if everything works as intended
    df = pd.DataFrame({"some_col": [1, np.nan], "yet_another_col": [2, np.nan], "another_col": [3, np.nan], "label_top": [1, 1], "label_mid": [2, 2], "index_col": [20, 21], "window_uuid": [1, 2]})
    log_file = {"reduce_data": {"needed_columns": [["some_col", "yet_another_col", "another_col"]]}}
    df_top_level_without_subcategories = pd.DataFrame({"some_col": [5, 7], "yet_another_col": [6, 8], "another_col": [7, 9], "another_col_2": [np.nan, np.nan], "label_top": [3, 3], "label_mid": [1, 1], "index_col": [30, 31], "window_uuid": [3, 4]})
    
    got = data_utils.create_top_level_df(df, log_file, df_top_level_without_subcategories)
    want = pd.DataFrame({"some_col": [1., 5, 7], "yet_another_col": [2., 6, 8], "another_col": [3., 7, 9], "label_top": [1, 3, 3], "label_mid": [2, 1, 1], "index_col": [20, 30, 31], "top_level_marker": [True, True, True], "window_uuid": [1., 3., 4.]})
    assert_frame_equal(got, want)


def test_create_mid_level_df():
    # test if everything works as intended
    df = pd.DataFrame({"some_col": [1, np.nan, 1], "yet_another_col": [2, np.nan, 2], "another_col": [np.nan, 3, np.nan], "label_top": [1, 1, 1], "label_mid": [2, 2, 1], "index_col": [0, 1, 2]})
    reduced_copy_df = pd.DataFrame({"some_col": [np.nan], "yet_another_col": [np.nan], "another_col": [3], "label_top": [1], "label_mid": [2], "index_col": [1]})
    log_file = {"reduce_data": {"needed_columns": [[], ["another_col"], []]}}
    top_level_df = pd.DataFrame({"index_col": [0, 2]})
    df_top_level_without_subcategories = pd.DataFrame({"some_col": [5]*3, "yet_another_col": [2]*3, "another_col": [np.nan]*3, "label_top": [3]*3, "label_mid": [1]*3, "index_col": [x for x in range(3)]})
    want = pd.DataFrame({"another_col": [3.], "label_top": [1], "label_mid": [2], "index_col": [1], "top_level_marker": False})
    got = data_utils.create_mid_level_df(df, reduced_copy_df, log_file, top_level_df, df_top_level_without_subcategories)
    assert_frame_equal(got, want)


def test_split_joined_mid_level_df():
    # Test if the label_top col is missing
    got_list, got_error = data_utils.split_joined_mid_level_df(pd.DataFrame(), {})
    assert got_error == "label_top_col_not_found"

    # test if everything works as intended
    df = pd.DataFrame({"another_col": [3., np.nan, 7], "yet_another_col": [np.nan, 5, np.nan], "label_top": [1, 2, 1], "label_mid": [2, 1, 3], "index_col": [1, 14, 24]})
    log_file = {"reduce_data": {"needed_columns": [[], ["another_col"], ["yet_another_col"]]}}
    want_lifting_df = pd.DataFrame({"another_col": [3., 7], "label_top": [1, 1], "label_mid": [2, 3], "index_col": [1, 24]})
    want_walking_df = pd.DataFrame({"yet_another_col": [5.], "label_top": [2], "label_mid": [1], "index_col": [14]})
    got_list, got_error = data_utils.split_joined_mid_level_df(df, log_file)
    assert got_error is None
    assert_frame_equal(got_list[0], want_lifting_df)
    assert_frame_equal(got_list[1], want_walking_df)
    assert len(got_list) == 2


def test_create_list_of_dfs_for_imputation():
    # Test when the data does not exist in featured format
    got_list, got_error = data_utils.create_list_of_dfs_for_imputation(pd.DataFrame(), {})
    assert got_error is None
    assert_frame_equal(got_list[0], pd.DataFrame())

    # Test when the data exists in featured format, but was not reduced
    got_list, got_error = data_utils.create_list_of_dfs_for_imputation(pd.DataFrame(), {"generate_features": [1]})
    assert got_error == "received_unreduced_data"
    assert got_list == []


def test_detect_label_col():
    # Test when the data does not exist in hierarchical featured format
    got_label_name, got_error = data_utils.detect_label_col(
        pd.DataFrame(), {})
    assert got_error is None
    assert got_label_name == "label"

    # Test when the seen columns are a subset of the top_level data
    log_file = {"reduce_data": {"needed_columns": [["some_col"], ["another_col"], ["yet_another_col"]]}, "generate_features": True}
    got_label_name, got_error = data_utils.detect_label_col(
        pd.DataFrame({"some_col": [1, 2], "col_for_internal_processing": [False, True], "label_top": [1, 1]}), log_file)
    assert got_error is None
    assert got_label_name == "label_top"

    # Test when the seen columns are a subset of the mid_level data
    got_label_name, got_error = data_utils.detect_label_col(pd.DataFrame({"another_col": [
                                                            1, 2], "col_for_internal_processing": [False, True], "label_top": [1, 2]}), log_file)
    assert got_error is None
    assert got_label_name == "label_mid"

    got_label_name, got_error = data_utils.detect_label_col(pd.DataFrame({"yet_another_col": [
                                                            1, 2], "col_for_internal_processing": [False, True], "label_top": [1, 1]}), log_file)
    assert got_error is None
    assert got_label_name == "label_mid"

    # Test when the data cannot be matched
    got_label_name, got_error = data_utils.detect_label_col(pd.DataFrame({"unseen_col": [
                                                            1, 2], "col_for_internal_processing": [False, True], "label_top": [1, 1]}), log_file)
    assert got_error == "detection_of_label_col_failed"
    assert not got_label_name


def test_create_top_and_mid_df():
    # Test if everything works as intended
    df = pd.DataFrame({"some_col": [1, np.nan], "yet_another_col": [2, np.nan], "another_col": [
                        np.nan, 3], "another_col_2": [3, np.nan], "label_top": [1, 1], "label_mid": [2, 2], "label": [12, 12], "window_uuid": ["1", "2"]})
    log_file = {"reduce_data": {"needed_columns": [["some_col", "yet_another_col", "another_col_2"], ["another_col"], []]}, "generate_features": True}
    got_top_level_df, got_mid_level_df, got_error = data_utils.create_top_and_mid_df(df, log_file)
    want_top_level_df = pd.DataFrame({"some_col": [1.], "yet_another_col": [2.], "another_col_2": [
                                        3.], "label_top": [1], "label_mid": [2], "index_col": [0], "top_level_marker": [True], "window_uuid":["1"]})
    want_mid_level_df = pd.DataFrame({"another_col": [3.], "label_top": [
                                        1], "label_mid": [2], "index_col": [1], "top_level_marker": [False]})
    assert got_error is None
    assert_frame_equal(got_top_level_df, want_top_level_df)
    assert_frame_equal(got_mid_level_df, want_mid_level_df)


def test_unexpected_na_in_hierarchical_features():
    # Test if everything works as intended when no NA values are in the original df
    df = pd.DataFrame({"some_col": [1, np.nan], "yet_another_col": [2, np.nan], "another_col": [
                        np.nan, 3], "another_col_2": [3, np.nan], "label_top": [1, 1], "label_mid": [2, 2], "label": [12, 12], "window_uuid": ["1", "1"]})
    log_file = {"reduce_data": {"needed_columns": [
        ["some_col", "yet_another_col", "another_col_2"], ["another_col"], []]}, "generate_features": True}
    got_unexpected_na_found, got_error = data_utils.unexpected_na_in_hierarchical_features(df, log_file)
    assert got_error is None
    assert not got_unexpected_na_found

    # Test if everything works as intended when too much NA values are in the original df
    df = pd.DataFrame({"some_col": [np.nan, np.nan], "yet_another_col": [2, np.nan], "another_col": [
                        np.nan, 3], "another_col_2": [3, np.nan], "label_top": [1, 1], "label_mid": [2, 2], "label": [12, 12], "window_uuid": ["1", "1"]})
    got_unexpected_na_found, got_error = data_utils.unexpected_na_in_hierarchical_features(df, log_file)
    assert got_error == "too_much_nan_values_or_naive_imputation"
    assert got_unexpected_na_found

    # Test if everything works as intended when the amount of NA might be solved by resampling
    df = pd.DataFrame({"some_col": [np.nan, np.nan, 1, 2], "yet_another_col": [2, np.nan, 2, 1], "another_col": [
                        np.nan, 3, np.nan, np.nan], "another_col_2": [3, np.nan, 3, 3], "label_top": [1, 1, 1, 1], "label_mid": [2, 2, 2, 2], "label": [12, 12, 12, 12], "window_uuid": ["2", "2", "3", "3"]})
    got_unexpected_na_found, got_error = data_utils.unexpected_na_in_hierarchical_features(df, log_file)
    assert got_error is None
    assert got_unexpected_na_found


def test_join_list_of_sub_dfs():
    # test when one of the elements is not a pd.DataFrame
    got_df, got_error = data_utils.join_list_of_sub_dfs([pd.DataFrame(), 43])
    assert got_error == "got_non_pdDataframe_dt"
    assert_frame_equal(got_df, pd.DataFrame())

    # Test if everything works as intended
    got_df, got_error = data_utils.join_list_of_sub_dfs([pd.DataFrame({"some_col": [1, 2]}), pd.DataFrame({"some_col": [3, 4]})])
    assert got_error is None
    want_df = pd.DataFrame({"some_col": [1, 3, 2, 4], "index": [
                            0, 0, 1, 1]}).set_index("index")
    want_df.index.name = None
    assert_frame_equal(got_df, want_df)


def test_prepare_labels_hierarchical_classification():
    # Test if everything works as intended with data of only one more general class
    df = pd.DataFrame({"label": [11, 12, 15]})
    want_df = pd.DataFrame({"label": [11, 12, 15], "label_top": [
                            1, 1, 1], "label_mid": [1, 2, 3]})
    got_df, got_error = data_utils.prepare_labels_hierarchical_classification(df, 2)
    assert got_error is None
    assert_frame_equal(got_df, want_df,check_dtype=False)

    # Test if everything works as intended with data from several more general classes
    df = pd.DataFrame({"label": [11, 12, 15, 21, 22, 23, 31]})
    want_df = pd.DataFrame({"label": [11, 12, 15, 21, 22, 23, 31], "label_top": [
                            1, 1, 1, 2, 2, 2, 3], "label_mid": [1, 2, 3, 1, 2, 3, 1]})
    got_df, got_error = data_utils.prepare_labels_hierarchical_classification(
        df, 2)
    assert got_error is None
    assert_frame_equal(got_df, want_df,check_dtype=False)


def test_prepare_label_for_featured_data():
    # test when only top-level classification was asked
    df = pd.DataFrame({"label": [11, 12, 15, 21, 22, 23, 31]})
    got_df, got_error = data_utils.prepare_label_for_featured_data(df, 1)
    want_df = pd.DataFrame({"label": [1, 1, 1, 2, 2, 2, 3]})
    assert got_error is None
    assert_frame_equal(got_df, want_df,check_dtype=False)

    # test when mid-level classification was asked
    df = pd.DataFrame({"label": [11, 12, 15, 21, 22, 23, 31]})
    got_df, got_error = data_utils.prepare_label_for_featured_data(df, 2)
    want_df = pd.DataFrame({"label": [11, 12, 15, 21, 22, 23, 31], "label_top": [
                            1, 1, 1, 2, 2, 2, 3], "label_mid": [1, 2, 3, 1, 2, 3, 1]})
    assert got_error is None
    assert_frame_equal(got_df, want_df,check_dtype=False)


def test_z_normalize_df():
    df = pd.DataFrame({"some_col": [1, 2], "another_col": [10, 20]})
    normalization_df = pd.DataFrame({"some_col": [1.5, 1], "another_col": [15, 10]})
    want = pd.DataFrame({"some_col": [-0.5, 0.5], "another_col": [-0.5, 0.5]})
    got = data_utils.z_normalize_df(df, normalization_df)
    assert_frame_equal(got, want)


def test_determine_standardization_type():
    # test when the file contains data for max_scaling
    df = pd.DataFrame(index=["max_abs", "scale"])
    want_scaling = "max_abs"
    got_scaling, got_error = data_utils.determine_standardization_type(df)
    assert got_error is None
    assert got_scaling == want_scaling

    # Test when the file contains data for z_norm
    df = pd.DataFrame(index=["mean", "scale", "var"])
    want_scaling = "normal"
    got_scaling, got_error = data_utils.determine_standardization_type(df)
    assert got_error is None
    assert got_scaling == want_scaling


def test_prepare_standardization_df():
    # Test if everything works as intended for z-norm scaling
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "data_utils_test", "test_prepare_standardization_df")

    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    standardizing_df = pd.DataFrame({"some_col": [1, 1, 1]}, index=["mean", "scale", "var"])

    standardizing_df.to_csv(os.path.join(folder, "test_z_normalize_std.csv"))

    got_standardizing_df, got_standardization_type, got_error = data_utils.prepare_standardization_df(folder, True)
    assert got_error is None
    assert got_standardization_type == "normal"
    assert_frame_equal(got_standardizing_df, standardizing_df.reset_index(drop=True))


def test_standardize():
    # Test if everything works as intended for z-norm
    df = pd.DataFrame([[1, 10], [2, 20]])
    normalization_df = pd.DataFrame([[1.5, 15], [1, 10]])
    want = pd.DataFrame([[-0.5, -0.5], [0.5, 0.5]])
    got = data_utils.standardize(df, "normal", normalization_df)
    assert_frame_equal(got, want)

    # Test if everything works as intended for max_abs
    normalization_df = pd.DataFrame([[10, 20]])
    want = pd.DataFrame([[0.1, 0.5], [0.2, 1.]])

    got = data_utils.standardize(df, "max_abs", normalization_df)
    assert_frame_equal(got, want)


def test_bring_standardizing_df_in_correct_order():
    # Test when lifting_data is being handled
    df = pd.DataFrame({"some_col": [x for x in range(7)]}).T
    want = pd.DataFrame([[0, 4, 1, 2, 3, 5, 6]])
    got_df, got_error = data_utils.bring_standardizing_df_in_correct_order(df, "lifting")
    assert got_error is None
    assert_frame_equal(got_df, want)


def test_prepare_standardization():
    # test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "data_utils_test", "test_prepare_standardization")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    standardizing_df = pd.DataFrame({"some_col": [1, 1, 1]}, index=["mean", "scale", "var"])

    standardizing_df.to_csv(os.path.join(folder, "test_z_normalize_std.csv"))

    got_df, got_standardization_type, got_error = data_utils.prepare_standardization(folder, True, "top")
    assert got_error is None
    assert got_standardization_type == "normal"
    assert_frame_equal(got_df, standardizing_df.reset_index(drop=True))
