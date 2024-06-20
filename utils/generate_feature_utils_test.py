import os
import sys
import shutil
import pandas as pd
from pandas.testing import assert_frame_equal

from utils import generate_feature_utils, file_utils, data_utils
from config.test_config import test_config

def test_valid_table():
    root = test_config['TEST_ROOT']
    existing_datapath = os.path.join(root, "generate_feature_utils_test", "test_valid_table")
    if os.path.exists(existing_datapath):
        shutil.rmtree(existing_datapath)
    os.makedirs(existing_datapath)

    # test when the datapath does not exist
    non_existing_folder = os.path.join(existing_datapath, "non_existing_subfolder")
    if os.path.exists(non_existing_folder):
        shutil.rmtree(non_existing_folder)
    table = {"data_path": non_existing_folder}
    assert generate_feature_utils.valid_table(table) == "dp_does_not_exist"

    valid_anonymize_file = {"create_windows": {"needed_columns": [], "data_columns": []},
                            "svm_features": {"top": [], "lifting": [], "walking": []},
                            "feature_utils": {"accel_cols": [], "differences_cols": [], "needed_cols": [], "on_single_column": [], "magn_feat_name": []}}
    error = file_utils.save_json(valid_anonymize_file, os.path.join(existing_datapath, "anonymization_file.json"))
    assert error is None

    # test if an invalid value for dryrun was given
    os.makedirs(os.path.join(existing_datapath, "labeled_data"))
    table = {"data_path": existing_datapath, "dryrun": [15]}
    assert generate_feature_utils.valid_table(table) == "got_dryrun_of_nonbool_datatype"

    # Test if the given input_date is not in the correct format
    table = {"data_path": existing_datapath, "dryrun": True, "input_date": "wrong_date_format"}
    assert generate_feature_utils.valid_table(table) == "got_input_date_of_invalid_format"

    # Test if there are no files in the given folder
    valid_date = "2022-01-01T12:00"
    full_input_folder = os.path.join(existing_datapath, "windows", valid_date)
    if os.path.exists(full_input_folder):
        shutil.rmtree(full_input_folder)
    os.makedirs(full_input_folder)
    table = {"data_path": existing_datapath, "dryrun": True, "input_date": valid_date}
    assert generate_feature_utils.valid_table(table) == "no_data_in_dir"

    # Test when the parameters of the found log file are not identical to the currently chosen params
    df = pd.DataFrame({"some_col": [1, 2, 3]})
    df.to_csv(os.path.join(full_input_folder, "somefile.csv"), index=False)

    table = {"data_path": existing_datapath, "dryrun": True, "input_date": valid_date}
    log_file = {"create_windows": {"flatten": True}}
    file_utils.save_json(log_file, os.path.join(full_input_folder, "log.json"))
    assert generate_feature_utils.valid_table(table) == "generating_features_for_flattened_data_not_allowed"

    # Test when the parameters of the found log file are not identical to the currently chosen params
    table = {"data_path": existing_datapath, "dryrun": True, "input_date": valid_date}
    log_file = {"create_windows": {"flatten": False, "method": "linear"}}
    file_utils.save_json(log_file, os.path.join(full_input_folder, "log.json"))
    assert generate_feature_utils.valid_table(table) == "generating_features_for_interpolated_data_not_allowed"

    # test when the given output_dat has a wrong format
    log_file = {"create_windows": {"flatten": False, "method": "ffill"}}
    file_utils.save_json(log_file, os.path.join(full_input_folder, "log.json"))

    table = {"data_path": existing_datapath, "dryrun": True, "input_date": valid_date, "output_date": valid_date + valid_date}
    assert generate_feature_utils.valid_table(table) == "got_time_str_of_invalid_len"

    # test when everything is valid
    table = {"data_path": existing_datapath, "dryrun": True, "input_date": valid_date, "bundle_sessions": True}
    assert generate_feature_utils.valid_table(table) is None


def test_generate_features_for_file():
    # test when there are no files in the input folder
    root = test_config['TEST_ROOT']
    existing_datapath = os.path.join(root, "generate_feature_utils_test", "test_generate_features_for_file")
    if os.path.exists(existing_datapath):
        shutil.rmtree(existing_datapath)

    file_folder = os.path.join(existing_datapath, "windows", "2022-01-01T12-00")
    os.makedirs(file_folder)
    full_path_to_file = os.path.join(file_folder, "some_file_session_1(1)_1.csv")

    valid_anonymize_file = {"create_windows": {"needed_columns": [], "data_columns": []},
                            "svm_features": {"top": [], "lifting": [], "walking": []},
                            "feature_utils": {"accel_cols": ["column_a", "column_b", "column_c", "column_k"], "differences_cols": ["column_d", "column_e", "column_j"],
                                              "needed_cols": ["column_a", "column_b", "column_c", "column_d", "column_e", "column_f", "column_g", "column_h", "column_i", "time"],
                                              "on_single_column": ["mean", "median"],
                                              "magn_feat_name": ["some_colname"]}}

    file_with_attributes = {"data_path": existing_datapath, "file": full_path_to_file, "input_date": "2022-01-01T12-00", "store_local": True, "saving_time": "2022-01-01T12-00",
                            "feature_functions_dict": {"on_single_column": ["mean", "median"], "on_multiple_columns": []}, "anonymize_file": valid_anonymize_file}

    got = generate_feature_utils.generate_features_for_file(file_with_attributes)
    assert got == data_utils.create_result_string(full_path_to_file, [1], 0, "file_does_not_exist")

    # test when the given df cannot be used to generate_features

    # No label col in df
    df = pd.DataFrame({"some_col": [1, 2, 3]})
    df.to_csv(os.path.join(file_folder, "some_file_session_1(1)_1.csv"))
    got = generate_feature_utils.generate_features_for_file(file_with_attributes)
    assert got == data_utils.create_result_string(full_path_to_file, [1], 0, "no_label_col_found")

    # df too small
    df = pd.DataFrame({"some_col": [1], "label": [11]})
    df.to_csv(os.path.join(file_folder, "some_file_session_1(1)_1.csv"))
    got = generate_feature_utils.generate_features_for_file(file_with_attributes)
    assert got == data_utils.create_result_string(full_path_to_file, [1], 0, "received_df_of_too_small_size")

    # no time_col in df
    df = pd.DataFrame({"some_col": [1, 2, 3], "label": [11, 11, 11]})
    df.to_csv(os.path.join(file_folder, "some_file_session_1(1)_1.csv"))
    got = generate_feature_utils.generate_features_for_file(file_with_attributes)
    assert got == data_utils.create_result_string(full_path_to_file, [1], 0, "time_col_not_found")

    # time_col does not contain ints
    df = pd.DataFrame({"some_col": [1, 2, 3], "label": [11, 11, 11], "time": ["morning", "afternoon", "evening"]})
    df.to_csv(os.path.join(file_folder, "some_file_session_1(1)_1.csv"))
    got = generate_feature_utils.generate_features_for_file(file_with_attributes)
    assert got == data_utils.create_result_string(full_path_to_file, [1], 0, "time_col_has_non_int_entries")

    # test when the df does not contain all for expanding the df necessary columns
    df = pd.DataFrame({"some_col": [1, 2, 3], "label": [11, 11, 11], "time": [100, 105, 108]})
    df.to_csv(os.path.join(file_folder, "some_file_session_1(1)_1.csv"))
    got = generate_feature_utils.generate_features_for_file(file_with_attributes)
    assert got == data_utils.create_result_string(full_path_to_file, [1], 0, "needed_cols_missing")

    # test if everything works as intended
    correct_df = pd.DataFrame({"column_a": [1, 1], "column_b": [1, 1], "column_c": [1, 1], "column_d": [50, 0],
                               "column_e": [0, 50], "time": [1000, 1500], "column_f": [70, 75],
                               "column_g": [10, 15], "column_h": [0, 2], "column_i": [0, 3], "label": [11, 11]})

    file_path = os.path.join( file_folder, "some_file_session_1(1)_1.csv").replace("\\","/")
    correct_df.to_csv(file_path, index=False)
    got = generate_feature_utils.generate_features_for_file(file_with_attributes)
    want = data_utils.create_result_string(full_path_to_file, [None], 1, "None")
    if "win" in sys.platform:
        want = want.replace("\\","/")
    assert got == want

    # Test if the file exists
    resulting_filename = os.path.join(file_folder, "some_file_session_1(1)_1.csv").replace("windows", "features")
    assert os.path.exists(resulting_filename)
    
    # Check if the df has the correct dimensionality
    df_read_in = pd.read_csv(resulting_filename)
    assert df_read_in.shape == (1, 48)


def test_save_generated_features():
    valid_df = pd.DataFrame({"some_col": [1, 2, 3]})
    malformed_table = {"store_local": True,
                       "saving_time": "2022-01-01T12-00"}

    # test when needed entries are missing from the table
    got_result_string, got_error, got_handled_windows = generate_feature_utils.save_generated_features(valid_df, "somefile(1)_1.csv", malformed_table, 52)
    assert got_error == "not_all_needed_keys_in_table_for_saving"
    assert got_handled_windows == 52
    want = data_utils.create_result_string("somefile(1)_1.csv", [1], 52, "not_all_needed_keys_in_table_for_saving")
    assert got_result_string == want


def test_read_prepare_and_ffill_file():
    # test if the given file does not exist
    root = test_config['TEST_ROOT']
    existing_datapath = os.path.join(root, "generate_feature_utils_test", "test_read_prepare_and_ffill_df")
    if os.path.exists(existing_datapath):
        shutil.rmtree(existing_datapath)
    os.makedirs(existing_datapath)
    non_existing_file = os.path.join(existing_datapath, "non_existing.csv")
    got_df, got_result_string, got_label, got_window_uuids = generate_feature_utils.read_prepare_and_ffill_file(non_existing_file, 52)
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_result_string == data_utils.create_result_string(non_existing_file, [1], 52, "file_does_not_exist")
    assert got_label == 0

    # Test if the label col is not in the df
    valid_df = pd.DataFrame({"some_col": [1, 2, 3]})
    path_to_file = os.path.join(existing_datapath, "somefile.csv")
    valid_df.to_csv(path_to_file, index=False)
    got_df, got_result_string, got_label, got_window_uuids = generate_feature_utils.read_prepare_and_ffill_file(path_to_file, 52)
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_result_string == data_utils.create_result_string(path_to_file, [1], 52, "no_label_col_found")
    assert got_label == 0

    # test if the df is too small
    valid_df = pd.DataFrame({"label": [1]})
    path_to_file = os.path.join(existing_datapath, "somefile.csv")
    valid_df.to_csv(path_to_file, index=False)
    got_df, got_result_string, got_label, got_window_uuids = generate_feature_utils.read_prepare_and_ffill_file(path_to_file, 52)
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_result_string ==  data_utils.create_result_string(path_to_file, [1], 52, "received_df_of_too_small_size")
    assert got_label == 0

    # test if everything works as intended
    valid_df = pd.DataFrame({"label": [1, 1, 1], "time": [100, 102, 104]})
    path_to_file = os.path.join(existing_datapath, "somefile.csv")
    valid_df.to_csv(path_to_file, index=False)
    got_df, got_result_string, got_label, got_window_uuids = generate_feature_utils.read_prepare_and_ffill_file(path_to_file, 52)
    assert got_result_string is None
    assert got_label == 1
    assert_frame_equal(got_df, pd.DataFrame({"time": [102]}))


def test_create_list_of_file_dicts():
    root = test_config['TEST_ROOT']
    existing_datapath = os.path.join(root, "generate_feature_utils_test", "test_create_list_of_file_dicts")
    if os.path.exists(existing_datapath):
        shutil.rmtree(existing_datapath)
    os.makedirs(existing_datapath)

    valid_anonymize_file = {"create_windows": {"needed_columns": [], "data_columns": []},
                            "svm_features": {"top": [], "lifting": [], "walking": []},
                            "feature_utils": {"accel_cols": [], "differences_cols": [], "needed_cols": [], "on_single_column": [], "magn_feat_name": []}}
    error = file_utils.save_json(valid_anonymize_file, os.path.join(existing_datapath, "anonymization_file.json"))
    assert error is None

    df = pd.DataFrame({"some_col": [1, 2, 3]})
    file_dir = os.path.join(existing_datapath, "windows", "2022-01-01T12:00")
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    df.to_csv(os.path.join(file_dir, "somefile.csv"))

    # Test if everything works as intended
    table = {"data_path": existing_datapath, "input_date": "2022-01-01T12:00",
             "saving_time": "2022-01-01T12:00", "dryrun": True}
    all_window_files = ["/somewhere/some_session(1)_2"]
    want = [{
        "file": "/somewhere/some_session(1)_2",
        "data_path": existing_datapath,
        "input_date": "2022-01-01T12:00",
        "saving_time": "2022-01-01T12:00",
        "store_local": False,
        "feature_functions_dict": {},
        "anonymize_file": valid_anonymize_file}]
    got_list, got_error = generate_feature_utils.create_list_of_file_dicts(all_window_files, table, {}, valid_anonymize_file)
    assert got_error is None
    assert want == got_list
