import os
import shutil
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from utils import create_windows_utils, data_utils, file_utils
from config.test_config import test_config



def test_prepare_time_col():
    # test if the wrong datatype was given
    got_df, got_error = create_windows_utils.prepare_time_col([], "/somewhere/some_file.csv")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error ==  f"expected_pdDataFrame_got_{type([])}"

    # Test if time not in the df columns
    got_df, got_error = create_windows_utils.prepare_time_col(pd.DataFrame({"some_col": [1, 2, 3]}), "/somewhere/some_file.csv")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error ==  "time_col_missing"

    # Check if a correct error gets thrown, when the time columns contains data which cannot be transformed
    got_df, got_error = create_windows_utils.prepare_time_col(pd.DataFrame({"time": [[1, 15], [1854, 484], [546851, 456]], "some_col": [1, 2, 3]}), "/somewhere/some_file.csv")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == 'got_col_time_of_nonint_dt'

    # Test if the given time col cannot be brought to the same len
    got_df, got_error = create_windows_utils.prepare_time_col(pd.DataFrame({"time": [1669279860, 0, 1669279868], "some_col": [1, 2, 3]}), "/somewhere/some_file.csv")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "cannot_bring_0_to_maxlen"

    # Test if the seen time begins too early
    got_df, got_error = create_windows_utils.prepare_time_col(pd.DataFrame({"time": [1669279860000, 1000000000000, 1669279868000], "some_col": [1, 2, 3]}), "/somewhere/some_file.csv")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "earliest_date_seen_from_year_smaller_2018"

    # Test if the seen time ends too late
    got_df, got_error = create_windows_utils.prepare_time_col(pd.DataFrame({"time": [2041408633505, 2041408633506, 2041408633507], "some_col": [1, 2, 3]}), "/somewhere/some_file.csv")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "latest_date_seen_from_year_greater_2023"

    # Test if the seen time duration is too long
    got_df, got_error = create_windows_utils.prepare_time_col(pd.DataFrame({"time": [1669279860000, 1669309861000, 1669279868000], "some_col": [1, 2, 3]}), "/somewhere/some_file.csv")
    assert_frame_equal(got_df, pd.DataFrame())
    assert got_error == "seen_duration_longer_eight_hours"

    # Test if everything works as intended
    got_df, got_error = create_windows_utils.prepare_time_col(pd.DataFrame({"time": [166927986000, 1669279861000, 1669279865000], "some_col": [1, 2, 3]}), "/somewhere/some_file.csv")
    want_df = pd.DataFrame({"time": [1669279860000, 1669279861000, 1669279865000], "some_col": [1, 2, 3]})
    assert_frame_equal(got_df, want_df)
    assert got_error is None


def test_create_list_of_windows():
    # test if there is no time col in the given df
    df = pd.DataFrame({"another_col": [1669279860000, 1669279860010, 1669279860015], "some_col": [1, 2, 3]})
    got_list, got_error = create_windows_utils.create_list_of_windows(
        df, 1000, 500, "some_file")
    assert not got_list
    assert got_error == "no_time_col_found"

    # test if the window_length is greater than the observed session duration
    df = pd.DataFrame({"time": [1669279860000, 1669279860010, 1669279860015], "some_col": [1, 2, 3]})
    got_list, got_error = create_windows_utils.create_list_of_windows(df, 1000, 500, "some_file")
    assert not got_list
    assert got_error is None

    # test if everything works as intended
    df = pd.DataFrame({"time": [1669279860000, 1669279864000], "some_col": [1, 2]})
    got_list, got_error = create_windows_utils.create_list_of_windows(df, 1000, 1000, "some_file")
    want_list = [[1669279860000, 1669279861000], [1669279861000, 1669279862000], [1669279862000, 1669279863000], [1669279863000, 1669279864000]]
    assert got_list == want_list
    assert got_error is None


def test_handle_window():
    root = test_config['TEST_ROOT']
    valid_df = pd.DataFrame({"time": [x for x in range(1669279860000, 1669279870000, 1000)], "some_col": [x for x in range(0, 10000, 1000)], "label": [11]*10})
    no_time_col_df = pd.DataFrame({"another_col": [x for x in range(1669279860000, 1669279870000, 1000)], "some_col": [x for x in range(0, 10000, 1000)], "label": [11]*10})
    only_nan_df = pd.DataFrame({"time": [x for x in range(1669279860000, 1669279870000, 1000)], "some_col": [np.nan]*10, "label": [11]*10})
    no_label_col_df = pd.DataFrame({"time": [x for x in range(1669279860000, 1669279870000, 1000)], "some_col": [x for x in range(0, 10000, 1000)]})

    data_columns = ["some_col"]
    valid_window = [1669279860000, 1669279861000]

    valid_saving_folder = os.path.join(root, "create_windows_utils_test", "test_handle_windows")

    # Test if the df has the wrong datatype
    got_success, got_windows_saved, got_error = create_windows_utils.handle_window([], valid_window, valid_df, data_columns, True, "some_file.csv", 1000, 10, True, valid_saving_folder, 0)
    assert got_success is None
    assert got_windows_saved == 0
    assert got_error == "given_df_no_pdDataFrame"

    # test if the df_filled has wrong dt
    got_success, got_windows_saved, got_error = create_windows_utils.handle_window(valid_df, valid_window, [], data_columns, True, "some_file.csv", 1000, 10, True, valid_saving_folder, 0)
    assert got_success is None
    assert got_windows_saved == 0
    assert got_error == "given_df_filled_no_pdDataFrame"

    # test if label not in df.columns
    got_success, got_windows_saved, got_error = create_windows_utils.handle_window(
        no_label_col_df, valid_window, valid_df, data_columns, True, "some_file.csv", 1000, 10, True, valid_saving_folder, 0)
    assert got_success is None
    assert got_windows_saved == 0
    assert got_error == "no_label_col_found"

    # test if time not in df.columns
    got_success, got_windows_saved, got_error = create_windows_utils.handle_window(
        no_time_col_df, valid_window, valid_df, data_columns, True, "some_file.csv", 1000, 10, True, valid_saving_folder, 0)
    assert got_success is None
    assert got_windows_saved == 0
    assert got_error == "no_time_col_found"
    
    # Test if the window is invalid
    got_success, got_windows_saved, got_error = create_windows_utils.handle_window(no_time_col_df, ["sdf", "fdh"], valid_df, data_columns, True, "some_file.csv", 1000, 10, True, valid_saving_folder, 0)
    assert got_success is None
    assert got_windows_saved == 0
    assert got_error == "no_time_col_found"

    # test if the window only contains filled values
    got_success, got_windows_saved, got_error = create_windows_utils.handle_window(only_nan_df, valid_window,  valid_df, data_columns, True, "some_file.csv", 1000, 10, True, valid_saving_folder, 0)
    assert not got_success
    assert got_windows_saved == 0
    assert got_error is None

    # Test if the correct result gets returned without saving and without flatten
    got_success, got_windows_saved, got_error = create_windows_utils.handle_window(valid_df, valid_window, valid_df, data_columns, False, "some_file.csv", 1000, 10, False, valid_saving_folder, 0)
    assert got_success
    assert got_windows_saved == 0
    assert got_error is None

    # Test if the correct result gets returned with saving and without flatten
    if os.path.exists(valid_saving_folder):
        shutil.rmtree(valid_saving_folder)
    got_success, got_windows_saved, got_error = create_windows_utils.handle_window(
        valid_df, valid_window, valid_df, data_columns, False, "some_file.csv", 1000, 10, True, valid_saving_folder, 0)
    assert got_success
    assert got_windows_saved == 1
    assert got_error is None
    assert os.path.exists(os.path.join(valid_saving_folder, "some_file_1.csv"))
    saved_df = pd.read_csv(os.path.join(valid_saving_folder, "some_file_1.csv"))
    want_df_not_flattened = pd.DataFrame({"time": [1669279860000, 1669279861000], "some_col": [0, 1000], "label": [11, 11]})
    assert_frame_equal(saved_df, want_df_not_flattened)

    # Test if the wrong amount of columns were given
    if os.path.exists(valid_saving_folder):
        shutil.rmtree(valid_saving_folder)
    got_success, got_windows_saved, got_error = create_windows_utils.handle_window(
        valid_df, valid_window, valid_df, data_columns, True, "some_file.csv", 1000, 10, True, valid_saving_folder, 0)
    assert got_success is None
    assert got_error == "wrong_col_amount_expected_11_got_3"
    assert not got_windows_saved

    # Test if everything works as intended with flattening and saving
    valid_df = pd.DataFrame({"time": [11, 12]*50, "label": [10, 10]*50, "some_col_a": [12, 13]*50, "some_col_b": [12, 13]*50, "some_col_c": [12, 13]*50, "some_col_d": [12, 13]*50,
                            "some_col_e": [12, 13]*50, "some_col_f": [12, 13]*50, "some_col_g": [12, 13]*50, "some_col_h": [12, 13]*50, "some_col_i": [12, 13]*50})

    # Create the resulting wanted df
    data_columns = [x for x in valid_df.columns]
    for x in ["label", "time"]:
        data_columns.remove(x)

    entries = [12]*9
    entries.extend([13]*9)

    flattened_colnames = []
    for iterations in range(len(valid_df)):
        for colname in valid_df.columns:
            if colname == "label" or colname == "time":
                continue
            flattened_colnames.append(f"{colname}_{str(iterations)}")
    want_df = pd.DataFrame({"0": entries*50}).T
    want_df.index = pd.RangeIndex(start=0, stop=1, step=1)
    want_df.columns = flattened_colnames
    want_df["label"] = 10

    if os.path.exists(valid_saving_folder):
        shutil.rmtree(valid_saving_folder)

    got_success, got_windows_saved, got_error = create_windows_utils.handle_window(
        valid_df, [10, 13], valid_df, data_columns, True, "some_file.csv", 1000, 10, True, valid_saving_folder, 0)

    assert got_error is None
    assert got_success
    assert got_windows_saved == 1

    # test if the saving was done correctly and the file contains the correct df
    assert os.path.isfile(os.path.join(valid_saving_folder, "some_file_1.csv"))
    saved_df = pd.read_csv(os.path.join(valid_saving_folder, "some_file_1.csv"))
    assert_frame_equal(saved_df, want_df)


def test_create_windows_for_file():
    root = test_config['TEST_ROOT']
    # Test if df does not contain the needed columns
    foldername = os.path.join(root, "create_windows_utils_test", "test_create_windows_for_file")

    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)
    path_to_file = os.path.join(foldername, "some_file.csv")
    file_with_attributes = {"file": path_to_file,
                            "needed_columns": ["i_am_needed", "another_col"]}

    # Test if the preparation of the time col fails
    df = pd.DataFrame({"time": [1001408633], "some_col": [12]})
    file_with_attributes["needed_columns"] = ["time", "some_col"]
    df.to_csv(path_to_file, index=False)
    got = create_windows_utils.create_windows_for_file(file_with_attributes)
    want = data_utils.create_result_string(path_to_file, [1], 0, "earliest_date_seen_from_year_smaller_2018")
    assert got == want

    # Test if an unknown filling method is given
    df = pd.DataFrame({"time": [1669305215000, 1669307215000], "some_col": [12, 11]})
    df.to_csv(path_to_file, index=False)
    file_with_attributes["filling_method"] = "unknown_filling_method"
    file_with_attributes["window_length"] = 1000
    file_with_attributes["step_length"] = 500
    got = create_windows_utils.create_windows_for_file(file_with_attributes)
    want = data_utils.create_result_string(path_to_file, [1], 0, "invalid_filling_method_unknown_filling_method_given")
    assert got == want

    # Test if everything works as intended
    df = pd.DataFrame({"time": [1669305215000, 1669305216000], "some_col": [12, 11], "label": [11, 11]})
    df.to_csv(path_to_file, index=False)
    file_with_attributes["needed_columns"] = ["time", "some_col", "label"]
    file_with_attributes["filling_method"] = "ffill"
    file_with_attributes["data_columns"] = ["some_col"]
    file_with_attributes["flatten"] = False
    file_with_attributes["resampling_rate"] = 10
    file_with_attributes["store_local"] = True
    file_with_attributes["saving_folder"] = foldername
    file_with_attributes["windows_saved_current_file"] = 0
    want = data_utils.create_result_string(path_to_file, [], 1, None)
    got = create_windows_utils.create_windows_for_file(
        file_with_attributes)
    assert got == want


def test_valid_attributes():
    labeled_files_valid = ["/somewhere/somefile.csv"]
    labeled_files_invalid = [5]
    needed_columns_valid = ["time", "label", "some_col"]
    needed_columns_empty = []
    window_length_valid = 1000
    window_length_invalid = -1000
    step_length_valid = 100
    window_length_invalid = -100
    filling_method_valid = "ffill"
    filling_method_invalid = "not_valid"
    resampling_rate_valid = 10
    data_columns_valid = ["some_col"]
    flatten_valid = True
    store_local_valid = True
    root = test_config['TEST_ROOT']
    saving_folder_valid = os.path.join(root, "create_windows_utils_test", "test_valid_attributes")

    # test wrong datatype labeled_files list
    got = create_windows_utils.valid_attributes(5, needed_columns_valid, window_length_valid, step_length_valid, filling_method_valid,
                                                resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "given_labeled_files_not_a_list"

    # test non str in labeled_files list
    got = create_windows_utils.valid_attributes(labeled_files_invalid, needed_columns_valid, window_length_valid, step_length_valid,
                                                filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "got_file_of_nonstr_type"

    # test empty labeled_files
    got = create_windows_utils.valid_attributes(needed_columns_empty, needed_columns_valid, window_length_valid, step_length_valid,
                                                filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "labeled_files_empty"

    # test wrong datatype needed columns
    got = create_windows_utils.valid_attributes(labeled_files_valid, 5, window_length_valid, step_length_valid, filling_method_valid,
                                                resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "needed_columns_not_a_list"

    # test emty needed columns
    got = create_windows_utils.valid_attributes(labeled_files_valid, [], window_length_valid, step_length_valid, filling_method_valid,
                                                resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "needed_columns_empty"

    # test wrong window_length
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, [], step_length_valid, filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "non_int_window_length"

    # test too small window Length
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_invalid, step_length_valid,
                                                filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "window_length_too_small"

    # test too small step length
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_invalid,
                                                filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "step_length_too_small"

    # Test step length > window_length
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid + 200, filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "step_length_bigger_window_length"

    # test invalid filling method
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid,
                                                filling_method_invalid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "filling_method_not_implemented"

    # wrong dtype resampling_rate
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid, filling_method_valid, [resampling_rate_valid], data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "got_resampling_rate_of_nonint_type"

    # Too small resampling_rate
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid,
                                                filling_method_valid, resampling_rate_valid-20, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "resampling_rate_too_small"

    # Too big resampling_rate
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid,
                                                filling_method_valid, resampling_rate_valid*1000, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "resampling_rate_too_big"

    # data columns is not subset of needed cols
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid, filling_method_valid, resampling_rate_valid, [
                                                "another_col"], flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got == "data_col_not_subset_of_needed_cols"

    # wrong datatype for flatten
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid,
                                                filling_method_valid, resampling_rate_valid, data_columns_valid, [flatten_valid], store_local_valid, saving_folder_valid, None)
    assert got == "got_flatten_of_nonbool_type"

    # wrong datatype for store_local
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid,
                                                filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, [store_local_valid], saving_folder_valid, None)
    assert got == "got_store_local_of_nonbool_type"

    # wrong datatype saving_folder
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid,
                                                filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, [saving_folder_valid], None)
    assert got == "got_saving_folder_of_nonstr_type"

    # test correct creation of saving folder
    if os.path.exists(saving_folder_valid):
        shutil.rmtree(saving_folder_valid)

    # test if it correctly return True when everything is valid
    got = create_windows_utils.valid_attributes(labeled_files_valid, needed_columns_valid, window_length_valid, window_length_valid, filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None)
    assert got is None
    assert os.path.exists(saving_folder_valid)

def test_create_list_of_file_dicts():
    labeled_files_valid = ["/somewhere/somefile.csv", "/somewhere/another_file.csv"]
    needed_columns_valid = ["time", "label", "some_col"]
    window_length_valid = 1000
    window_length_invalid = -1000
    step_length_valid = 100
    filling_method_valid = "ffill"
    resampling_rate_valid = 10
    data_columns_valid = ["some_col"]
    flatten_valid = True
    store_local_valid = True
    root = test_config['TEST_ROOT']
    saving_folder_valid = os.path.join(root, "create_windows_utils_test", "test_valid_attributes")
    normalization_df = pd.DataFrame()

    # test if an empty List gets returned when at least one of the arguments is incorrect
    got_list, got_error = create_windows_utils.create_list_of_file_dicts(labeled_files_valid, needed_columns_valid,  [
                                                                            window_length_invalid], step_length_valid, filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None, normalization_df)
    assert not got_list
    assert got_error == "non_int_window_length"

    # test if everything works as intended when all the parameters are valid
    got_list, got_error = create_windows_utils.create_list_of_file_dicts(labeled_files_valid, needed_columns_valid, window_length_valid, step_length_valid,
                                                                            filling_method_valid, resampling_rate_valid, data_columns_valid, flatten_valid, store_local_valid, saving_folder_valid, None, normalization_df)
    want = [
        {"file": labeled_files_valid[0],
            "needed_columns":needed_columns_valid,
            "window_length":window_length_valid,
            "step_length":step_length_valid,
            "filling_method": filling_method_valid,
            "resampling_rate": resampling_rate_valid,
            "data_columns":data_columns_valid,
            "flatten":flatten_valid,
            "store_local":store_local_valid,
            "saving_folder":saving_folder_valid,
            "normalize":None,
            "normalization_df": normalization_df},
        {"file": labeled_files_valid[1],
            "needed_columns":needed_columns_valid,
            "window_length":window_length_valid,
            "step_length":step_length_valid,
            "filling_method": filling_method_valid,
            "resampling_rate": resampling_rate_valid,
            "data_columns":data_columns_valid,
            "flatten":flatten_valid,
            "store_local":store_local_valid,
            "saving_folder":saving_folder_valid,
            "normalize":None,
            "normalization_df":normalization_df},
    ]
    assert got_list == want
    assert got_error is None

def test_valid_table():
    root = test_config['TEST_ROOT']
    existing_datapath = os.path.join(root, "create_windows_utils_test", "test_valid_table")
    if not os.path.exists(existing_datapath):
        os.makedirs(existing_datapath)

    # test when the datapath does not exist
    non_existing_folder = "/some_non_existing_folder/non_existing_subfolder"
    if os.path.exists(non_existing_folder):
        os.remove(non_existing_folder)

    table = {"data_path": non_existing_folder}
    assert create_windows_utils.valid_table(table) ==  "dp_does_not_exist"

    # test if the labeled_data_folder is missing
    if os.path.exists(existing_datapath):
        shutil.rmtree(existing_datapath)
        os.makedirs(existing_datapath)
    table = {"data_path": existing_datapath}

    valid_anonymize_file = {"create_windows": {"needed_columns": [], "data_columns": []},
                            "svm_features": {"top": [], "lifting": [], "walking": []},
                            "feature_utils": {"accel_cols": [], "differences_cols": [], "needed_cols": [], "on_single_column": [], "magn_feat_name": []}}
    error = file_utils.save_json(valid_anonymize_file, os.path.join(existing_datapath, "anonymization_file.json"))
    assert error is None

    assert create_windows_utils.valid_table(table) ==  "dir_labeled_data_not_subfolder_of_dp"

    # test if an invalid value for store_local was given
    os.makedirs(os.path.join(existing_datapath, "labeled_data"))
    table = {"data_path": existing_datapath, "dryrun": 15}
    assert create_windows_utils.valid_table(table) == "got_dryrun_of_nonbool_datatype"

    # Test if the given window_length or step_length are nor valid
    table = {"data_path": existing_datapath, "dryrun": True, "window_length": -20, "step_length": 200000, }
    assert create_windows_utils.valid_table(table) == "given_window_length_smaller_1"

    # Test if the given resampling_rate is not valid
    table = {"data_path": existing_datapath, "dryrun": True, "window_length": 1000,
                "step_length": 500, "resampling_rate": "wrong_datatype"}
    assert create_windows_utils.valid_table(table) == "got_resampling_rate_of_nonint_dataytpe"

    # Test if the given resampling_rate is not valid
    table = {"data_path": existing_datapath, "dryrun": True, "window_length": 1000, "step_length": 500, "resampling_rate": -200}
    assert create_windows_utils.valid_table(table) == "resampling_rate_too_small"

    # Test if the given flatten value is not valid
    table = {"data_path": existing_datapath, "dryrun": True, "window_length": 1000, "step_length": 500, "resampling_rate": 10, "flatten": "not_a_bool"}
    assert create_windows_utils.valid_table(table) == "got_flatten_of_nonbool_type"

    # Test if the given date is not in the correct format
    table = {"data_path": existing_datapath, "dryrun": True, "window_length": 1000,
                "step_length": 500, "resampling_rate": 10, "flatten": True, "output_date": "wrong_date_format"}
    assert create_windows_utils.valid_table(table) == "got_time_str_of_invalid_len"

    # Test if there are no files in the given folder
    table = {"data_path": existing_datapath, "dryrun": True, "window_length": 1000,
                "step_length": 500, "resampling_rate": 10, "flatten": True}
    assert create_windows_utils.valid_table(table) == "no_data_in_dir"

    # test when everything is valid
    # Create a file in the labeled_data subfolder
    df = pd.DataFrame({"some_col": [1, 2, 3]})
    df.to_csv(os.path.join(existing_datapath, "labeled_data", "somefile.csv"), index=False)

    table = {"data_path": existing_datapath, "dryrun": True, "window_length": 1000,
                "step_length": 500, "resampling_rate": 10, "flatten": True}
    assert create_windows_utils.valid_table(table) is None


def test_return_normalization_df():
    # test if everything works as intended for z_normalization
    anonymize_file = {"create_windows": {"data_columns": ["some_col", "another_col"],
                                            "z_normalize_mean": [1, 4],
                                            "z_normalize_std": [2, 3]}}
    got_df, got_error = create_windows_utils.return_normalization_df("z_normalize", anonymize_file)
    assert got_error is None
    want_df = pd.DataFrame({"some_col": [1., 2.], "another_col": [4., 3.]})
    assert_frame_equal(got_df, want_df, check_dtype=False)
