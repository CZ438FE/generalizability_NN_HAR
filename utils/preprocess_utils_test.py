import os
import sys
import shutil
import json
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from utils import preprocess_utils, data_utils, file_utils
from config.test_config import test_config


def test_appropriate_label_depth():
    # Test if an invalid granularity was given:
    invalid_granularity = "invalid"
    got_label_depth, got_error = preprocess_utils.appropriate_label_depth(invalid_granularity)
    assert got_error == "invalid_granularity_given"
    assert got_label_depth == 0

    # test if it works correctly given a correct input
    for number, valid_granularity in enumerate(['top', 'mid'], start=1):
        assert preprocess_utils.appropriate_label_depth(valid_granularity)[0] == number


def test_return_svm_features():
    # Test if an invalid depth was given
    invalid_depth = 541651
    got_list, got_error = preprocess_utils.return_svm_features(invalid_depth, "")
    assert not got_list
    assert got_error == "invalid_label_depth_given"

    # test if the input is valid for a label_depth of one
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_return_svm_features")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    result = file_utils.save_json({"svm_features": {"top": ["col_1", "col_2"], "lifting": [
                                    "col_2", "col_3"], "walking": ["col_1", "col_4"]}}, os.path.join(folder, "anonymization_file.json"))
    assert result is None

    got_list, got_error = preprocess_utils.return_svm_features(1, folder)
    assert got_error is None
    assert got_list == [["col_1", "col_2"]]

    # test if the input is valid for a label_depth of 2
    got_list, got_error = preprocess_utils.return_svm_features(2, folder)
    assert got_error is None
    assert got_list == [["col_1", "col_2"], ["col_2", "col_3"], ["col_1", "col_4"]]


def test_update_label_distributions():
    # test if a malformed df is given
    malformed_df = pd.DataFrame()
    valid_label_depth = 1
    valid_label_dict = {}
    got_dict, got_error = preprocess_utils.update_label_distributions(valid_label_dict, malformed_df,  valid_label_depth)
    assert not got_dict

    # test if the correct result gets returned when all parameters are valid
    valid_df = pd.DataFrame({"label": [21]})
    want = {"2": 1}
    got_dict, got_error = preprocess_utils.update_label_distributions(valid_label_dict, valid_df,  valid_label_depth)
    assert got_dict == want


def test_update_classes_files():
    # test if a malformed df is given
    valid_file = "somewhere/some_file.csv"
    malformed_df = pd.DataFrame()
    valid_label_depth = 1
    valid_classes_to_files_dict = {"1": ["somewhere/one_file.csv"], "2": ["somewhere/another_file.csv"]}
    got_dict, got_error = preprocess_utils.update_classes_files(valid_classes_to_files_dict, malformed_df,  valid_label_depth, valid_file)
    assert got_error == "label_col_missing"

    # Test if everything gets executed correctly when a valid_df is given
    valid_df = pd.DataFrame({"label": [12, 12, 12, 12, 12]})
    got_dict, got_error = preprocess_utils.update_classes_files(valid_classes_to_files_dict, valid_df,  valid_label_depth, valid_file)
    want = {"1": ["somewhere/one_file.csv", "somewhere/some_file.csv"],
            "2": ["somewhere/another_file.csv"]}
    assert got_dict == want


def test_create_labels_overview_dict():
    file_with_params = {"label_depth": 4.2,
                        "file": "somewhere/somefile.csv"}
    # test if a wrong label_depth was given
    got = preprocess_utils.create_labels_overview_dict(file_with_params)
    assert got == "invalid_label_depth_given"

    # test if no files were given
    file_with_params = {"label_depth": 2}
    got = preprocess_utils.create_labels_overview_dict(file_with_params)
    assert got == "no_file_given"

    # test if everything_works_as intended
    df = pd.DataFrame({"label": [11]})

    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_create_labels_overview_dict")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    file_path = os.path.join(folder, "example.csv")
    df.to_csv(file_path, index=False)
    want = {"classes_count": {"1": 1},
            "classes_files": {"1": [file_path]}}
    file_with_params = {"label_depth": 1, "file": file_path}
    got = preprocess_utils.create_labels_overview_dict(file_with_params)
    os.remove(file_path)
    assert got == want


def test_calculate_needed_extra_samples():
    # test if an empty dict gets returned when the dict with the labels is empty
    got = preprocess_utils.calculate_needed_extra_samples({}, "balancing_under", False)
    assert got == {}
    # test if everything works with valid input
    valid_labels_dict = {"classes_count": {"1": 100, "2": 150, "3": 120, "4": 100}}
    want = {"1": 50, "2": 0, "3": 30, "4": 50}
    got = preprocess_utils.calculate_needed_extra_samples(valid_labels_dict, "balancing_over", False)
    assert got == want


def test_find_reference_class():
    # test if no labels_overview_dict is given
    got = preprocess_utils.find_reference_class({}, "balancing_over", False)
    assert got == []
    # test if correctly finds the reference class, when oversampling is chosen
    valid_labels_dict = {"classes_count": {
        "1": 100, "2": 150, "3": 120, "4": 100}}
    got = preprocess_utils.find_reference_class(valid_labels_dict, "balancing_over", False)
    want = ["2", 150]
    assert got == want
    # test if correctly finds the reference class, when undersampling is chosen
    got = preprocess_utils.find_reference_class(valid_labels_dict, "balancing_under", False)
    want = ["1", 100]
    assert got == want


def test_calculate_differences():
    # Test if no ref_class is given
    valid_labels_dict = {"classes_count": {"1": 100, "2": 150, "3": 120, "4": 100}}
    valid_reference_class = ["2", 150]
    got = preprocess_utils.calculate_differences([], valid_labels_dict)
    assert got == {}
    # Test if the dict containing the labels is empty
    got = preprocess_utils.calculate_differences(valid_reference_class, {})
    assert got == {}
    # Test if everything gets evaluated correctly when everything is correct
    got = preprocess_utils.calculate_differences(valid_reference_class, valid_labels_dict)
    want = {"1": 50, "2": 0, "3": 30, "4": 50}
    assert got == want


def test_adjust_granularity_of_label():
    # Test if everything gets downsized correctly
    colname = "some_col"
    label_depth = 1
    df = pd.DataFrame({colname: [15, 15, 15]})
    want_df = pd.DataFrame({colname: [1, 1, 1]}).astype(np.int64)
    got_df, got_error = preprocess_utils.adjust_granularity_of_label(df, colname, label_depth)
    assert_frame_equal(got_df, want_df)
    assert got_error is None


def test_calculate_prop_original_data_dict():
    # test if an empty dict gets returned when the input is malformed
    invalid_needed_samples = pd.DataFrame()
    labels_overview_dict = []
    got = preprocess_utils.calculate_prop_original_data_dict(invalid_needed_samples, labels_overview_dict)
    assert got == {}
    # test if the keys do not match
    valid_labels_overview_dict = {"classes_count": {"11": 12, "12": 16, "15": 63}}
    valid_needed_samples = {"1": 100, "2": 0, "3": 250}
    got = preprocess_utils.calculate_prop_original_data_dict(valid_needed_samples, valid_labels_overview_dict)
    assert got == {}
    # test if it calculates when everything is valid
    valid_labels_overview_dict = {"classes_count": {"1": 200, "2": 300, "3": 50}}
    valid_needed_samples = {"1": 100, "2": 0, "3": 250}
    got = preprocess_utils.calculate_prop_original_data_dict(valid_needed_samples, valid_labels_overview_dict)
    want = {"class": [1, 2, 3],
            "prop_existing_after_balancing": [0.67, 1, 0.17]}
    assert got == want


def test_dangerously_much_resampling_needed():
    # test if malformed input was given
    invalid_prop_original_data_dict = pd.DataFrame()
    got = preprocess_utils.dangerously_much_oversampling_needed(invalid_prop_original_data_dict)
    assert got is None
    # Test if the input dict does not contain the needed keys
    invalid_prop_original_data_dict = {"some_other_key": 15}
    got = preprocess_utils.dangerously_much_oversampling_needed(invalid_prop_original_data_dict)
    assert got is None
    # test if everything is valid and no dangerous proportion got detected
    valid_prop_original_data_dict = {"prop_existing_after_balancing": [0.99, 0.98, 0.97]}
    got = preprocess_utils.dangerously_much_oversampling_needed(valid_prop_original_data_dict)
    assert not got
    # test if everything is valid and  dangerous proportion was detected
    valid_prop_original_data_dict = {"prop_existing_after_balancing": [0.981, 0.972, 0.007], "class": [1, 2, 3]}
    got = preprocess_utils.dangerously_much_oversampling_needed(valid_prop_original_data_dict)
    assert got


def test_naive_oversampling():
    # Test if everything works as intended when no resampling needs to be done
    valid_needed_samples = {"1": 2, "2": 0, "3": 3}
    valid_key = "2"
    valid_df_to_resample = pd.DataFrame({"some_col": [100, 150], "another_col": [200, 260]})
    got_df, got_error = preprocess_utils.naive_oversampling(valid_needed_samples, valid_key, valid_df_to_resample)
    assert_frame_equal(got_df, valid_df_to_resample)

    # Test if everything works as intended when resampling needs to be done
    got_df, got_error = preprocess_utils.naive_oversampling(valid_needed_samples, "1", valid_df_to_resample)
    want = pd.DataFrame({"some_col": [100, 150, 100, 150], "another_col": [200, 260, 200, 260]})
    assert_frame_equal(got_df, want)


def test_perform_naive_oversampling():
    # Test if the program correctly returns when none of the given files exist:
    non_existing_file = "somewhere_undefined/some_where/non_existing_file.csv"
    if os.path.exists(non_existing_file):
        os.remove(non_existing_file)

    needed_samples = {"1": 0, "2": 1}
    root = test_config['TEST_ROOT']
    saving_folder = os.path.join(root, "preprocess_utils_test", "test_perform_naive_oversampling")
    if os.path.exists(saving_folder):
        shutil.rmtree(saving_folder)
    os.makedirs(saving_folder)

    key_with_params = {"key": "2", "needed_samples": needed_samples, "labels_overview_dict": {"classes_files": {"1": [
        non_existing_file], "2": []}}, "store_local": True, "saving_folder": saving_folder, "label_depth": 1, "seed": 42}
    got_str = preprocess_utils.perform_naive_oversampling(key_with_params)
    want = [data_utils.create_result_string("no_file_found", [1], 0, "classes_files_did_not_contain_files_for_key_2")]
    assert got_str == want

    # Test if the resampled df gets saved correctly
    df = pd.DataFrame({"some_col": [1, 2, 3], "label": [11, 12, 15]})
    df_2 = pd.DataFrame({"some_col": [1, 2, 3, 1], "label": [21, 222, 23, 24]})
    # Create csvs to read in
    df.to_csv(os.path.join(saving_folder, "file_1.csv").replace("\\","/"))
    df_2.to_csv(os.path.join(saving_folder, "file_2.csv").replace("\\","/"))
    key_with_params["labels_overview_dict"] = {"classes_files": {"1": [os.path.join(
        saving_folder, "file_1.csv")], "2": [os.path.join(saving_folder, "file_2.csv").replace("\\","/")]}}
    # If the resulting files already exist, delete them
    if os.path.exists(os.path.join(saving_folder, "1_file_1.csv").replace("\\","/")):
        os.remove(os.path.join(saving_folder, "1_file_1.csv").replace("\\","/"))
    if os.path.exists(os.path.join(saving_folder, "2_file_2.csv").replace("\\","/")):
        os.remove(os.path.join(saving_folder, "2_file_2.csv").replace("\\","/"))

    # Check if the program finished as intended
    got_str = preprocess_utils.perform_naive_oversampling(key_with_params)
    want = [data_utils.create_result_string(os.path.join(
        saving_folder, "file_2.csv"), [None], 1, "None")]
    assert got_str == want

    # Check if the resulting files exist
    assert os.path.isfile(os.path.join(saving_folder, "2_file_2.csv").replace("\\","/"))
    # Check if the naive oversampling got done correctly, meaning that an additional row got appended
    df = pd.read_csv(os.path.join(saving_folder, "2_file_2.csv").replace("\\","/"))
    assert_frame_equal(df, pd.DataFrame({"some_col": [1, 2, 3, 1, 3], "label": [2, 2, 2, 2, 2]}))


def test_remove_unneeded_keys():
    # test if everything works as intended
    dict_to_reduce = {"1": 10, "2": 20, "41": 42}
    reduced_keys_as_list = ["1", "41"]
    got_dict, got_removed_keys = preprocess_utils.remove_unneeded_keys(
        reduced_keys_as_list, dict_to_reduce)
    assert got_dict == {"1": 10, "41": 42}
    assert ["2"] == got_removed_keys


def test_create_dict_params():
    # test if everything works as intended
    all_unbalanced_files = ["some_file.csv", "another_file.csv"]
    label_depth = 1
    got = preprocess_utils.create_dict_params(
        all_unbalanced_files, label_depth)
    want = [{"file": "some_file.csv", "label_depth": 1},
            {"file": "another_file.csv", "label_depth": 1}]
    assert got == want


def test_update_overall_labels_overview_dict():
    # Test when no errors occur
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_update_overall_labels_overview_dict", "files_with_labels")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    file_path = os.path.join(folder, "some_file(1)_1.csv")

    pd.DataFrame({"some_col": [1], "label": [2]}).to_csv(file_path, index=False)
    file_path = os.path.join(folder, "some_file(1)_1.csv")
    file_with_params = {"label_depth": 1, "file": file_path}
    example_results = [preprocess_utils.create_labels_overview_dict(file_with_params)]
    got = preprocess_utils.update_overall_labels_overview_dict(example_results, {})
    want = {'classes_count': {'2': 1},
            'classes_files': {'2': [os.path.join(root, "preprocess_utils_test", "test_update_overall_labels_overview_dict", "files_with_labels", "some_file(1)_1.csv")]}}
    if "win" in sys.platform:
        got["classes_files"]["2"][0] = got["classes_files"]["2"][0].replace("\\","/")
    assert got == want


def test_key_with_params():
    # Test if everything works as intended
    needed_samples = {"1": 100, "2": 0}
    labels_overview_dict = {"classes_count": {"1": 100, "2": 200}, "classes_files": {"1": [], "2": []}}
    label_depth = 1
    store_local = True
    saving_folder = "/somewhere/some_folder"
    seed = 42
    got = preprocess_utils.key_with_params(
        needed_samples, labels_overview_dict, label_depth, store_local, saving_folder, seed)
    want = [{"key": "1",
                "labels_overview_dict": labels_overview_dict,
                "label_depth": label_depth,
                "store_local": store_local,
                "saving_folder": saving_folder,
                "needed_samples": needed_samples,
                "seed": seed},
            {"key": "2",
                "labels_overview_dict": labels_overview_dict,
                "label_depth": label_depth,
                "store_local": store_local,
                "saving_folder": saving_folder,
                "needed_samples": needed_samples,
                "seed": seed}]
    assert got == want


def test_create_list_of_file_dicts():
    table = {"needed_columns": ["some_col"], "label_depth": 1,
             "store_local": True, "saving_folder": "/somewhere/some_folder"}
    all_unreduced_files = ["file_1.csv", "file_2.csv"]
    got = preprocess_utils.create_list_of_file_dicts(all_unreduced_files, table)
    want = [{"file": "file_1.csv",
                "needed_columns": table.get("needed_columns"),
                "label_depth": table.get("label_depth"),
                "store_local": not table.get("dryrun"),
                "saving_folder": table.get("saving_folder")},
            {"file": "file_2.csv",
                "needed_columns": table.get("needed_columns"),
                "label_depth": table.get("label_depth"),
                "store_local": not table.get("dryrun"),
                "saving_folder": table.get("saving_folder")}]
    assert got == want


def test_reduce_data_of_file():
    # Test if the file does not exist
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_reduce_data_of_file")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    file_location = os.path.join(folder, "file.csv").replace("\\","/")
    file_with_params = {"file": file_location, "needed_columns": [["some_col"]], "label_depth": 1, "store_local": True, "saving_folder": folder}
    got = preprocess_utils.reduce_data_of_file(file_with_params)
    want = data_utils.create_result_string(os.path.join(folder, "file.csv"), [1], 0, "file_not_found").replace("\\","/")
    assert got == want

    file_with_params = {"file": file_location, "needed_columns": ["some_col"], "label_depth": 2, "store_local": True, "saving_folder": folder}
    pd.DataFrame({"label": [31, 31, 31]}).to_csv(file_location, index=False)
    got = preprocess_utils.reduce_data_of_file(file_with_params)
    want = data_utils.create_result_string(file_location, [], 1, "None")
    assert got == want

    # test if df does not contain all the needed cols
    file_with_params = {"file": os.path.join(folder, "file.csv").replace("\\","/"), "needed_columns": [["some_col", "another_col", "label"]], "label_depth": 1, "store_local": True, "saving_folder": folder}
    pd.DataFrame({"some_col": [1]}).to_csv(os.path.join(folder, "file.csv").replace("\\","/"), index=False)
    got = preprocess_utils.reduce_data_of_file(file_with_params)
    want = data_utils.create_result_string(os.path.join(folder, "file.csv"), [1], 0, "label_col_not_found")
    assert got == want

    # test if everything works as intended
    pd.DataFrame({"label": [1, 1, 1], "some_col": [1, 1, 1], "another_col": [2, 2, 2]}).to_csv(os.path.join(folder, "file.csv").replace("\\","/"), index=False)
    got = preprocess_utils.reduce_data_of_file(file_with_params)
    want = data_utils.create_result_string(os.path.join(folder, "file.csv"), [], 1, "None").replace("\\","/")
    assert got == want


def test_count_label_distribution():
    # test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_count_label_distribution")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    files_label_1 = []
    files_label_2 = []
    # create some files
    for i in range(100):
        filename = os.path.join(folder, f"file_{i}.csv")
        pd.DataFrame({"label": [1], "some_col": [1]}).to_csv(filename)
        files_label_1.append(filename)
        if i % 10 == 0:
            pd.DataFrame({"label": [2], "some_col": [1]}).to_csv(os.path.join(folder, f"2_file_{i}.csv"))
            files_label_2.append(os.path.join(folder, f"2_file_{i}.csv"))

    all_files, error = file_utils.get_files(folder, "csv")
    got = preprocess_utils.count_label_distribution(all_files, 1)
    want = {"classes_count": {"1": 100, "2": 10},
            "classes_files": {"1": files_label_1, "2": files_label_2}}
    # As multiprocessing is being used, the order of the files in classes_files might vary, therefore resorting to comparing sets
    assert got["classes_count"] == want["classes_count"]
    assert set(want["classes_files"]["1"]) == set(want["classes_files"]["1"])
    assert set(want["classes_files"]["2"]) == set(want["classes_files"]["2"])


def test_remove_labels_to_skip():
    # test if everything works as intended:
    dict_with_labels = {i : 10 for i in range(10, 15)}
    want = {10: 10, 11: 10, 12: 10}
    got_dict, _ = preprocess_utils.remove_labels_to_skip(dict_with_labels)
    assert got_dict == want


def test_create_subdicts_labels_overview_dict():
    # Test if wrong label_depth is given
    got_list, got_str = preprocess_utils.create_subdicts_labels_overview_dict(15, {}, {})
    assert got_list == []
    assert got_str == "given_label_depth_not_yet_implemented"

    # Test if everything works as intended
    labels_overview_dict = {"classes_count": {"1": 10, "2": 10}, "classes_files": {"1": "label_1_file", "2": "label_2_file"}}
    got_list, got_str = preprocess_utils.create_subdicts_labels_overview_dict(2, labels_overview_dict, {"generate_features": 12})
    want_list = [{"classes_count": {"1": 10}, "classes_files": {"1": "label_1_file"}}, {"classes_count": {"2": 10}, "classes_files": {"2": "label_2_file"}}]
    assert got_list == want_list


def test_split_df_to_handle_into_chunks():
    # Test if everything works as intended
    example_df = pd.DataFrame({"label": list(range(24))*5, "some_col": list(range(24, 48))*5})
    got = preprocess_utils.split_df_to_handle_into_chunks(example_df)
    fragment_df = pd.DataFrame({"label": list(range(24)), "some_col": list(range(24, 48))})
    assert len(got) == 5
    for i in range(5):
        assert_frame_equal(got[i], fragment_df)


def test_create_list_of_names():
    # test when the input contains errors
    got_list_of_names, got_error = preprocess_utils.create_list_of_names("some_folder", "prefix", "some_process", ".filetype", 15, False, ["a", "b"])
    assert got_list_of_names == []
    assert got_error == "length_of_list_of_specific_prefixes_does_not_match_len_other_list"

    # test when specific prefixes were given
    individual_prefixes = ["individual_prefix_1", "individual_prefix_2", "individual_prefix_3"]
    window_uuids = ["a", "b", "c"]
    got_list_of_names, got_error = preprocess_utils.create_list_of_names("some_folder", "prefix", "some_process", ".filetype", 3, False, individual_prefixes, window_uuid=window_uuids)

    assert len(got_list_of_names) == 3

    # test if the individual prefixes got found
    for name in got_list_of_names:
        prefix_found = False
        for prefix in individual_prefixes:
            if prefix in name:
                prefix_found = True
                individual_prefixes.remove(prefix)
        assert prefix_found

    # Test when no individual prefixes were given
    got_list_of_names, got_error = preprocess_utils.create_list_of_names("some_folder", "prefix", "some_process", ".filetype", 3, False, None, window_uuids)
    assert len(got_list_of_names) == 3
    for name in got_list_of_names:
        assert"some_folder" in name
        assert "prefix" in name
        assert "some_process" in name
        assert ".filetype" in name

def test_save_in_subfiles():
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_save_in_subfiles")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # test if no data was given
    df_with_params = {"df": pd.DataFrame(), "saving_folder": folder}
    got = preprocess_utils.save_in_subfiles(df_with_params)
    assert got ==  data_utils.create_result_string("some_file", [1], 0, "got_empty_df")

    # test if everything works as intended
    df_with_params = {"df": pd.DataFrame({"label": [1, 2], "some_col": [ 15, 30]}), "saving_folder": folder, "label": 15, "grid": False, "hierarchical_featured": False, "window_uuid": ["a", "b"]}
    got = preprocess_utils.save_in_subfiles(df_with_params)
    results = got[0].split("+++")

    assert results[1] ==  "True"
    assert results[2] == "None"
    assert results[3] == "2"


def test_handle_labels_df():
    # test when the df to append was not successful
    result_list = [data_utils.create_result_string("some_file", [1], 0, "label_not_in_cols")]
    df_labels = pd.DataFrame({"filename": ["some_file"], "label": [1.]})
    got = preprocess_utils.handle_labels_df(result_list, df_labels)
    assert_frame_equal(got, df_labels)

    # test when the df to append was successful
    result_list = [data_utils.create_result_string("somewhere/1_some_file_2.csv", [], 1, "None")]
    df_labels = pd.DataFrame({"filename": ["some_file"], "label": [1.]})
    got = preprocess_utils.handle_labels_df(result_list, df_labels)
    want = pd.DataFrame({"filename": ["some_file", "somewhere/1_some_file_2.csv"], "label": [1., 1.]})
    assert_frame_equal(got, want)


def test_prepare_folder_for_custom_loading():
    # Test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_prepare_folder_for_custom_loading")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    pd.DataFrame({"col_1": [2, 2], "col_2": [3, 3], "window_uuid": ["1a", "2a"]}).to_csv(os.path.join(folder, "1_df.csv"), index=False)
    pd.DataFrame({"col_1": [2, 2, 2], "col_2": [3, 3, 3], "window_uuid": ["3a", "4a", "5a"]}).to_csv(os.path.join(folder, "2_df.csv"), index=False)
    pd.DataFrame({"col_1": [2], "col_2": [3], "window_uuid": ["6a"]}).to_csv(os.path.join(folder, "3_df.csv"), index=False)
    pd.DataFrame({"col_1": [2], "col_2": [3], "window_uuid": ["7a"]}).to_csv(os.path.join(folder, "4_df.csv"), index=False)
    pd.DataFrame({"col_1": [2], "col_2": [3], "window_uuid": ["8a"]}).to_csv(os.path.join(folder, "5_df.csv"), index=False)
    got_df, got_log, error = preprocess_utils.prepare_folder_for_custom_loading(folder, {}, {})

    assert error is None
    assert set([1, 2, 3, 4, 5]) == set(got_df["label"])
    assert len(got_df) == 8
    got_sorted = got_df.sort_values("label", ascending=True).reset_index(drop=True)
    assert got_sorted["label"].value_counts().sort_index().iloc[0] == 2
    assert got_sorted["label"].value_counts().sort_index().iloc[1] == 3
    assert got_sorted["label"].value_counts().sort_index().iloc[2] == 1
    assert got_sorted["label"].value_counts().sort_index().iloc[3] == 1
    assert got_sorted["label"].value_counts().sort_index().iloc[4] == 1

    assert os.path.isfile(os.path.join(folder, "labels.csv"))
    all_files, error = file_utils.get_files(folder, "csv")
    assert len(all_files) == 9
    assert_frame_equal(got_df, pd.read_csv(os.path.join(folder, "labels.csv")))


def test_remove_processed_key_with_params():
    # Test if everything works as intended and some files have not been processed yet
    keys_with_parms = [{"key": 1, "labels_overview_dict": {"classes_files": {1: ["somewhere/somefile.csv"]}}},
                        {"key": 2, "labels_overview_dict": {"classes_files": {2: ["somewhere/another_file.csv"]}}}]
    log_file = {"test-method": {"successfully_processed_files":["somewhere/somefile.csv", "yet_another_file"]}}
    got = preprocess_utils.remove_processed_key_with_params(keys_with_parms, log_file, "test-method")
    assert got == [{"key": 2, "labels_overview_dict": {"classes_files": {2: ["somewhere/another_file.csv"]}}}]

    # Test if everythig works as intended and nothing is left
    log_file = {"test-method": {"successfully_processed_files": ["somewhere/somefile.csv", "somewhere/another_file.csv", "yet_another_file"]}}
    got = preprocess_utils.remove_processed_key_with_params(keys_with_parms, log_file, "test-method")
    assert got == []


def test_valid_table():
    # test if no datapath was given
    error = preprocess_utils.valid_table({})
    assert error == "no_data_path_given"

    # test if the datapath does not exist
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_valid_table")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    error = preprocess_utils.valid_table({"data_path": folder})
    assert error == "datapath_does_not_exist"

    os.makedirs(folder)
    valid_anonymize_file = {"create_windows": {"needed_columns": [], "data_columns": []},
                            "svm_features": {"top": [], "lifting": [], "walking": []},
                            "feature_utils": {"accel_cols": [], "differences_cols": [], "needed_cols": [], "on_single_column": [], "magn_feat_name": []}}
    error = file_utils.save_json(valid_anonymize_file, os.path.join(folder, "anonymization_file.json"))
    assert error is None

    # test if dryrun was not given
    error = preprocess_utils.valid_table({"data_path": folder})
    assert error == "no_dryrun_given"

    # test if dryrun has a wrong dt
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": 10})
    assert error == "got_dryrun_of_nonbool_datatype"

    # test if input_data was not given
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True})
    assert error == "no_input_data_folder_given"

    # test if the input data folder does not exist
    input_data_folder = os.path.join(folder, "input_data")
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True, "input_data": input_data_folder})
    assert error == "input_data_folder_does_not_exist"

    # test if the input data folder is empty
    os.makedirs(input_data_folder)
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True, "input_data": input_data_folder})
    assert error == "no_data_in_dir"

    # Test if incorrect output date was given
    pd.DataFrame({"some_col": [1, 2, 3, 4]}).to_csv(os.path.join(input_data_folder, "label.csv"), index=False)
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True, "input_data": input_data_folder, "output_date": "455s5sdfsdf"})
    assert error == "got_time_str_of_invalid_len"

    # Test if the wrong granularity was given
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True, "input_data": input_data_folder, "granularity": "low"})
    assert error == "granularity_level_low_is_not_implemented_yet"

    # test if plot has a wrong dt
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True, "input_data": input_data_folder, "granularity": "top", "plot": 12})
    assert error ==  "got_plot_of_nonbool_datatype"

    # test if the method reduce_data was chosen, but the data does not contain features
    error = file_utils.save_json({"another_preprocessing_step": 12}, os.path.join(input_data_folder, "log.json"))
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True, "input_data": input_data_folder, "granularity": "top", "plot": True, "method": "reduce_data"})
    assert error == "got_non_featured_input_data_for_method_reduce_data"

    # test if the data was not oversampled before prepare_dataset
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True, "input_data": input_data_folder, "granularity": "top", "plot": True, "method": "prepare_dataset"})
    assert error == "got_unbalanced_set_for_training"

    # test if Grid of a nonbool dt was given
    error = file_utils.save_json({"balancing_over": 12}, os.path.join(input_data_folder, "log.json"))
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True, "input_data": input_data_folder,
                                            "granularity": "top", "plot": True, "method": "prepare_dataset", "grid": [True]})
    assert error == "got_grid_of_non_bool_dt"

    # test if everything works as intended
    error = preprocess_utils.valid_table({"data_path": folder, "dryrun": True, "input_data": input_data_folder,
                                            "granularity": "top", "plot": True, "method": "prepare_dataset", "grid": True})
    assert error is None

def test_handle_data_for_feature_based_ffnn():
    # Test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_handle_data_for_feature_based_ffnn")
    input_data_folder = os.path.join(folder, "input_folder")
    saving_folder = os.path.join(folder, "saving_folder")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(input_data_folder)
    os.makedirs(saving_folder)

    method = "example_method"
    log_file = {method: {
                "successfully_processed_files": [],
                "occurred_errors": {},
                "saved_files": 0,
                "dryrun": False},
                "balancing_over": {}}
    error = file_utils.save_json(log_file, os.path.join(saving_folder, "log.json"))
    assert error is None
    label_depth = 2
    table = {}
    store_local = True

    example_df = pd.DataFrame({"some_col": [1, 2, 3], "label": [11, 21, 31], "window_uuid": ["a", "b", "c"]})
    example_df.to_csv(os.path.join(input_data_folder,"some_file.csv"), index=False)

    got = preprocess_utils.handle_data_for_feature_based_ffnn(input_data_folder, saving_folder, method, log_file, label_depth, table, store_local)

    assert got is None
    resulting_files, error = file_utils.get_files(saving_folder)
    assert error is None
    resulting_files = [file.replace("\\","/") for file in resulting_files]
    assert len(resulting_files) == 4
    assert os.path.isfile(os.path.join(saving_folder, "labels.csv").replace("\\","/"))

    resulting_labels_df = pd.read_csv(os.path.join(saving_folder, "labels.csv").replace("\\","/"))
    assert set(resulting_labels_df["label"]) == set([11, 21, 31])
    resulting_files.remove(os.path.join(saving_folder, "labels.csv").replace("\\","/"))
    assert set(resulting_files) == set(resulting_labels_df["filename"])
    # Test if the label (via the filename) is equal to the label found in the labels.csv

    for file in resulting_files:
        label_of_file = int(file.split("/")[-1].split("_")[0])
        assert resulting_labels_df[resulting_labels_df["filename"] == file]["label"].to_numpy()[0] == label_of_file

    assert os.path.isfile(os.path.join(saving_folder, "log.json"))

    with open(os.path.join(saving_folder, "log.json")) as f:
        created_log = json.load(f)
    assert created_log[method]["successfully_processed_files"] == "all"
    assert created_log[method]["saved_files"] == 3

def test_prepare_dataset_for_flattened_data():
    # Test if everything works as intended
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_prepare_dataset_for_flattened_data")
    input_data_folder = os.path.join(folder, "input_folder")
    saving_folder = os.path.join(folder, "saving_folder")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(input_data_folder)
    os.makedirs(saving_folder)

    method = "prepare_dataset"
    log_file = {method: {
                "successfully_processed_files": [],
                "occurred_errors": {},
                "saved_files": 0,
                "dryrun": False}}
    error = file_utils.save_json(log_file, os.path.join(saving_folder, "log.json"))
    assert error is None
    label_depth = 2
    table = {}
    store_local = True

    # create some flattened data after balancing as input
    for i in [11, 12, 15]:
        example_df = pd.DataFrame([[i+10]*99]*20)
        example_df["label"] = i
        example_df['window_uuid'] = "a"
        example_df.to_csv(os.path.join(input_data_folder,f"{i}_some_file.csv"), index=False)

    got = preprocess_utils.prepare_dataset_for_flattened_data(input_data_folder, saving_folder, method, log_file, label_depth, table, store_local, False)

    assert got is None
    resulting_files, error = file_utils.get_files(saving_folder)
    resulting_files = [file.replace("\\","/") for file in resulting_files]
    assert error is None
    # The resulting mount of files = 1 per label + labels.csv = 4
    assert len(resulting_files) == 4
    assert os.path.isfile(os.path.join(saving_folder, "labels.csv"))

    resulting_labels_df = pd.read_csv(os.path.join(saving_folder, "labels.csv"))
    assert set(resulting_labels_df["label"]) == set([1, 2, 3])
    resulting_files.remove(os.path.join(saving_folder, "labels.csv").replace("\\","/"))
    assert set(resulting_files) == set(resulting_labels_df["filename"])
    for file in resulting_files:
        label_of_file = int(file.split("/")[-1].split("_")[0])
        assert resulting_labels_df[resulting_labels_df["filename"] == file]["label"].to_numpy()[0] == label_of_file

    assert os.path.isfile(os.path.join(saving_folder, "log.json"))

    with open(os.path.join(saving_folder, "log.json")) as f:
        created_log = json.load(f)
    assert created_log[method]["successfully_processed_files"] == "all"
    assert created_log[method]["saved_files"] == 60


def test_return_df_channelwise_if_needed():
    # Test if the data is shall not be returned in channelwise format
    df = pd.DataFrame([x for x in range(100)])
    got = preprocess_utils.return_df_channelwise_if_needed(df, False, 10)
    assert_frame_equal(got, df)

    # test if everything works as intended
    got = preprocess_utils.return_df_channelwise_if_needed(df, True, 10)
    want = pd.DataFrame(pd.DataFrame(
        [[i for i in range(x*10, x*10+10)] for x in range(10)]).to_numpy()).T
    assert_frame_equal(got, want)


def test_correct_format_for_multilevel_classification():
    # Test if top-level classification is required
    df = pd.DataFrame({"col_1": [1], "col_2": [2], "col_3": [3], "col_4": [4], "col_5": [
                        5], "col_6": [6], "col_7": [7], "col_8": [8], "col_9": [9], "label": [11]})
    needed_columns = [["label", "col_1", "col_3", "col_5"]]
    got_df, got_error = preprocess_utils.correct_format_for_multilevel_classification(df, needed_columns)
    want_df = pd.DataFrame({"label": [11], "col_1": [1], "col_3": [3], "col_5": [5]})
    assert got_error is None
    assert_frame_equal(got_df, want_df)

    # test if a mid_level classification was requested, but the data seen is resting data
    df["label"] = 31
    want_df["label"] = 31

    needed_columns.extend([["label", "col_1", "col_7"]])
    needed_columns.extend([["label", "col_3", "col_9"]])
    got_df, got_error = preprocess_utils.correct_format_for_multilevel_classification(df, needed_columns)
    assert got_error is None
    assert_frame_equal(got_df, want_df)

    # Test if everything works as intended for lifting data
    df["label"] = 11
    want_df["label"] = 11
    got_df, got_error = preprocess_utils.correct_format_for_multilevel_classification(df, needed_columns)
    assert got_error is None
    want_df = pd.DataFrame({"label": [11, 11], "col_1": [1, 1], "col_3": [3.0, np.nan], "col_5": [5.0, np.nan], "col_7": [np.nan, 7.0]})
    assert_frame_equal(got_df, want_df)

    # Test if everything works as intended for walking data
    df["label"] = 211
    want_df["label"] = 211
    got_df, got_error = preprocess_utils.correct_format_for_multilevel_classification(df, needed_columns)
    assert got_error is None
    want_df = pd.DataFrame({"label": [211, 211], "col_1": [1, np.nan], "col_3": [3, 3], "col_5": [5.0, np.nan], "col_9": [np.nan, 9.0]})
    assert_frame_equal(got_df, want_df)


def test_split_into_file_chunks():
    df = pd.DataFrame({"some_col": [1, 2, 3, 4]})

    # test when the seen data is nor hierarchical
    got = preprocess_utils.split_into_file_chunks(df, False, {"label_depth": 2})
    want = [pd.DataFrame({"some_col": [1]}), pd.DataFrame({"some_col": [2]}), pd.DataFrame({"some_col": [3]}), pd.DataFrame({"some_col": [4]})]
    for number, _ in enumerate(want):
        assert_frame_equal(got[number], want[number])

    got = preprocess_utils.split_into_file_chunks(df, False, {"label_depth": 1})
    for number, _ in enumerate(want):
        assert_frame_equal(got[number], want[number])

    # Test when hierarchical data is being processed, but the currently seen label exists only as a top-level class
    got = preprocess_utils.split_into_file_chunks(df, True, {"label_depth": 2})
    for number, _ in enumerate(want):
        assert_frame_equal(got[number], want[number])

    # test when the data truly comes from hierarchical data and needs splitting
    df = pd.DataFrame({"some_col": [1, np.nan, 3, np.nan]})
    want = [pd.DataFrame({"some_col": [1, np.nan]}),
            pd.DataFrame({"some_col": [3, np.nan]})]
    got = preprocess_utils.split_into_file_chunks(df, True, {"label_depth": 2})
    for number, _ in enumerate(want):
        assert_frame_equal(got[number], want[number])


def test_correct_label_for_saving_prepared_file():
    # test when the data is not in hierarchical format
    got = preprocess_utils.correct_label_for_saving_prepared_file("somewhere/15_some_info.csv", False, pd.DataFrame())
    assert got == 15

    # Test when the data exists in hierarchical format
    got = preprocess_utils.correct_label_for_saving_prepared_file("somewhere/15_some_info.csv", True, pd.DataFrame({"label_top": [1], "label_mid": [3]}))
    assert got == 13


def test_save_file():
    # Test when the data is not in hierarchical format
    root = test_config['TEST_ROOT']
    folder = os.path.join(root, "preprocess_utils_test", "test_save_file")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    preprocess_utils.save_file(pd.Series([1, 2, 3, 4]), False, os.path.join(folder, "file.csv"), False)
    assert os.path.isfile(os.path.join(folder, "file.csv"))
    got = pd.read_csv(os.path.join(folder, "file.csv"), header=None, index_col=None).iloc[:, 0]
    want = pd.Series([1, 2, 3, 4])
    assert_series_equal(got, want, check_names=False)

    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # test when the data is in hierarchical format
    preprocess_utils.save_file(pd.Series([1, 2, 3, 4]), False, os.path.join(folder, "file.csv"), True)
    assert os.path.isfile(os.path.join(folder, "file.csv"))
    got = pd.read_csv(os.path.join(folder, "file.csv"), header=None, index_col=None).iloc[:, 0]
    want = pd.Series([1, 2])
    assert_series_equal(got, want, check_names=False)


def test_create_two_level_hierarchical_dict_of_labels():
    # Test if everything works as intended
    got = preprocess_utils.create_two_level_hierarchical_dict_of_labels([11, 12, 15, 21, 22, 23, 32])
    want = {"1": {"subkeys": [11, 12, 15]}, "2": {"subkeys": [21, 22, 23]}, "3": {"subkeys": [32]}}
    assert got == want


def test_calculate_needed_extra_samples_hierarchical_data():
    # Test if everything works as intended

    got = preprocess_utils.calculate_needed_extra_samples_hierarchical_data(
        {"classes_count": {"11": 180, "12": 120, "15": 100, "21": 50, "22": 30, "23": 70, "32": 40}})
    want = {"11": 0, "12": 60, "15": 80, "21": 130, "22": 150, "23": 110, "32": 500}
    assert got == want


def test_create_hierarchical_dict_of_samples():
    # Test if everything works as intended
    seen_labels = [11]*20
    seen_labels.extend([12]*30 + [15]*25 + [21]*10 + [22]*30 + [23]*15 + [31]*40)
    label_series = pd.Series(seen_labels)

    got_dict, got_error = preprocess_utils.create_hierarchical_dict_of_samples(label_series)
    want = {"top_level": {1: 75, 2: 55, 3: 40}, "mid_level": {1: {11: 20, 12: 30, 15: 25}, 2: {21: 10, 22: 30, 23: 15}, 3: {31: 40}}, "low_level": {}}
    assert got_dict == want


def test_valid_hierarchical_oversampling_seen():
    # Test if everything works as intended when the oversampling was not valid
    seen_labels = [11]*20
    seen_labels.extend([12]*30 + [15]*25 + [21]*10 + [22]*30 + [23]*15 + [31]*40)
    label_series = pd.Series(seen_labels)

    result = preprocess_utils.valid_hierarchical_oversampling_seen(label_series)
    assert not result

    seen_labels = [11]*20
    seen_labels.extend([12]*20 + [15]*20 + [21]*20 + [22]*20 + [23]*20 + [31]*60)
    label_series = pd.Series(seen_labels)

    result = preprocess_utils.valid_hierarchical_oversampling_seen(label_series)
    assert result


def test_draw_random_indexes():
    # Test if everything works as intended
    got = preprocess_utils.draw_random_indexes(60, 50, 42)
    want = np.array([12, 13, 14, 15, 20, 21, 28, 29, 38, 39])
    assert np.array_equal(got, want)
