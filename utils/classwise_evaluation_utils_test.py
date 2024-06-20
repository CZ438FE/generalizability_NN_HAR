import os
import math
import pandas as pd
from pandas.testing import assert_frame_equal
from shutil import rmtree
import numpy as np

from config.test_config import test_config
from utils import classwise_evaluation_utils


def test_check_classifier_existence():
    root = test_config['TEST_ROOT']
    # test when the folder does not exist
    non_existing_folder = os.path.join(root, "classwise_evaluation_utils", "test_check_classifier_existence", "non-existing-folder")
    if os.path.exists(non_existing_folder):
        rmtree(non_existing_folder)
    got = classwise_evaluation_utils.check_classifier_existence(non_existing_folder, "", [])
    assert got == "folder_not_found"

    # Test when the predictions.csv does not exist
    existing_empty_foder = os.path.join(root, "classwise_evaluation_utils", "test_check_classifier_existence", "existing-folder-empty")
    if os.path.exists(existing_empty_foder):
        rmtree(existing_empty_foder)
    os.makedirs(existing_empty_foder)
    got = classwise_evaluation_utils.check_classifier_existence(existing_empty_foder, "", [])
    assert got == "predictions_not_found_in_top_level_folder_not_found"

    # test when required cols are missing
    existing_foder = os.path.join(root, "classwise_evaluation_utils", "test_check_classifier_existence", "folder-wrong-cols")
    if os.path.exists(existing_foder):
        rmtree(existing_foder)
    os.makedirs(existing_foder)
    predictions_path = os.path.join(existing_foder, "predictions.csv")
    pd.DataFrame({"not-the-needed-col": [1, 2]}).to_csv(predictions_path)
    assert os.path.isfile(predictions_path)
    
    got = classwise_evaluation_utils.check_classifier_existence(existing_foder, "", ["a_required_col", "another_required_col"])
    assert got == "Needed_columns_missing"

    # Test when all is valid
    existing_foder = os.path.join(root, "classwise_evaluation_utils", "test_check_classifier_existence", "folder-all-valid")
    if os.path.exists(existing_foder):
        rmtree(existing_foder)
    os.makedirs(existing_foder)
    predictions_path = os.path.join(existing_foder, "predictions.csv")
    pd.DataFrame({"a_required_col": [1, 2], "another_required_col": [1, 2]}).to_csv(predictions_path)
    assert os.path.isfile(predictions_path)
    
    got = classwise_evaluation_utils.check_classifier_existence(existing_foder, "", ["a_required_col", "another_required_col"])
    assert got is None


def test_valid_table(): 
    root = test_config['TEST_ROOT']

    # Test when correctly classifiers valid top-level as valid
    existing_foder = os.path.join(root, "classwise_evaluation_utils", "test_valid_table", "top-level-valid")
    if os.path.exists(existing_foder):
        rmtree(existing_foder)
    os.makedirs(existing_foder)
    predictions_path = os.path.join(existing_foder, "predictions.csv")
    pd.DataFrame({"file": [1, 2], "true_label": [1, 2], "probability_correct": [0.2, 0.7], "probability_incorrect": [0.8, 0.3]}).to_csv(predictions_path)
    assert os.path.isfile(predictions_path)
    got = classwise_evaluation_utils.valid_table({"top_level_classifier_folder": existing_foder })
    assert got is None

    # Test when lifting or walking folder are invalid: same folder
    non_existing_folder = os.path.join(root, "classwise_evaluation_utils", "test_valid_table", "non-existing-folder")
    if os.path.exists(non_existing_folder):
        rmtree(non_existing_folder)
    got = classwise_evaluation_utils.valid_table({"top_level_classifier_folder": existing_foder,
                                                  "lifting_classifier_folder": non_existing_folder,
                                                  "walking_classifier_folder": non_existing_folder })
    assert got == "identical_folders_given"

    # Test when lifting or walking folder are invalid: Do not exist
    non_existing_folder_2 = os.path.join(root, "classwise_evaluation_utils", "test_valid_table", "non-existing-folder-2")
    if os.path.exists(non_existing_folder_2):
        rmtree(non_existing_folder_2)
    got = classwise_evaluation_utils.valid_table({"top_level_classifier_folder": existing_foder,
                                                  "lifting_classifier_folder": non_existing_folder,
                                                  "walking_classifier_folder": non_existing_folder_2 })
    assert got == "folder_not_found"


def test_plot_ROC_curve():
    root = test_config['TEST_ROOT']

    # Test if saving does work as intended
    y_true = pd.Series([0,0,0,0,1,1,1,1])
    probas = [0.1, 0.4, 0.8, 1.0, 0.8, 0.6, 0.7, 0.2]
    store_local = True
    seen_class = "Test class"
    saving_folder = os.path.join(root, "classwise_evaluation_utils", "test_plot_ROC_curve")
    if os.path.exists(saving_folder):
        rmtree(saving_folder)
    os.makedirs(saving_folder)
    classwise_evaluation_utils.plot_ROC_curve(y_true, probas, store_local, saving_folder, seen_class)
    assert os.path.isfile(os.path.join(saving_folder, f"ROC_class_{seen_class}.png"))


def test_classwise_evaluation_metrics():
    probabilities_dict = {"1": pd.DataFrame({"true_label_two_class": [0, 1, 1], "probability_target_class": [0.2, 0.8, 0.4]})}
    got = classwise_evaluation_utils.classwise_evaluation_metrics(probabilities_dict)
    want = {"1": {"AUC": 1.0, "Precision": 1.0, "Specificity": 0.5, "fbeta_scores": 0.6666666666666666, "Supports": 2}}
    assert got == want


def test_get_UUID_from_filename():
    # Tets if all works as intended
    want = "ioasdhalskdglasgdljahgsdljgalsdjgalsd"
    mock_uuid = f"/home/some-user/some_folder/some-label_prepared_{want}.csv"
    got = classwise_evaluation_utils.get_UUID_from_filename(mock_uuid)
    assert got ==  want

def test_valid_filename():
    assert not classwise_evaluation_utils.valid_filename("i am not a valid filename")
    assert classwise_evaluation_utils.valid_filename(f"/home/some-user/some_folder/some-label_prepared_some_file.csv")


def test_extend_probabilities_dict_with_class():
    classifier_df = pd.DataFrame({"true_label": [1, 2, 3],
                                "Raw_prediction_label_1": [1, 4, 5],
                                "Raw_prediction_label_2": [2, 2, 2],
                                "Raw_prediction_label_3": [8, 2, 3],
                                "file": ["file_1", "file_2", "file_3"]
                                })
    probability_wanted_class = [math.exp(8) / (math.exp(8) + math.exp(2)),
                                math.exp(2) / (math.exp(2) + math.exp(4)),
                                math.exp(3) / (math.exp(3) + math.exp(5))]
    probability_rest_class = [1 - value for value in probability_wanted_class]
    relevant_class = 3
    want_df = pd.DataFrame({"file": ["file_1", "file_2", "file_3"], 'true_label_two_class': [0, 0, 1], 'probability_rest_class': probability_rest_class,'probability_target_class': probability_wanted_class, '1_class': relevant_class})
    got = classwise_evaluation_utils.extend_probabilities_dict_with_class({}, classifier_df, relevant_class)
    assert str(relevant_class) in got.keys()
    assert_frame_equal(got[str(relevant_class)], want_df)


def test_append_subclassifier_raw_predictions_to_top_level_df():
    # Contains 2 lifting files, one walking and one resting
    top_level_df = pd.DataFrame({"file": ["/file_1_prepared_1.csv", "/file_2_prepared_2.csv", "/file_3_prepared_3.csv", "/file_4_prepared_4.csv"],
                                 "true_label": [1, 1, 2, 3]})
    
    lifting_df = pd.DataFrame({"file": ["/file_1_prepared_1.csv", "/file_2_prepared_2.csv"],
                               "uuid": ["1", "2"],
                               "true_label": [1, 2],
                               "Raw_prediction_label_1": [10, 5],
                               "Raw_prediction_label_2": [7, 8]})

    walking_df = pd.DataFrame({"file": ["/file_3_prepared_3"],
                               "uuid": ["3"],
                               "true_label": [3],
                               "Raw_prediction_label_1": [2],
                               "Raw_prediction_label_2": [3],
                               "Raw_prediction_label_3": [4]})
    want = top_level_df.copy(True)
    want['all_class_label'] = [1, 2, 5, 6]
    want['sub_classifier_label'] = [1, 2, 3, 0]
    want['Subclass_Raw_prediction_label_1'] = [10, 5, 2, -np.inf]
    want['Subclass_Raw_prediction_label_2'] = [7, 8, 3, -np.inf]
    want['Subclass_Raw_prediction_label_3'] = [-np.inf, -np.inf, 4, -np.inf]
    got = classwise_evaluation_utils.append_subclassifier_raw_predictions_to_top_level_df(top_level_df, lifting_df, walking_df)
    assert_frame_equal(got, want)


def test_same_subclass():
    # test for resting classes
    assert not classwise_evaluation_utils.same_subclass(3, 7, False)
    assert not classwise_evaluation_utils.same_subclass(3, 6, True)

    # Test for lifting classes and aggregation
    assert classwise_evaluation_utils.same_subclass(1, 1, True)
    assert classwise_evaluation_utils.same_subclass(1, 2, True)
    assert not classwise_evaluation_utils.same_subclass(1, 4, False)

    # Test for walking classes and aggregation
    assert classwise_evaluation_utils.same_subclass(2, 3, True)
    assert classwise_evaluation_utils.same_subclass(2, 5, True)
    assert not classwise_evaluation_utils.same_subclass(1, 5, False)


def test_create_two_class_classifier_df():
    top_level_classifier_df = pd.DataFrame({"file": ["/file_1_prepared_1.csv", "/file_2_prepared_2.csv", "/file_3_prepared_3.csv", "/file_4_prepared_4.csv", "/file_5_prepared_5.csv", "/file_6_prepared_6.csv"],
                                            "true_label": [1, 1, 2, 2, 2, 3],
                                            "all_class_label": [1, 2, 3, 4, 5, 6],
                                            "sub_classifier_label": [1, 2, 1, 2, 3, 0],
                                            "Raw_prediction_label_1":          [1,             1, 1, 1,  1, 1],
                                            "Raw_prediction_label_2":          [2,             2, 2, 2,  2, 2],
                                            "Raw_prediction_label_3":          [3,             3, 3, 3,  3, 3],
                                            "Subclass_Raw_prediction_label_1": [10,            5, 2, 4,  6, -np.inf],
                                            "Subclass_Raw_prediction_label_2": [7,             8, 3, 7,  8, -np.inf],
                                            "Subclass_Raw_prediction_label_3": [-np.inf, -np.inf, 9, 10, 4, -np.inf]})
    label = 5
    want = pd.DataFrame({"true_label": [0, 0, 0, 0, 5, 0], 
                        "Raw_prediction_label_5":   [2., 2., 2., 7., 4., 2.], 
                        "Raw_prediction_label_0": [3., 3., 9., 10., 8., 3.], 
                        "file": top_level_classifier_df['file']})
    got = classwise_evaluation_utils.create_two_class_classifier_df(top_level_classifier_df, label)
    assert_frame_equal(got, want)