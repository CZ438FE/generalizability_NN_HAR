import os
import uuid
import math
import json
import itertools
import pandas as pd
import numpy as np
from multiprocessing import Pool

from config.log import logger
from utils import file_utils, data_utils, time_utils


def appropriate_label_depth(granularity: str):
    """ Returns the correct label depth based on the given granularity
    """
    if granularity not in ['top', 'mid', 'low']:
        logger.error(f"Got invalid granularity {granularity}")
        return 0, "invalid_granularity_given"
    if granularity == "low":
        return 0, "granularity low not implemented yet"

    return {"mid": 2, "top": 1}.get(granularity), None


def return_svm_features(label_depth: int, data_path: str):
    """Returns the features which are being used by the baseline svm of the given granularity as input 
    """
    if label_depth not in [1, 2]:
        logger.error(f"Cannot return the needed features when an invalid label_depth is given")
        return [], "invalid_label_depth_given"

    with open(os.path.join(data_path, "anonymization_file.json")) as f:
        anonymize_file: dict = json.load(f)

    # this are the features used for the top-level classifier
    needed_columns = [anonymize_file.get("svm_features").get("top")]

    if label_depth == 2:
        # Additionally keep the features for the subactivity-weigt handeling analyzer
        needed_columns.extend([anonymize_file.get("svm_features").get("lifting")])

        # Additionally keep the features for the subactivity-walking analyzer
        needed_columns.extend([anonymize_file.get("svm_features").get("walking")])

    return needed_columns, None


def update_label_distributions(label_dict: dict, df: pd.DataFrame, label_depth: int):
    """Counts the appearances of the labels in the seen df and updates dict correctly using the label_depth
    """
    if "label" not in df.columns:
        logger.error("Counting the labels is not possible when df does not contain a column label")
        return {}, "label_col_missing"

    label_appearances = df["label"].value_counts()

    for i in range(len(label_appearances)):
        key_in_label_dict = str(label_appearances.index[i])[:label_depth]
        if key_in_label_dict == "-":
            return {}, "strange_entry_detected"
        # check if the seen label has the expected granularity
        if len(key_in_label_dict) < label_depth and label_depth != 3:
            logger.error(f"Got label of inappropriate granularity: Expected depth {label_depth}, got {len(key_in_label_dict)}")
            return {}, "label_of_inappropriate_granularity_found"
        # As every window contains only a singular label, keep this label
        if key_in_label_dict not in label_dict.keys():
            label_dict[key_in_label_dict] = 1

    return label_dict, None


def update_classes_files(label_dict: dict, df: pd.DataFrame, label_depth: int, file: str):
    """updates the collection of files belonging to each seen class
    """
    if "label" not in df.columns:
        logger.error(f"No label column given, cannot sort given file {file}")
        return {}, "label_col_missing"

    label_appearances = df["label"].value_counts()
    for i in range(len(label_appearances)):
        if str(label_appearances.index[i])[:label_depth] not in label_dict.keys():
            label_dict[str(label_appearances.index[i])[:label_depth]] = [file]
        else:
            label_dict[str(label_appearances.index[i])[:label_depth]].append(file)
    return label_dict, None


def create_labels_overview_dict(file_with_params: dict):
    """counts how many occurrences each label group has and which file belongs to which label group 
    """
    label_depth = file_with_params.get("label_depth")
    file = file_with_params.get("file")

    if label_depth not in [1, 2, 3]:
        logger.error("received invalid label_depth")
        return "invalid_label_depth_given"

    classes_to_files_dict = {"classes_count": {}, "classes_files": {}}
    if file is None:
        return "no_file_given"
    if not os.path.exists(file):
        return "file_does_not_exist"

    df = pd.read_csv(file, usecols=["label"], dtype=np.int16)
    classes_to_files_dict["classes_count"], error = update_label_distributions(classes_to_files_dict["classes_count"], df, label_depth)
    if error:
        return error

    classes_to_files_dict["classes_files"], error = update_classes_files(classes_to_files_dict["classes_files"], df, label_depth, file)
    if error:
        return error

    return classes_to_files_dict


def calculate_needed_extra_samples(labels_overview_dict: dict, method: str, hierarchical: bool):
    """Calculates how much extra samples are needed for each class with the chosen method
    """
    ref_class = find_reference_class(labels_overview_dict, method, hierarchical)
    if hierarchical:
        return calculate_needed_extra_samples_hierarchical_data(labels_overview_dict)
    return calculate_differences(ref_class, labels_overview_dict)


def find_reference_class(labels_overview_dict: dict, method: str, hierarchical: bool):
    """Returns a list of the name and the amount of observations of the reference class
    """
    if not labels_overview_dict:
        logger.error(f"reference class not defined, when no classes exist")
        return []

    biggest_class = ["", 0]
    smallest_class = ["", 1000000000]
    for key in labels_overview_dict["classes_count"].keys():
        if labels_overview_dict["classes_count"][key] > biggest_class[1]:
            biggest_class = [key, labels_overview_dict["classes_count"][key]]
        if labels_overview_dict["classes_count"][key] < smallest_class[1]:
            smallest_class = [key, labels_overview_dict["classes_count"][key]]

    if method == "balancing_over":
        return biggest_class

    return smallest_class


def calculate_differences(ref_class: list, labels_overview_dict: dict):
    """ Returns a dict containing how much additional samples are needed for each label class
    """
    if ref_class == []:
        logger.error(f"Calculation of differences not possible when the reference class is undefined")
        return {}

    if not labels_overview_dict:
        logger.error("Cannot calculate the differences when the dict with labels and their appearance is empty")
        return {}

    differences = {}
    for key in labels_overview_dict["classes_count"].keys():
        differences[key] = ref_class[1] - labels_overview_dict["classes_count"][key]
    return differences


def adjust_granularity_of_label(df: pd.DataFrame, colname: str, label_depth: int):
    """returns a df which has the same granularity defined by the label_depth in the given col
    """
    if label_depth not in [1, 2]:
        logger.error(f"Adjusting granularity for this label_depth {label_depth} not implemented yet")
        return None, "invalid_label_depth given"

    cutted_label = np.where(df[colname].to_numpy() < 100, df[colname].to_numpy(), df[colname].to_numpy()/10).astype(int)
    if label_depth == 1:
        cutted_label = np.divide(cutted_label, 10).astype(int)

    df[colname] = cutted_label

    if 32 in df[colname].unique():
        df[colname] = 31

    return df, None


def calculate_prop_original_data_dict(needed_samples: dict, labels_overview_dict: dict):
    """ Calculates a dict containing how big the proportion of original data of the each class on the data of the class after resampling will be
    """
    if not isinstance(needed_samples, dict) or not isinstance(labels_overview_dict, dict):
        logger.error(f"Got invalid data given, got needed_samples of type {type(needed_samples)} and labels_overview of type {type(labels_overview_dict)} . Expected dicts")
        return {}
    if not set(needed_samples.keys()).issubset(labels_overview_dict["classes_count"].keys()):
        logger.error(f"The keys of needed_samples and the classes observed in the labels_overview_dict differ.")
        return {}

    # As the top-level class driving is excluded, skip the processing of this class
    # This list will be extended, once finer classification is tried
    labels_to_skip = [13, 14, 24, 4, 41, 42]
    # Initialize a dict for the storing of the calculated proportions
    prop_original_data_dict = {"class": [],
                               "prop_existing_after_balancing": []}
    for key in sorted([int(x) for x in needed_samples.keys()]):
        if key in labels_to_skip:
            continue
        # Calculate, how much of the data after resampling would be the currently seen data (A low number indicates a high amount of necessary resampling)
        prop_exis = labels_overview_dict.get("classes_count").get(str(key)) / (labels_overview_dict.get("classes_count").get(str(key)) + needed_samples[str(key)])
        # Append the calculated proportions and the respective class to the dict
        prop_original_data_dict.get("prop_existing_after_balancing").append(round(prop_exis, 2))
        prop_original_data_dict.get("class").append(key)

    return prop_original_data_dict


def dangerously_much_oversampling_needed(prop_original_data_dict: dict, warning_boundary=0.9):
    """ Iterates over the given dict and warns the user if too much resampling is needed
    """
    # As much needed resampling equates with having less information about the class in the data and
    # a strong dependence of the resulting cost function on very few datapoints, warning the user when
    # trying to perform great amounts of resampling on certain classes is necessary
    if not isinstance(prop_original_data_dict, dict):
        logger.error(f"Malformed prop_original_data_dict given: Expected dict, got {type(prop_original_data_dict)} ")
        return
    if not set(["prop_existing_after_balancing"]).issubset(prop_original_data_dict.keys()):
        logger.error(f"Malformed prop_original_data_dict given: key prop_existing_after_balancing not found")
        return

    for i in range(len(prop_original_data_dict.get("prop_existing_after_balancing"))):
        if prop_original_data_dict.get("prop_existing_after_balancing")[i] < warning_boundary:
            logger.warning(f"Dangerously much resampling needed for class {prop_original_data_dict.get('class')[i]} . More sophisticated Resampling might be necessary")
            return True

    return False


def naive_oversampling(needed_samples: dict, key: str, df_to_resample: pd.DataFrame, seed=42):
    """ A naive, yet fast oversampling algorithm
    """
    if df_to_resample.empty:
        return pd.DataFrame(), "oversampling_not_possible_for_empty_df"

    # Calculate how much rows the final oversampled df for this key should have
    optimal_length = len(df_to_resample) + needed_samples.get(key)

    # The following is my own implementation of oversampling to ensure quick convergence and minimal resampling error
    iteration = 0
    while len(df_to_resample) < optimal_length:
        #  If the amount of needed samples is huge, simply append the df (multiple times) to itself
        if optimal_length >= (len(df_to_resample)*2):
            # The following allows for multiple concats simultaneously, if much resampling is needed
            factor = int((needed_samples.get(key) + len(df_to_resample))/len(df_to_resample))

            df_to_resample = pd.concat([df_to_resample]*factor).reset_index(drop=True)
        else:
            random_indexes = draw_random_indexes(optimal_length, len(df_to_resample), seed)

            # Append the drawn rows to the df
            df_to_resample = pd.concat([df_to_resample, df_to_resample.iloc[random_indexes]]).reset_index(drop=True)

        iteration += 1

        # To prevent endless looping set a boundary for stopping.
        if iteration > 15:
            logger.error(f"Oversampling not finished after {iteration} iterations. Investigation needed.")
            return pd.DataFrame(), "too_much_iterations_needed_for_resampling"

    return df_to_resample.reset_index(drop=True), None


def perform_naive_oversampling(key_with_params: dict):
    """ performs naive oversampling and saves each df containing one label if specified
    """
    key = key_with_params.get("key")
    needed_samples = key_with_params.get("needed_samples")
    labels_overview_dict = key_with_params.get("labels_overview_dict")
    label_depth = key_with_params.get("label_depth")
    store_local = key_with_params.get("store_local")
    saving_folder = key_with_params.get("saving_folder")
    seed = key_with_params.get("seed")

    # Load all the data of a single label into a dataframe: these df are concat to df_to_resample
    df_to_resample = pd.DataFrame()
    saved_files = 0

    # test if all the key lead to a meaningful list of files
    for potential_class in labels_overview_dict["classes_files"].keys():
        if labels_overview_dict["classes_files"][potential_class] == []:
            return [data_utils.create_result_string("no_file_found", [1], saved_files, f"classes_files_did_not_contain_files_for_key_{potential_class}")]

    all_files_to_read_in = labels_overview_dict["classes_files"][key]

    # As joining all the files of one label into one df and oversampling  this df might cause too much strain on memory, the data is chunkswise oversampled
    # Get all the filechunks
    list_of_filchunks, error = file_utils.list_of_maxsized_filechunks(all_files_to_read_in, 0.5)
    if error:
        return [data_utils.create_result_string(list_of_filchunks[0], [1], saved_files, error)]

    # Calculate the number of needed samples for a chunk
    needed_samples_for_chunk = {key: math.ceil(needed_samples[key]/len(list_of_filchunks))}
    saved_needed_samples = 0

    logger.info(f"Starting the processing of files of key {key}")

    for number, chunk in enumerate(list_of_filchunks):
        with Pool() as pool:
            result = pool.map(file_utils.read_csv_safely, chunk)

        df_to_resample = pd.concat(result)

        # As the resolution of the df label may be higher than expected by the key, downsizing the label might be needed
        df_to_resample, error = adjust_granularity_of_label(df_to_resample, "label", label_depth)
        if error:
            return [data_utils.create_result_string(chunk[0], [1], saved_files, error)]

        # Check if the loaded df does contain only one label class
        if len(df_to_resample["label"].unique()) > 1:
            logger.error(f"found multiple labels in the chunk: {df_to_resample['label'].unique()}")
            return [data_utils.create_result_string(chunk[0], [1], saved_files, "multiple_label_values_found")]

        # Get hold of the number of samples before oversampling (used to find how much of the needed samples have been sampled)
        nr_samples_before_oversampling = len(df_to_resample)

        # In the last iteration the number of needed samples may need adjustment
        if store_local and number + 1 == len(list_of_filchunks):
            needed_samples_for_chunk = {key: needed_samples[key] - saved_needed_samples}

        # oversample the data
        df_resampled, error = naive_oversampling(needed_samples_for_chunk, key, df_to_resample, seed)
        if error:
            return [data_utils.create_result_string(chunk[0], [1], saved_files, error)]

        # overwrite all resting subactivities with resting, as there is only one resting class in the baseline model
        if 32 in df_resampled["label"].unique():
            df_resampled["label"] = 31

        if store_local:
            # This is needed, as multiple cores might command the creation of the folder simultaneously
            if not os.path.exists(saving_folder):
                os.makedirs(saving_folder, exist_ok=True)

            # Save each balanced_df under an own file
            error = file_utils.save_df_with_key(chunk[0], saving_folder, df_resampled, key)
            if error:
                return data_utils.create_result_string(chunk[0], [1], saved_files, error)
            saved_needed_samples += (len(df_resampled) - nr_samples_before_oversampling)

    # As all files of the given key have been processed successfully, this needs to be reflected in the list_of_result strings
    list_of_result_strings = [data_utils.create_result_string(successfully_processed_file, [None], 1, "None") for successfully_processed_file in labels_overview_dict["classes_files"][key]]

    return list_of_result_strings


def remove_unneeded_keys(reduced_keys_as_list: list, dict_to_reduce: dict):
    """removes all keys from dict, which are not in given list
    """
    keys_to_remove = []
    for key in dict_to_reduce.keys():
        if key not in reduced_keys_as_list:
            keys_to_remove.append(key)

    for key_to_remove in keys_to_remove:
        dict_to_reduce.pop(key_to_remove)

    return dict_to_reduce, keys_to_remove


def create_dict_params(all_unbalanced_files: list, label_depth: int):
    """ returns a list of dicts with the params for the creation of files
    """
    return [{"file": x, "label_depth": label_depth} for x in all_unbalanced_files]


def update_overall_labels_overview_dict(result: list, previous_dict: dict):
    """ uses the information from the dicts in result to update the overall_labels_overview_dict
    """
    overall_labels_overview_dict = previous_dict
    occurred_errors = []

    for number, labels_over_view_dict in enumerate(result):
        # When an error ocurred, stop to enable investigation
        if isinstance(labels_over_view_dict, str):
            occurred_errors.append(labels_over_view_dict)
            continue

        # Otherwise expand the overall_labels_overview_dict
        for key in labels_over_view_dict:
            if key not in overall_labels_overview_dict.keys():
                overall_labels_overview_dict[key] = {}

            for subkey in labels_over_view_dict[key]:
                if subkey not in overall_labels_overview_dict[key].keys():
                    overall_labels_overview_dict[key][subkey] = labels_over_view_dict[key][subkey]
                else:
                    if isinstance(overall_labels_overview_dict[key][subkey], int):
                        overall_labels_overview_dict[key][subkey] += labels_over_view_dict[key][subkey]
                    else:
                        overall_labels_overview_dict[key][subkey].append(labels_over_view_dict[key][subkey][0])

    if occurred_errors:
        return occurred_errors

    return overall_labels_overview_dict


def key_with_params(needed_samples: dict, labels_overview_dict: dict, label_depth: int, store_local: bool, saving_folder: str, seed: int):
    """append the for the processing needed information
    """
    keys_with_params = []
    for key in needed_samples.keys():
        keys_with_params.append({"key": key,
                                 "labels_overview_dict": labels_overview_dict,
                                 "label_depth": label_depth,
                                 "store_local": store_local,
                                 "saving_folder": saving_folder,
                                 "needed_samples": needed_samples,
                                 "seed": seed
                                 })
    return keys_with_params


def create_list_of_file_dicts(all_unreduced_files: list, table: dict):
    """ Creates a list with dicts of the file_name for one file for each file and parameters needed processing 
    """

    list_of_file_dicts = []
    for file in all_unreduced_files:
        list_of_file_dicts.append(
            {"file": file,
             "needed_columns": table.get("needed_columns"),
             "label_depth": table.get("label_depth"),
             "store_local": not table.get("dryrun"),
             "saving_folder": table.get("saving_folder")})

    return list_of_file_dicts


def reduce_data_of_file(file_with_params: dict):
    """Reduces the data of one file
    """
    # create the needed variables
    file = file_with_params.get("file")
    needed_columns = file_with_params.get("needed_columns")
    label_depth = file_with_params.get("label_depth")
    store_local = not file_with_params.get("dryrun")
    saving_folder = file_with_params.get("saving_folder")

    # Read in the file
    if not os.path.exists(file):
        return data_utils.create_result_string(file, [1], 0, "file_not_found")
    df = pd.read_csv(file)

    # initialize an object containing the amount of saved_windows
    saved_windows = 0

    if "label" not in df.columns:
        logger.error(f"got file which does not contain a label col")
        return data_utils.create_result_string(file, [1], saved_windows, "label_col_not_found")

    more_general_label_class = str(df["label"][0])[0]
    # Driving data is skipped, as there is no model  classifying it
    if label_depth == 2 and more_general_label_class.startswith("4"):
        return data_utils.create_result_string(file, [None], saved_windows, "None")

    df_to_save, error = correct_format_for_multilevel_classification(df, needed_columns)
    if error:
        data_utils.create_result_string(file, [1], saved_windows, error)

    if store_local:
        
        error = file_utils.save_df_with_key(file, saving_folder, df_to_save)
        if error:
            return data_utils.create_result_string(file, [1], saved_windows, error)
        saved_windows += 1

    # If everything was valid, return the fitting result string
    return data_utils.create_result_string(file, [None], saved_windows, "None")


def count_label_distribution(all_files: list, label_depth: int):
    """returns a list with the amount of seen windows per class, as well as a dict mapping from label to files
    """
    list_of_file_dicts = create_dict_params(all_files, label_depth)

    # Processing in chunks enables the possibility to stop the program and immediately investigate errors, if any occur
    file_chunks = data_utils.split_files_to_handle_into_chunks(list_of_file_dicts, 3072)

    # An object is needed, in which the results from the different cores are being gathered
    labels_overview_dict = {}

    logger.info(f"Started counting the seen labels in all the {len(all_files)} files...")
    # The following is a faster way calculate the overall appearance of labels
    for chunk_number, chunk in enumerate(file_chunks, start=1):
        if chunk_number % 10 == 0 or chunk_number == 1:
            logger.info(f"[{chunk_number}/{len(file_chunks)}] Started counting label in chunk")

        with Pool() as pool:
            result = pool.map(create_labels_overview_dict, chunk)
        labels_overview_dict = update_overall_labels_overview_dict(result, labels_overview_dict)
        if isinstance(labels_overview_dict, list):
            logger.warning(f"These errors occurred: {labels_overview_dict}")
            return

    # As there is no resting subanalyzer, all the resting data is gathered in one class
    if label_depth == 2:
        if "32" in labels_overview_dict["classes_count"].keys():
            labels_overview_dict["classes_count"]["31"] += labels_overview_dict["classes_count"].pop("32")
            labels_overview_dict["classes_files"]["31"] += labels_overview_dict["classes_files"].pop("32")

    return labels_overview_dict


def remove_labels_to_skip(dict_with_labels: dict):
    """remove all the labels to skip from the given dict (currently contains all the labels to skip)
    """
    removed_keys = []
    labels_to_skip = [13, 14, 24, 4, 41, 42]
    labels_to_skip.extend([str(x) for x in labels_to_skip])
    for key in labels_to_skip:
        if key in [x for x in dict_with_labels.keys()]:
            dict_with_labels.pop(key)
            removed_keys.append(key)

    return dict_with_labels, removed_keys


def create_subdicts_labels_overview_dict(label_depth: int, labels_overview_dict: dict, log_file:dict):
    """ Uses the label depth and the labels_overview_dict to generate the needed subdicts for the oversampling
    """
    if label_depth == 1:
        return [labels_overview_dict], None
    if label_depth == 2 and file_utils.processing_featured_data(log_file):
        walking_subdict = {"classes_count": {}, "classes_files": {}}
        lifting_subdict = {"classes_count": {}, "classes_files": {}}
        for key in labels_overview_dict:
            for subkey in labels_overview_dict[key]:
                if str(subkey).startswith("1"):
                    lifting_subdict[key][subkey] = labels_overview_dict[key][subkey]
                elif str(subkey).startswith("2"):
                    walking_subdict[key][subkey] = labels_overview_dict[key][subkey]
                else:
                    logger.error(f"received invalid subkey {subkey}. No submodel exists for this more_general_class")
        return [lifting_subdict, walking_subdict], None

    if label_depth == 2 and file_utils.processing_flattened_data(log_file):
        # As the baseline model has only one resting class, joining the two classes is needed
        labels_overview_dict["classes_files"]["31"].extend(
            labels_overview_dict["classes_files"]["32"])
        labels_overview_dict["classes_count"]["31"] += labels_overview_dict["classes_count"]["32"]
        labels_overview_dict["classes_count"].pop("32")
        labels_overview_dict["classes_files"].pop("32")

        return [labels_overview_dict], None

    logger.error(f"received not yet implemented label_depth")
    return [], "given_label_depth_not_yet_implemented"


def split_df_to_handle_into_chunks(df: pd.DataFrame, chunksize=24):
    """Splits a df into a list of smaller dfs
    """
    return [df.iloc[i:i + chunksize, :].reset_index(drop=True) for i in range(0, len(df), chunksize)]


def create_list_of_names(folder: str, prefix: str, process: str, file_ending: str, length_of_list: int, hierarchical_featured: bool, list_of_specific_prefixes=None, window_uuid = []):
    """creates a list of names of the form <folder>/<prefix>_<process>_<uuid>.file_ending
    """
    if hierarchical_featured:
        list_of_specific_prefixes = [int(str(prefix)[0]) if x % 2 == 0 else prefix for x in range(length_of_list)]
        return [os.path.join(folder, f"{list_of_specific_prefixes[x]}_{process}_{window_uuid[x]}.{file_ending}") for x in range(length_of_list)], None

    if list_of_specific_prefixes:
        if len(list_of_specific_prefixes) != length_of_list:
            return [], "length_of_list_of_specific_prefixes_does_not_match_len_other_list"
        return [os.path.join(folder, f"{list_of_specific_prefixes[x]}_{process}_{window_uuid[x]}.{file_ending}") for x in range(length_of_list)], None
    return [os.path.join(folder, f"{prefix}_{process}_{window_uuid[x]}.{file_ending}") for x in range(length_of_list)], None


def save_in_subfiles(df_with_params: dict):
    """Uses a df, to save the individual rows as csv file
    """
    df: pd.DataFrame = df_with_params.get("df")
    saving_folder: str = df_with_params.get("saving_folder")
    label: int = df_with_params.get("label")
    grid: bool = df_with_params.get("grid")
    hierarchical_featured: bool = df_with_params.get("hierarchical_featured")
    window_uuid: list = df_with_params.get("window_uuid")
    saved_windows = 0

    # Create the folder if it does not exist yet, ignore errors caused by multiprocessing
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder, exist_ok= True)

    if df.empty:
        return data_utils.create_result_string("some_file", [1], saved_windows, "got_empty_df")
    if len(df) == 0:
        return data_utils.create_result_string("some_file", [1], saved_windows, "no_data_in_df")
    if not window_uuid:
        return data_utils.create_result_string("some_file", [1], saved_windows, "no_window_uuid_given")
  

    # These names need to be created beforehand, as the specific uuid needs to be known
    list_of_names, error = create_list_of_names(saving_folder, label, "prepared", "csv", len(df), hierarchical_featured, None, window_uuid)
    if error:
        return data_utils.create_result_string("some_file", [1], saved_windows, error)

    cols_do_drop = ["label", 'window_uuid']
    for col_to_drop in cols_do_drop:
        if col_to_drop in df.columns:
            df.drop(col_to_drop, axis=1, inplace=True)

    [save_file(df.iloc[i, :], grid, list_of_names[i], hierarchical_featured) for i in range(len(df))]
    saved_windows += len(df)
    return [data_utils.create_result_string(f"{name}", [], saved_windows, str(error)) for name in list_of_names]


def handle_labels_df(result_list: list, df_labels: pd.DataFrame, separator="+++"):
    """handles the result_list to create the the labels.csv 
    """
    filenames = []
    labels = []
    result_list = file_utils.unpack_list_of_result_strings_if_needed(result_list)

    for result in result_list:
        result_string_splitted = result.split(separator)
        success = result_string_splitted[1] == "True"
        if success:
            filenames.append(result_string_splitted[0])
            labels.append(int(result_string_splitted[0].split("/")[-1].split("_")[0]))
        else:
            logger.warning(f"Found errors: {result}")

    return pd.concat([df_labels, pd.DataFrame({"filename": filenames, "label": labels})]).reset_index(drop=True)


def prepare_folder_for_custom_loading(folder: str, table: dict, log_file: dict, method="prepare_dataset", grid=False):
    """Brings files in folder into the correct format for using them as a custom pytorch dataset
    """
    all_unprepared_files, error = file_utils.get_files(folder, "csv")
    if error:
        return pd.DataFrame(), {}, error

    # The labels df is needed for the construction of the pytorch Dataset class
    labels_df = pd.DataFrame()

    processing_hierarchical_featured_data = "generate_features" in log_file.keys() and table.get("granularity") == "mid"

    # Handle each file after the other (slower, but may save RAM)
    for number, file in enumerate(all_unprepared_files, start=1):
        df = file_utils.read_csv_safely(file)
        list_of_sub_dfs = split_into_file_chunks(df, processing_hierarchical_featured_data, table)
        label = correct_label_for_saving_prepared_file(file, processing_hierarchical_featured_data, df)
        
        list_of_dfs_and_savingplace = []
        for sub_df in list_of_sub_dfs:
            window_uuids = []
            if "window_uuid" in sub_df.columns:
                window_uuids = sub_df['window_uuid'].to_list()
                sub_df.drop("window_uuid", axis=1, inplace=True)
            df_with_params = {"df": sub_df,
                              "saving_folder": folder,
                              "label": label,
                              "grid": grid,
                              "window_uuid": window_uuids,
                              "hierarchical_featured": processing_hierarchical_featured_data}
            list_of_dfs_and_savingplace.append(df_with_params)
        del list_of_sub_dfs

        with Pool() as pool:
            result = pool.map(save_in_subfiles, list_of_dfs_and_savingplace)

        log_file = file_utils.create_log_from_result_strings(result, os.path.join(folder, "log.json"), method)
        if len(all_unprepared_files) > 1:
            logger.info(f"[{number}/{len(all_unprepared_files)}] files finished. {len(log_file[method]['successfully_processed_files'])} files successfully processed")

        # If errors were found, warn the user and save the log file for data investigation purposes
        if len(log_file[method]["occurred_errors"]) != 0:
            # Inform the user
            logger.warning(f"These errors occurred: {log_file[method]['occurred_errors'].keys()}")
            logger.info(f"For debugging purposes the log file gets saved: Contains errors & incorrect files")
            # Update the log file
            error = file_utils.update_log_file(log_file, table, folder, method)
            if error:
                logger.error(f"saving log file failed: {error}")
                return pd.DataFrame(), log_file, error

        # Save the log file
        if not log_file[method].get("dryrun"):
            # Update the log file
            error = file_utils.update_log_file(log_file, table, folder, method)
            if error:
                logger.error(f"saving log file failed: {error}")
                return pd.DataFrame(), log_file, error

        labels_df = handle_labels_df(result, labels_df)

        # remove the original file
        os.remove(file)

    # Shuffle and save the labels.csv
    labels_df = labels_df.sample(frac=1, random_state=table.get("seed")).reset_index(drop=True)
    labels_df.to_csv(os.path.join(folder, "labels.csv"), index=False)
    return labels_df, log_file, None


def remove_processed_key_with_params(keys_with_params: list, log_file: dict, method: str):
    """removes a key from the keys_with_params if all the files of the respective key have been processed
    """
    already_handled_files = log_file[method]["successfully_processed_files"]

    key_with_params_to_remove = []
    for key_with_params in keys_with_params:
        files_to_handle_for_this_key = key_with_params.get("labels_overview_dict").get("classes_files").get(key_with_params.get("key"))
        if set(files_to_handle_for_this_key).issubset(set(already_handled_files)):
            key_with_params_to_remove.append(key_with_params)

    for key_with_params in key_with_params_to_remove:
        keys_with_params.remove(key_with_params)

    return keys_with_params


def valid_table(table: dict):
    """tests if the given table contains valid arguments
    """
    if "data_path" not in table.keys():
        logger.error(f"No data_path given")
        return "no_data_path_given"

    if not os.path.exists(table.get("data_path")):
        logger.error(f"Given data_path {table.get('data_path')} does not exist")
        return "datapath_does_not_exist"

    error = file_utils.valid_anonymize_file_found(table.get("data_path"))
    if error:
        return error

    if "dryrun" not in table.keys():
        logger.error(f"No dryrun given")
        return "no_dryrun_given"

    if not isinstance(table.get("dryrun"), bool):
        logger.error(f"Got dryrun of nonbool datatype")
        return "got_dryrun_of_nonbool_datatype"

    if "input_data" not in table.keys():
        logger.error(f"No input_data folder given")
        return "no_input_data_folder_given"

    if not os.path.exists(table.get("input_data")):
        logger.error(f"Input_data folder does not exist: {table.get('input_data')}")
        return "input_data_folder_does_not_exist"

    _, error = file_utils.get_files(table.get("input_data"), "csv")
    if error:
        logger.error(f"input_data folder does not contain any csv files ")
        return error

    if table.get("output_date") is not None:
        error = time_utils.valid_time_string(table.get("output_date"))
        if error:
            return error

    if table.get("granularity") == "low":
        logger.error(f"Granularity level low is not implemented yet")
        return "granularity_level_low_is_not_implemented_yet"

    if not isinstance(table.get("plot"), bool):
        logger.error(f"Got plot of nonbool datatype")
        return "got_plot_of_nonbool_datatype"

    if table.get("method") == "reduce_data":
        with open(os.path.join(table.get("input_data"), "log.json")) as f:
            log_file = json.load(f)
        if not file_utils.processing_featured_data(log_file):
            logger.error(f"Reducing set of features not possible when input data does not exist in featured form")
            return "got_non_featured_input_data_for_method_reduce_data"

    if table.get("method") == "prepare_dataset":
        with open(os.path.join(table.get("input_data"), "log.json")) as f:
            log_file = json.load(f)
        if not file_utils.processing_balanced_data(log_file):
            logger.error(f"Preparing dataset currently only implemented for already balanced data")
            return "got_unbalanced_set_for_training"

    if not isinstance(table.get("grid"), bool):
        logger.error(f"Got grid of type {type(table.get('grid'))}, expected bool")
        return "got_grid_of_non_bool_dt"

    if table.get("method") == 'balancing_evaluation' and len(set(table.get("classes_to_aggregate"))) == 1 :
        logger.error("Chose to aggregate labels, at least two different labels need to be given")
        return "Nr_classes_for_aggregation_given_too_small"

    return None


def handle_data_for_feature_based_ffnn(input_data_folder: str, saving_folder: str, method: str, log_file: dict, label_depth: int, table: dict, store_local: bool):
    """does the heavy lifting to handle the files for training a FFNN on the data in featured format
    """
    all_unprepared_files, error = file_utils.get_files(input_data_folder, "csv")
    if error:
        logger.error(f"The input data folder does not contain any files")
        return error
    # Check if the data sizes are valid
    size_of_all_files, error = file_utils.count_filesizes_of_dir(input_data_folder)
    if error:
        return error
    if size_of_all_files > 4:
        logger.error(f"received unexpectedly large amount of featured data in input_folder")
        return

    all_unprepared_files, error = file_utils.reduce_files_to_handle(all_unprepared_files, saving_folder, method)
    if error or all_unprepared_files == []:
        return error

    # use multiprocessing to read_in all the balanced files safely
    with Pool() as pool:
        result = pool.map(file_utils.read_csv_safely, all_unprepared_files)

    all_data_df = pd.concat(result).reset_index(drop=True)
    del result

    if label_depth == 2 and "balancing_over" in log_file.keys():
        if not valid_hierarchical_oversampling_seen(all_data_df["label"]):
            return

    # save the original labels  in the log file: Needed for later identification which data was processed
    original_labels = sorted([int(x) for x in all_data_df["label"].unique()])
    log_file[method]["original_labels"] = original_labels
    logger.info(f"These were the original labels of the currently processed files: {original_labels}")

    # Create the new labels: prepare_label for usage in train_models: Map meaningful label nrs to ints in range(1, nr_classes+1)
    all_data_df, error = data_utils.prepare_label_for_featured_data(all_data_df, label_depth)
    if error:
        return error

    all_data_df, error = data_utils.naively_impute_if_needed(all_data_df, log_file)
    if error:
        return error

    if "index_col" in all_data_df.columns:
        all_data_df = all_data_df.sort_values("index_col").reset_index(drop=True)
        all_data_df = all_data_df.drop(columns=["index_col"])
    if "window_uuid" in all_data_df.columns:
        all_data_df['window_uuid'] = [window_uuid.split(".csv")[0] for window_uuid in all_data_df['window_uuid']]
    error = file_utils.save_csvs_labelwise(all_data_df, saving_folder)
    if error:
        return error

    logger.info("Saved the prepared_df into labelwise files, proceeding to save featured windows as individual files")

    df_labels, log_file, error = prepare_folder_for_custom_loading(saving_folder, table, log_file, method, False)
    if error:
        return error

    # Shuffle, to ensure that the resulting batches do not contain clumped data
    labels_df = df_labels.sample(frac=1, random_state=table.get("seed")).reset_index(drop=True)

    labels_df.to_csv(os.path.join(saving_folder, "labels.csv"), index=False)
    logger.info("Successfully saved rows of each label as individual files")

    if store_local:
        if len(log_file[method]["occurred_errors"]) == 0:
            log_file[method]["successfully_processed_files"] = "all"

        error = file_utils.update_log_file(log_file, table, saving_folder, method)
        if error:
            return error

    return None


def prepare_dataset_for_flattened_data(input_data_folder: str, saving_folder: str, method: str, log_file: dict, label_depth: int, table: dict, store_local: bool, grid: bool):
    """does the heavy lifting to handle the files for training a NN on the flattened data (prepares either for FFNN or CNN depending on grid)
    """
    all_unprepared_files, error = file_utils.get_files(input_data_folder, "csv")
    if error:
        logger.error(f"The input data folder does not contain any files")
        return error

    all_unprepared_files, error = file_utils.reduce_files_to_handle(all_unprepared_files, saving_folder, method)
    if error or all_unprepared_files == []:
        return error

    original_labels, error = file_utils.get_unique_levels_from_folder(input_data_folder, "label", np.int16, "csv")
    if error:
        return error
    log_file[method]["original_labels"] = original_labels
    logger.info(f"These were the original labels of the currently processed files: {original_labels}")

    _, error = file_utils.estimate_needed_time(len(all_unprepared_files), method, os.cpu_count(), file_utils.count_filesizes_of_list(all_unprepared_files))
    if error:
        return error

    # Create dict containing the remapping of the label
    label_remapping = data_utils.create_label_remapping(original_labels)
    logger.info(f"This is the label remapping: {label_remapping}")

    logger.info(f"Starting to save the files labelwise")
    for number, file in enumerate(all_unprepared_files, start=1):
        df = file_utils.read_csv_safely(file, header=0)

        # bring the labels into the correct format (range & precision)
        df["label"], error = data_utils.prepare_label(df["label"], label_depth, label_remapping)
        if error:
            return error

        df, error = data_utils.naively_impute_if_needed(df, log_file)
        if error:
            return error
        
        if "window_uuid" in df.columns:
            df['window_uuid'] = [window_uuid.split(".csv")[0] for window_uuid in df['window_uuid']]

        error = file_utils.save_csvs_labelwise(df, saving_folder)
        if error:
            return error

    logger.info(f"Saving the files label-wise finished, starting to save window-wise")
    df_labels, log_file, error = prepare_folder_for_custom_loading(saving_folder, table, log_file, method, grid)
    if error:
        return error

    # Shuffle, to ensure that the resulting batches do not contain clumped data
    labels_df = df_labels.sample(frac=1, random_state=table.get("seed")).reset_index(drop=True)

    labels_df.to_csv(os.path.join(saving_folder, "labels.csv"), index=False)
    logger.info(f"Successfully prepared data for training a neural network")

    if store_local:
        if len(log_file[method]["occurred_errors"]) == 0:
            log_file[method]["successfully_processed_files"] = "all"

        error = file_utils.update_log_file(log_file, table, saving_folder, method)
        if error:
            return error

    return None


def return_df_channelwise_if_needed(df: pd.DataFrame, grid: bool, nr_channels=9):
    """returns a version of the df, where each row contains one input channel, if convolution == True
    """
    if not grid:
        return df
    return pd.DataFrame(df.to_numpy().reshape(-1, nr_channels).T)


def correct_format_for_multilevel_classification(df: pd.DataFrame, needed_columns: list):
    """returns a version of the df suited for the training of a hierarchical model
    """
    feature_list_top_level = needed_columns[0]
    if not set(feature_list_top_level).issubset(set(df.columns)):
        return pd.DataFrame(), "not_all_needed_cols_present_in_df"
    label = str(df["label"][0])
    if len(needed_columns) == 1 or label.startswith("3"):
        return df[feature_list_top_level], None

    feature_list_second_level = []

    if label.startswith("1"):
        feature_list_second_level = needed_columns[1]
    elif label.startswith("2"):
        feature_list_second_level = needed_columns[2]
    else:
        logger.error("Got a file with a label neither starting with 1 or 2 for mid-level classification")
        return pd.DataFrame(), "unexpected_label_seen"

    if len(feature_list_top_level) - len(feature_list_second_level) < 0:
        logger.error("received more features for sub-level analyzer ")
        return pd.DataFrame(), "wrong_feature_amount_for_sub_level_ananlyzer"

    first_row = df[feature_list_top_level]
    second_row = df[feature_list_second_level]

    final_df = pd.concat([first_row, second_row], axis=0).reset_index(drop=True)

    return final_df, None


def split_into_file_chunks(df: pd.DataFrame, processing_hierarchical_featured_data: bool, table: dict):
    """As the filechunks need to be handled differently for hierarchical data and unhierarchical, the chunks need to be made accordingly
    """
    if not processing_hierarchical_featured_data or table.get("label_depth") == 1:
        return split_df_to_handle_into_chunks(df, 1)

    seeing_only_top_level_data = not df.isna().any().any()

    if seeing_only_top_level_data:
        return split_df_to_handle_into_chunks(df, 1)

    return split_df_to_handle_into_chunks(df, table.get("label_depth"))


def correct_label_for_saving_prepared_file(file: str, processing_hierarchical_featured_data: bool, df: pd.DataFrame):
    """ returns the correct label under which to save the prepared file
    """
    file = file.replace("\\","/")
    if processing_hierarchical_featured_data:
        return int(str(int(df.iloc[0, len(df.columns)-2]))+str(int(df.iloc[0, len(df.columns)-1])))

    return int(file.split("/")[-1].split("_")[0])


def save_file(col: pd.Series, grid: bool, name: str, hierarchical_featured: bool):
    """
    """
    # Drop the last 2 cols, as they contain the top-level and the mid-level label and must not be passed as input
    if hierarchical_featured:
        col = col[:len(col)-2].dropna()

    return_df_channelwise_if_needed(col, grid).to_csv(
        name, index=False, header=False)


def calculate_needed_extra_samples_hierarchical_data(labels_overview_dict: dict):
    """calculates the needed samples, so that the top-level model and all sub_level models have balanced data
    """
    dict_of_analyzer_levels = create_two_level_hierarchical_dict_of_labels(
        labels_overview_dict["classes_count"].keys())

    # find the class which causes the most oversampling over the levels
    caused_samples_dict = {}
    for label in labels_overview_dict["classes_count"].keys():
        caused_samples_dict[label] = labels_overview_dict["classes_count"][label] * len(dict_of_analyzer_levels[label[0]]["subkeys"])

    max_samples = max([caused_samples_dict[key] for key in caused_samples_dict.keys()])
    if max_samples % 2 == 1:
        max_samples += 1

    needed_samples = {}
    for label in labels_overview_dict["classes_count"].keys():
        needed_samples[label] = int((max_samples / len(dict_of_analyzer_levels[label[0]] ["subkeys"])) - labels_overview_dict["classes_count"][label])

    return needed_samples


def create_two_level_hierarchical_dict_of_labels(list_of_labels: list):
    """maps each top-level-label to their subkeys (sublabels)
    """
    unique_list_of_labels = list(set(list_of_labels))
    dict_of_analyzer_levels = {}
    for top_level in [str(label)[0]for label in unique_list_of_labels]:
        dict_of_analyzer_levels[top_level] = {}
        for key in unique_list_of_labels:
            if str(key).startswith(top_level):
                if "subkeys" not in dict_of_analyzer_levels[top_level].keys():
                    dict_of_analyzer_levels[top_level]["subkeys"] = []
                dict_of_analyzer_levels[top_level]["subkeys"].append(key)

    return dict_of_analyzer_levels


def valid_hierarchical_oversampling_seen(label_series: pd.Series):
    """Tests if all the hierarchical data got balanced correctly
    """
    hierarchical_dict_of_samples, error = create_hierarchical_dict_of_samples(label_series)
    if error:
        return False
    for key in hierarchical_dict_of_samples.keys():
        if key == "top_level":
            seen_occurences = [hierarchical_dict_of_samples[key][subkey] for subkey in hierarchical_dict_of_samples[key].keys()]
            if len(set(seen_occurences)) != 1:
                logger.error(f"The top-level labels do not occur in the same (balanced) frecquency : {hierarchical_dict_of_samples[key]}")
                return False
            continue
        for subanalyzer in hierarchical_dict_of_samples[key].keys():
            seen_occurences = [hierarchical_dict_of_samples[key][subanalyzer][subkey] for subkey in hierarchical_dict_of_samples[key][subanalyzer].keys()]
            if len(set(seen_occurences)) != 1:
                logger.error(f"The data for the subanalyzer {subanalyzer} not occur in the same (balanced) frecquency : {hierarchical_dict_of_samples[key][subanalyzer]}")
                return False

    return True


def create_hierarchical_dict_of_samples(label_series: pd.Series):
    """Creates a dict containing the amount of samples seen per label in an hierarchical fashion separated into different models to be build
    """
    if len(str(max(label_series.unique()))) != len(str(min(label_series.unique()))):
        logger.error(f"received label series with different set of granularities: {len(str(max(label_series.unique())))} and {len(str(min(label_series.unique())))}")
        return {}, "different_granularity_levels_detected"

    levels_dict = {"top_level": {}, "mid_level": {}, "low_level": {}}

    counted_values = label_series.value_counts()

    for label in label_series.unique():
        if len(str(label)) >= 3:
            if int(str(label)[0:3]) not in levels_dict["low_level"].keys():
                levels_dict["low_level"][int(str(label)[0:2])] = {}
            levels_dict["low_level"][int(
                str(label)[0:2])][label] = counted_values[label]
        if len(str(label)) >= 2:
            if int(str(label)[0:1]) not in levels_dict["mid_level"].keys():
                levels_dict["mid_level"][int(str(label)[0:1])] = {}
                levels_dict["mid_level"][int(
                    str(label)[0:1])][int(str(label)[0:2])] = 0
            if int(str(label)[0:2]) not in levels_dict["mid_level"][int(str(label)[0:1])].keys():
                levels_dict["mid_level"][int(
                    str(label)[0:1])][int(str(label)[0:2])] = 0
            levels_dict["mid_level"][int(str(label)[0:1])][int(
                str(label)[0:2])] += counted_values[label]

        if int(str(label)[0]) not in levels_dict["top_level"].keys():
            levels_dict["top_level"][int(str(label)[0])] = 0
        levels_dict["top_level"][int(str(label)[0])] += counted_values[label]

    return levels_dict, ''


def draw_random_indexes(optimal_length: int, existing_length: int, seed: int):
    """draws random inxes for correct oversampling even for hierarchical data
    """
    np.random.seed(seed)

    if optimal_length - existing_length >= 2:
        random_indexes = np.unique(np.random.randint(int(existing_length/2), size=int((optimal_length - existing_length)/2)))*2
        random_indexes = np.concatenate([random_indexes, np.add(random_indexes, 1)])
        random_indexes.sort(0)
        return random_indexes

    return np.random.randint(existing_length, size=1)
