import os
import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from config.log import logger
from utils import plot_utils


def check_classifier_existence(classifier_folder: str, given_classifier: str, required_columns: list):
    """Checks if the given classifier folder does exist and contains all necessary contents
    """
    if not os.path.exists(classifier_folder):
        logger.error(f"The given folder for the {given_classifier} does not exist: {classifier_folder}")
        return "folder_not_found"
    if not os.path.exists(os.path.join(classifier_folder, "predictions.csv")):
        logger.error(f"No predictions.csv found in the {given_classifier} folder")
        return "predictions_not_found_in_top_level_folder_not_found"
    predictions_df = pd.read_csv(os.path.join(classifier_folder, "predictions.csv"), nrows=1)
    if not set(required_columns).issubset(set(predictions_df.columns)):
        logger.error(f"Not all needed columns are present within the cols of the predictions_df. Needed: {required_columns}")
        return "Needed_columns_missing"

    return None


def valid_table(table: dict):
    """Checks if the given parameters can be used for the plotting of classwise ROC curves
    """
    required_columns = ['file', 'true_label', 'probability_correct', 'probability_incorrect']

    error = check_classifier_existence(table.get("top_level_classifier_folder"), "top_level", required_columns)
    if error:
        return error
    
    if not table.get("lifting_classifier_folder") and not table.get("walking_classifier_folder"):
        return None
    
    if table.get("lifting_classifier_folder") and table.get("lifting_classifier_folder"):
        if len(set([table.get("lifting_classifier_folder"), table.get("walking_classifier_folder"), table.get("top_level_classifier_folder")])) != 3:
            logger.error(f"Received non-unique folders for the different models to evaluate")
            return "identical_folders_given"

        error = check_classifier_existence(table.get("lifting_classifier_folder"), "lifting", required_columns)
        if error:
            return error

        error = check_classifier_existence(table.get("walking_classifier_folder"), "walking", required_columns)
        if error:
            return error
        
        return None

    logger.error("When a hierarchically classifying model needs to be evaluated, botha classifier folder for the lifting and walking subclassifier need to be given")
    return "Not_all_needed_folders_given_for_hierarchically_classifying_folder"


def plot_ROC_curve(y_true: pd.Series, probas: list, store_local: bool, saving_folder: str, seen_class: str, fontsize:int = 15):
    """ Plots the ROC curve for a single class a 2 class classification problem
    """
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_true, probas)

    plt.rc('xtick', labelsize=fontsize) 
    plt.rc('ytick', labelsize=fontsize) 

    plt.subplots(1, figsize=(10,10))
    plt.title(f'Receiver Operating Characteristic Class:  {seen_class}')
    plt.plot(false_positive_rate1, true_positive_rate1)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    if store_local:
        plt.savefig(os.path.join(saving_folder, f"ROC_class_{seen_class}.png"))
    plt.show()


def plot_precision_recall_curve(y_true: pd.Series, probas:list, store_local: bool, saving_folder: str, seen_class: str):
    """Plots a performance plot which is more robust regarding heavy class imbalance
    """
    precision, recall, _ = precision_recall_curve(y_true, probas)
    # plot the precision-recall curves
    plt.subplots(1, figsize=(6,6))
    plt.plot(recall, precision, marker='.', label='DeepFFNet')
    # axis labels
    plt.xlabel('Recall', fontsize = 18)
    plt.ylabel('Precision', fontsize = 18)
    plt.title(f'Precision-Recall Curve: {seen_class}', fontsize = 20)
    plt.legend(loc = 'best', fontsize = 18)
    if store_local:
        plt.savefig(os.path.join(saving_folder, f"Precision_Recall_class_{seen_class}.png"))
    plt.show()


def plot_all_ROC_curves(probabilities_dict: dict, store_local: bool, saving_folder: str):
    """plots and saves all class-wise ROC curves 
    """
    human_readable_labels = plot_utils.get_class_names_from_len(len(probabilities_dict.keys()), "")
    human_readable_labels = [label.replace("\n", " ") for label in human_readable_labels]

    for label in probabilities_dict.keys():
        y_true = probabilities_dict[label]['true_label_two_class']
        probas: list = probabilities_dict[label][['probability_target_class']].values.tolist()
        plot_ROC_curve(y_true, probas, store_local, saving_folder, seen_class=human_readable_labels[int(label) -1])
        plot_precision_recall_curve(y_true, probas, store_local, saving_folder, seen_class=human_readable_labels[int(label) -1])

def classwise_evaluation_metrics(probabilities_dict: dict):
    """returns a dict containing class-wise evaluation metrics for each label
    """
    results = {}
    for label in probabilities_dict.keys():
        y_true = probabilities_dict[label]['true_label_two_class']
        probas: list = probabilities_dict[label]['probability_target_class'].values.tolist()
        labels = [0, 1]
        results[label] = {}
        results[label]["AUC"] = roc_auc_score(y_true, probas, labels=labels)
       
        classifications = [1 if prob > 0.5 else 0 for prob in probas]

        prfs = precision_recall_fscore_support(y_true, classifications, average=None, labels=labels)
        results[label]["Precision"] = prfs[0].tolist()[1]
        results[label]["Specificity"] = prfs[1].tolist()[1]
        results[label]["fbeta_scores"] = prfs[2].tolist()[1]
        results[label]["Supports"] = prfs[3].tolist()[1]

        logger.info(f"Class-wise evaluations for class {label}: {results[label]}")

    return results


def evaluate_classwise(top_level_classifier_folder: str, lifting_classifier_folder: str, walking_classifier_folder: str, store_local: bool, saving_folder: str) -> dict:
    """Wraps all utilities for a class-wise evaluation of a (potentially hierarchical) classifier, i.e. when looked as a collection of 2-class classifiers
    """
    probabilities_dict = aggregate_probabilities(top_level_classifier_folder, lifting_classifier_folder, walking_classifier_folder)
    if not probabilities_dict:
        return {}

    plot_all_ROC_curves(probabilities_dict, store_local, saving_folder)

    results = classwise_evaluation_metrics(probabilities_dict)
    return results


def get_UUID_from_filename(file: str):
    """returns the windowUUID from the filename
    """
    return file.split("/")[-1].split("_prepared_")[1].split(".csv")[0]


def valid_filename(file:str) -> bool:
    return '_prepared_' in file.split("/")[-1] and file.endswith(".csv")


def extend_probabilities_dict_with_class(probabilities_dict: dict, classifier_df: pd.DataFrame, relevant_class: int) -> dict:
    """appends the classwise probabilities of classifying a given sample in a df as correct or incorrect
    """
    if str(relevant_class) in probabilities_dict.keys():
        logger.error(f"Creating classwise probabilities for correctly classifying sample as two-class poblem failed, as class is already present in given dict")
        return {}
    if 'true_label' not in classifier_df.columns:
        logger.error(f"The given classifier df does not contain any information about the true_label of samples, creating classwise priobabilities failed")
        return {}
    
    raw_prediction_classes = [f"Raw_prediction_label_{label}" for label in classifier_df['true_label'].unique()]
    if not set(raw_prediction_classes).issubset(classifier_df.columns):
        logger.error(f"Not all needed columns for the creation of classwise-2-class probabilities given. Expected {raw_prediction_classes} Got {classifier_df.columns}")
        return {}
    relevant_class_col = f"Raw_prediction_label_{relevant_class}"

    if relevant_class_col not in raw_prediction_classes:
        logger.error(f"No Raw predictions for the relevant class {relevant_class} found in the columns of the classifier df: {relevant_class_col} missing")
        return {}

    raw_prediction_classes.remove(relevant_class_col)
    non_relevant_raw_prediction_classes = raw_prediction_classes

    # Create the two class labels: class 1 if sample belongs to the relevant class, class 0 if sample belongs to the rest class 
    two_class_labels = [1 if label == relevant_class else 0  for label in classifier_df['true_label']]
    # This calculation uses a modified version of the softmax, as the standard softmax cannot be used, as it would lead to changed classification behavior:
    # Example: 3 class problems with class probs 0.4(relevant class), 0.3, 0.3 -> relevant class correct predicted
    # But simply aggregating the rest into a common class would lead to a different classification behavior: 0.4, 0.6 -> rest class predicted, even though this is not the case
    # Therefore the probs get calculated only based on the raw prediction of the relevant class and the highest restclass -> 0.4, 0.3 -> 0.57, 0.42 -> class 0 predicted
    probability_wanted_class = [math.exp(classifier_df[relevant_class_col].iloc[i]) / (math.exp(classifier_df[relevant_class_col].iloc[i]) + math.exp(max(classifier_df[non_relevant_raw_prediction_classes].iloc[i]))) for i in range(len(classifier_df))]
    probability_rest_class = [1-predicted_probability for predicted_probability in probability_wanted_class]

    result_df = pd.DataFrame({"file": classifier_df['file'], 'true_label_two_class': two_class_labels, 'probability_rest_class': probability_rest_class,'probability_target_class': probability_wanted_class, '1_class': relevant_class})
    probabilities_dict[str(relevant_class)] = result_df

    return probabilities_dict


def append_subclassifier_raw_predictions_to_top_level_df(top_level_df:pd.DataFrame, lifting_df:pd.DataFrame, walking_df: pd.DataFrame): 
    """ appends the raw predictions made by the correct subclassifier to the top-level df
    """
    all_lifting_uuids = lifting_df['uuid']
    all_walking_uuids = walking_df['uuid']
    lifting_lowering_aggregated = lifting_df['true_label'].nunique() == 2
    if lifting_lowering_aggregated:
        lifting_mapping = {1: 1, 2: 2}
        walking_mapping = {1:3, 2:4, 3:5}
        resting_mapping = {3: 6}
    else:
        lifting_mapping = {1: 1, 2: 2, 3:3}
        walking_mapping = {1:4, 2:5, 3:6}
        resting_mapping = {3: 7}

    aggregated_labels = []
    sub_classifier_label = []
    subclassifier_raw_prediction_class_1 = []
    subclassifier_raw_prediction_class_2 = []
    subclassifier_raw_prediction_class_3 = []

    for index, file in enumerate(top_level_df['file']):
        top_level_uuid = get_UUID_from_filename(file)
        sub_classifier_index = None
        # Append for a file which was also handled by the lifting subclassifier
        try:
            sub_classifier_index = all_lifting_uuids[all_lifting_uuids == top_level_uuid].index[0]
        except:
            pass

        if sub_classifier_index is not None:
            aggregated_labels.append(lifting_mapping[lifting_df['true_label'].iloc[sub_classifier_index]])
            sub_classifier_label.append(lifting_df['true_label'].iloc[sub_classifier_index])

            subclassifier_raw_prediction_class_1.append(lifting_df['Raw_prediction_label_1'].iloc[sub_classifier_index])
            subclassifier_raw_prediction_class_2.append(lifting_df['Raw_prediction_label_2'].iloc[sub_classifier_index])
            subclassifier_raw_prediction_class_3.append(lifting_df['Raw_prediction_label_3'].iloc[sub_classifier_index] if not lifting_lowering_aggregated else -np.inf)
            continue
        
        # Append for a file which was also handled by the walking subclassifier
        try:
            sub_classifier_index = all_walking_uuids[all_walking_uuids == top_level_uuid].index[0]
        except:
            pass
        if sub_classifier_index is not None:
            aggregated_labels.append(walking_mapping[walking_df['true_label'].iloc[sub_classifier_index]])
            sub_classifier_label.append(walking_df['true_label'].iloc[sub_classifier_index])

            subclassifier_raw_prediction_class_1.append(walking_df['Raw_prediction_label_1'].iloc[sub_classifier_index])
            subclassifier_raw_prediction_class_2.append(walking_df['Raw_prediction_label_2'].iloc[sub_classifier_index])
            subclassifier_raw_prediction_class_3.append(walking_df['Raw_prediction_label_3'].iloc[sub_classifier_index])
            continue
        
        # Append for a file which was only handled by the top-level classifier (resting data)
        aggregated_labels.append(resting_mapping[3])
        sub_classifier_label.append(0)
        subclassifier_raw_prediction_class_1.append(-np.inf)
        subclassifier_raw_prediction_class_2.append(-np.inf)
        subclassifier_raw_prediction_class_3.append(-np.inf)
    
    top_level_df['all_class_label'] = aggregated_labels
    top_level_df['sub_classifier_label'] = sub_classifier_label
    top_level_df['Subclass_Raw_prediction_label_1'] = subclassifier_raw_prediction_class_1
    top_level_df['Subclass_Raw_prediction_label_2'] = subclassifier_raw_prediction_class_2
    top_level_df['Subclass_Raw_prediction_label_3'] = subclassifier_raw_prediction_class_3
    return top_level_df


def same_subclass(top_level_label: int, label:int, aggregated_lifting_lowering: bool):
    """Checks if the given label does belong to the same sub-analyzer as the given top-level label
    """
    # The resting data does not have subclasses
    if label == 7:
        return False
    if aggregated_lifting_lowering and label == 6:
        return False
    nr_lifting_activities = 2 if aggregated_lifting_lowering else 3
    # When data comes from the top-level lifting class
    if top_level_label == 1:
        return label <= nr_lifting_activities
    # When data comes from the top-level walking class
    return label <= (nr_lifting_activities +3)
    

def create_two_class_classifier_df(top_level_classifier_df: pd.DataFrame, label: int):
    """Generates the df of the raw predictions when evaluating of a label is done as two-class problem
    """
    top_level_raw_colnames = [f"Raw_prediction_label_{label}" for label in top_level_classifier_df['true_label'].unique()]
    subclassifier_raw_colnames = [f"Subclass_Raw_prediction_label_{label}" for label in top_level_classifier_df['sub_classifier_label'].unique() if label != 0]
    raw_predictions_class_0 = []
    raw_predictions_class_1 = []

    aggregated_lifting_lowering = top_level_classifier_df['all_class_label'].nunique() == 6
    # Maps all_class_label to respective true_label
    if aggregated_lifting_lowering:
        top_level_mapping = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3}
    else:
        top_level_mapping = {1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3}
    nr_classes = top_level_classifier_df['all_class_label'].nunique()

    for index, all_class_label in enumerate(top_level_classifier_df['all_class_label']):
        # 1. Calculate the top-level prob
        top_level_label = top_level_mapping[label]
        target_class_colname = f"Raw_prediction_label_{top_level_label}"
        rest_columns = [col for col in top_level_raw_colnames if col != target_class_colname]
        top_level_raw = top_level_classifier_df[target_class_colname].iloc[index]
        top_level_rest_raw = max(top_level_classifier_df[rest_columns].iloc[index])
     
       # When resting data is seen, no calculations need to be made for subclass handling
        if all_class_label == nr_classes:
            raw_predictions_class_1.append(top_level_raw)
            raw_predictions_class_0.append(top_level_rest_raw)
            continue
        
        # When a misclassification happened at the top-level
        if not same_subclass(top_level_classifier_df['true_label'].iloc[index], label, aggregated_lifting_lowering):
            raw_predictions_class_1.append(top_level_raw)
            raw_predictions_class_0.append(top_level_rest_raw)
            continue


        # 2. Calculate the sub classifier prob
        target_class_colname = f"Subclass_Raw_prediction_label_{top_level_classifier_df['sub_classifier_label'].iloc[index]}"

        rest_columns = [col for col in subclassifier_raw_colnames if col != target_class_colname]
        subclass_raw = top_level_classifier_df[target_class_colname].iloc[index]
        subclass_rest_raw = max(top_level_classifier_df[rest_columns].iloc[index])
        top_level_proba = math.exp(top_level_raw) / (math.exp(top_level_raw) + math.exp(top_level_rest_raw))
        sub_classif_proba = math.exp(subclass_raw) / (math.exp(subclass_raw) + math.exp(subclass_rest_raw))

        # Here the lower of the two is appended, as missing the border for the classifier closer to misclassification is enough for misclassification
        if top_level_proba > sub_classif_proba:
            raw_predictions_class_1.append(subclass_raw)
            raw_predictions_class_0.append(subclass_rest_raw)
        else:
            raw_predictions_class_1.append(top_level_raw)
            raw_predictions_class_0.append(top_level_rest_raw)
    
    relevant_labels = [label if seen_label == label else 0 for seen_label in top_level_classifier_df['all_class_label']]
    df = pd.DataFrame({"true_label": relevant_labels, f"Raw_prediction_label_{label}": raw_predictions_class_1, "Raw_prediction_label_0": raw_predictions_class_0, "file": top_level_classifier_df['file']})
    return df


def aggregate_probabilities(top_level_classifier_folder: str, lifting_classifier_folder: str, walking_classifier_folder: str) -> dict:
    """Aggregates the probabilities of a hierarchically classifying classifier. As a misclassification would occur if even one of the involved classifiers would misclassify, the worst classification is used  
    """
    probabilities_dict = {}
    top_level_classifier_df = pd.read_csv(os.path.join(top_level_classifier_folder, "predictions.csv"))
    # When the model directly classifies all possible categories, a quick exit can be used
    if not lifting_classifier_folder or not walking_classifier_folder:
        for label in top_level_classifier_df['true_label'].unique():
            probabilities_dict = extend_probabilities_dict_with_class(probabilities_dict, top_level_classifier_df, label)
            if not probabilities_dict:
                return {}
        return probabilities_dict
    
    walking_classifier_df = pd.read_csv(os.path.join(walking_classifier_folder, "predictions.csv"))
    lifting_classifier_df = pd.read_csv(os.path.join(lifting_classifier_folder, "predictions.csv"))
    

    if not sum([valid_filename(file) for file in top_level_classifier_df['file']]) == len(top_level_classifier_df):
        logger.error(f"Received invalid filename in the top-level classifier. Aggregating probabilities failed")
        return pd.DataFrame()
    if not sum([valid_filename(file) for file in walking_classifier_df['file']]) == len(walking_classifier_df):
        logger.error(f"Received invalid filename in the walking-level classifier. Aggregating probabilities failed")
        return pd.DataFrame()
    if not sum([valid_filename(file) for file in lifting_classifier_df['file']]) == len(lifting_classifier_df):
        logger.error(f"Received invalid filename in the lifting-level classifier. Aggregating probabilities failed")
        return pd.DataFrame()

    top_level_classifier_df['uuid'] = [get_UUID_from_filename(file) for file in top_level_classifier_df['file']]
    walking_classifier_df['uuid'] = [get_UUID_from_filename(file) for file in walking_classifier_df['file']]
    lifting_classifier_df['uuid'] = [get_UUID_from_filename(file) for file in lifting_classifier_df['file']]

    top_df = append_subclassifier_raw_predictions_to_top_level_df(top_level_classifier_df, lifting_classifier_df, walking_classifier_df)

    for label in top_df['all_class_label'].unique():
        two_class_df = create_two_class_classifier_df(top_df, label)
        probabilities_dict = extend_probabilities_dict_with_class(probabilities_dict, two_class_df, label)
        if not probabilities_dict:
            return {}

    return probabilities_dict

