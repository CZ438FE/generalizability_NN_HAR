import copy
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from config.log import logger


def return_dict_nr_to_label(level: str):
    """returns a dict which maps the activity numbers to the activity labels: 1 -> Lifting
    """
    valid_levels = ['top', 'mid', 'low']
    if level not in valid_levels:
        logger.error(f"Invalid granularity given. Choose one of {valid_levels} ")
        return {}, "invalid_granularity_given"
    if level == 'top':
        return {"1": "Lifting", "2": "Walking", "3": "Resting", "4": "Driving"}, None
    if level == 'mid':
        return {"11": "Lifting", "12": "lowering", "15": "Holding", "21": "Walking Straight", "22": "Walking Upstairs", "23": "Walking Downstairs", "24": "Walking Sideways",
                "31": "Resting", "32": "Sitting"}, None
    return {"11": "Lifting", "12": "Lowering", "15": "Holding", "211": "Walking Free", "212": "Carrying", "213": "Walking Pushing",
            "214": "Walking Pulling", "221": "Upstairs Free", "222": "Upstairs Carrying", "231": "Downstairs Free", "241": "Sidesteps Free", "242": "Sidesteps carrying",
            "31": "Standing", "32": "Sitting",}, None


def change_label_nr_to_human_readable(label_dict: dict, dict_nr_to_label: dict):
    """ Returns a version of the inserted label_dict, which has human-readable labels
    """
    if not label_dict:
        logger.error("Cannot change the labels of an empty dict")
        return {}
    if not dict_nr_to_label:
        logger.error("Cannot change the labels when the dict containing the mapping information is empty")
        return {}
    # The following sorts the numerical labels and replaces the numbers with the activity
    final_dict = {}
    for key in sorted(list(label_dict.keys())):
        final_dict[dict_nr_to_label[key].replace(" ", "\n")] = label_dict[key]
    return final_dict


def show_label_distribution(granularity: str, labels_distribution: dict, saving_folder: str, store_local=False):
    """Visualizes the distribution of the seen labels
    """
    # Gather a dict which maps activity labels as int to human-readable activities, e.g. "1":"Lifting"
    nr_to_label_dict, error = return_dict_nr_to_label(granularity)
    if error:
        return

    # Replace the activity numbers with human-readable names
    human_readable_dict = change_label_nr_to_human_readable(labels_distribution, nr_to_label_dict)
    if not human_readable_dict:
        logger.error(f"Cannot show label distribution with an empty dict")
        return

    # Create  a DataFrame from the dict containing the distribution of the labels in human-readable form
    df = pd.DataFrame(human_readable_dict, index=[0])

    # Show the distribution of the df
    plt.figure(figsize=(12, 12))

    ax = sns.barplot(x=df.columns, y=df.iloc[0, :])
    plt.title("Windows per label")
    for i in ax.containers:
        ax.bar_label(i,)
    if store_local:
        plt.savefig(os.path.join(saving_folder, "distribution_seen_labels.png"))
    plt.show()
    return True


def infer_level(label: list):
    """infers the level based on a label / list of labels
    """
    if isinstance(label, str):
        max_len = len(label)
    elif isinstance(label, list):
        max_len = max([len(str(x)) for x in label])
    else:
        logger.error(f"Got invalid datatype for inferring the label {type(label)}")
        return "", "invalid_dtype_given_for_inferring_level"

    if max_len == 3:
        return "low", None
    if max_len == 2:
        return "mid", None
    if max_len == 1:
        return "top", None

    logger.error(f"Could not generate a valid level fot given label {label}")
    return "", f"level_not_defined_for_maxlen_{max_len}"


def replace_list_labels_with_human_readable_form(list_of_labels: list):
    """Changes the label keys of a dict to human-readable form
    """
    # Inferring the seen level is necessary to get the correct dict which maps label_nr to human-readable names
    level, error = infer_level(list_of_labels)
    if error:
        return [], error

    # Get the dict containing the correct keys
    dict_nr_to_label, error = return_dict_nr_to_label(level)
    if error:
        return [], error

    # Iterate over all the levels and construct list with human-readable names
    human_readable_levels = [dict_nr_to_label.get(str(label)) for label in list_of_labels]

    return human_readable_levels, None


def plot_original_samples_on_data_after_resampling(prop_original_data_dict: dict, saving_folder: str, store_local: bool):
    """Plots the proportion of original samples on all samples of the class after resampling in a barchart
    """
    # test if class and prop_existing_after_balancing are the keys of given dict
    if not set(["class", "prop_existing_after_balancing"]).issubset(set(prop_original_data_dict.keys())):
        logger.error(f"Got malformed dict with keys {prop_original_data_dict.keys()}")
        return "malformed_prop_original_data_dict_given"

    internal_prop_original_data_dict = copy.deepcopy(prop_original_data_dict)

    # Change the labels of the Plot to human_readable format
    internal_prop_original_data_dict["class"], error = replace_list_labels_with_human_readable_form(internal_prop_original_data_dict.get("class"))
    if error:
        return error

    plt.figure(figsize=(12, 12))
    plt.title("Proportion of original data on all data after balancing each class")
    bar1 = sns.barplot(x="class", y="prop_existing_after_balancing",
                       data=internal_prop_original_data_dict)
    for i in bar1.containers:
        bar1.bar_label(i,)
    plt.ylabel('Proportion of original Data')
    plt.xlabel('Label')
    if store_local:
        plt.savefig(os.path.join(saving_folder, "proportion_of_original_data_on_data_after_resampling.png"))
    plt.show()
    return None


def plot_accuracies_and_loss_over_batches(train_accuracies: list, val_accuracies: list, trainings_losses: list, val_losses: list, store_local: bool, saving_folder: str, validation_data_step_length: int):
    """Plots the change of loss and accuracy over the training- (and validation-set, if given)
    """
    # Keep track if this function displayed at least one plot, if so, it was successful
    plotted_something = False
    if train_accuracies != []:
        plotted_something = plot_course(train_accuracies, "minibatch", "Accuracy", "Training", store_local, saving_folder)
    else:
        logger.warning(f"Could not plot the change of accuracies, as the list of train_accuracies was empty")
    if trainings_losses != []:
        plotted_something = plot_course(trainings_losses, "minibatch", "Loss", "Training", store_local, saving_folder)
    else:
        logger.warning(f"Could not plot the change of losses, as the list of train_losses was empty")

    if val_accuracies != []:
        plotted_something = plot_course(val_accuracies, f"minibatch (in {validation_data_step_length})", "Accuracy", "Validation", store_local, saving_folder)
    if val_losses != []:
        plotted_something = plot_course(val_losses, f"minibatch (in {validation_data_step_length})", "Loss", "Validation", store_local, saving_folder)

    # Inform the user that the plots were stored
    if store_local and plotted_something:
        logger.info(f"Saved the plot(s) in the specified saving folder {saving_folder}")

    return plotted_something


def get_class_names_from_len(len_df: int, hierarchical_model: str):
    """return a list of class names depending on the amount of unique label classes
    """
    if len_df not in [2, 3, 6, 7]:
        return [x for x in range(len_df)]
    if len_df == 3 and hierarchical_model == "top":
        return ["Lifting", "Walking", "Resting"]
    elif len_df == 3 and hierarchical_model == "lifting":
        return ["Lifting", "Lowering", "Holding"]
    elif len_df == 2 and hierarchical_model == "lifting":
        return ["Lifting &\nLowering", "Holding"]
    elif len_df == 3 and hierarchical_model == "walking":
        return ["Walking\nStraight", "Ascending\nstairs", "Descending\nstairs"]
    elif len_df == 7:
        return ["Lifting", "Lowering", "Holding", "Walking\nstraight", "Ascending\nstairs", "Descending\nstairs", "Resting"]
    elif len_df == 6:
        return ["Lifting &\nLowering", "Holding", "Walking\nstraight", "Ascending\nstairs", "Descending\nstairs", "Resting"] 
    return [x for x in range(len_df)]


def plot_confusion_matrix(confusion_matrix, store_local: bool, saving_folder: str, data_set: str, hierarchical_model: str):
    """Shows the confusion matrix in a better visible form
    """
    df_cm = pd.DataFrame(confusion_matrix).astype("int")
    class_names = get_class_names_from_len(len(df_cm), hierarchical_model)
    df_cm.index = class_names
    df_cm.columns = class_names

    plt.figure(figsize=(9, 6))
    ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn', annot_kws={"size": 11})
    ax.set_ylabel("True label", rotation=0, fontsize = 12, labelpad=16.0)
    plt.yticks(rotation=0) 
    plt.xlabel("Prediction", fontsize = 12)
    plt.title(f"Confusion Matrix of the {data_set.split('_')[0]} data")
    if store_local:
        plt.savefig(os.path.join(saving_folder, f"{data_set}_confusion_matrix.png"))
    plt.show()

    return df_cm


def plot_course(list_of_variable: list, x_unit: str, y_unit: str,  dataset: str, store_local: bool, saving_folder: str, figsize=(12, 12)):
    """plots the course of a given variable and saves the resulting plot in the saving_folder, if specified
    """
    plt.figure(figsize=figsize)
    plt.title(f"{y_unit} over {dataset}-data {x_unit}")
    plt.plot(range(len(list_of_variable)), list_of_variable, label=dataset)
    plt.ylabel(y_unit)
    plt.xlabel(x_unit)
    plt.legend(loc="best")
    if store_local:
        plt.savefig(os.path.join(saving_folder, f"{dataset}_{x_unit}.png"))
    plt.show()

    return True
