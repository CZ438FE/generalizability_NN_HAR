import argparse
import sys
import os
from config import log
from create_windows import create_windows
from generate_features import generate_features
from preprocess import preprocess
from train_model import train_model
from evaluate_model import evaluate_model
from evaluate_hierarchical import evaluate_hierarchical
from evaluate_classwise import evaluate_classwise
from compare_models import compare_models

def main(**table):
    """Reads the arguments from the command line    
    """
    parser = argparse.ArgumentParser(description='Perform steps in the pipeline for training models')
    subparser = parser.add_subparsers(help='Functionalities that can be executed.', dest='service')

    # global variables
    parser.add_argument(
        '-l', '--log_level',
        metavar='',
        dest='log_level',
        type=str,
        default='info',
        help='set logging level, default: %(default)s'
    )
    parser.add_argument(
        '-dp', '--data_path',
        metavar='',
        dest='data_path',
        required=True,
        type=str,
        help='pass local data path to folder in which all the results are stored'
    )
    parser.add_argument(
        '-d', '--dryrun',
        action='store_const',
        const=True,
        default=False,
        help='run the program without saving results.'
    )

    # Arguments for creating windows
    subparser_create_windows = subparser.add_parser('create_windows', help='Cut the labeled data into windows')
    create_windows_group = subparser_create_windows.add_argument_group(title='Arguments for creation of windows from labeled data')
    create_windows.add_args(create_windows_group)

    # Arguments for generating_features
    subparser_generate_features = subparser.add_parser('generate_features', help='Generate features for windows')
    generate_features_group = subparser_generate_features.add_argument_group(title='Arguments for the generation of features from the windows')
    generate_features.add_args(generate_features_group)

    # Arguments for preprocess
    subparser_preprocess = subparser.add_parser('preprocess', help='Preprocess data for training of models')
    preprocess_group = subparser_preprocess.add_argument_group(title='Arguments for the preprocessing before training')
    preprocess.add_args(preprocess_group)

    # Arguments for train_model
    subparser_train_models = subparser.add_parser('train_model', help='Train Neural Networks')
    train_models_group = subparser_train_models.add_argument_group(title='Arguments for the training of the neural network')
    train_model.add_args(train_models_group)

    # Arguments for evaluate_model
    subparser_evaluate_model = subparser.add_parser('evaluate_model', help='Evaluate Model on test data')
    evaluate_models_group = subparser_evaluate_model.add_argument_group(title='Arguments for the evaluation of the model')
    evaluate_model.add_args(evaluate_models_group)

    # Arguments for evaluate_hierarchical
    subparser_evaluate_hierarchical = subparser.add_parser('evaluate_hierarchical', help='Evaluate hierarchical models')
    evaluate_hierarchical_group = subparser_evaluate_hierarchical.add_argument_group(title='Arguments for the evaluation of hierarchical models')
    evaluate_hierarchical.add_args(evaluate_hierarchical_group)

    # Arguments for plotting of classwise ROC curves
    subparser_evaluate_classwise = subparser.add_parser('evaluate_classwise', help='Classwise evaluations')
    evaluate_classwise_group = subparser_evaluate_classwise.add_argument_group(title='Arguments for the classwise evaluations of models')
    evaluate_classwise.add_args(evaluate_classwise_group)

    # Arguments for statistical comparison of models
    subparser_compare_model = subparser.add_parser('compare_models', help='Statistical comparison of models')
    compare_models_group = subparser_compare_model.add_argument_group(title='Arguments for the statistical comparison of models')
    compare_models.add_args(compare_models_group)


    # set arguments into the table
    if not table:
        table = vars(parser.parse_args())

    # set logging level
    logger = log.setup_logger('main', table.get('log_level'))
    current_service = table.get('service')

    # service dependent functions
    if current_service == 'create_windows':
        logger.info(f"Service \x1b[33;1m{current_service}\x1b[0m triggered.")
        table = create_windows.check_table_args(table)
        create_windows.create_windows(table)
    elif current_service == 'generate_features':
        logger.info(f"Service \x1b[33;1m{current_service}\x1b[0m triggered.")
        table = generate_features.check_table_args(table)
        generate_features.generate_features(table)
    elif current_service == 'preprocess':
        logger.info(f"Service \x1b[33;1m{current_service}\x1b[0m triggered.")
        table = preprocess.check_table_args(table)
        preprocess.preprocess(table)
    elif current_service == 'train_model':
        logger.info(f"Service \x1b[33;1m{current_service}\x1b[0m triggered.")
        table = train_model.check_table_args(table)
        train_model.train_model(table)
    elif current_service == 'evaluate_model':
        logger.info(f"Service \x1b[33;1m{current_service}\x1b[0m triggered.")
        table = evaluate_model.check_table_args(table)
        evaluate_model.evaluate_model(table)
    elif current_service == 'evaluate_hierarchical':
        logger.info(f"Service \x1b[33;1m{current_service}\x1b[0m triggered.")
        table = evaluate_hierarchical.check_table_args(table)
        evaluate_hierarchical.evaluate_hierarchical(table)
    elif current_service == 'evaluate_classwise':
        logger.info(f"Service \x1b[33;1m{current_service}\x1b[0m triggered.")
        table = evaluate_classwise.check_table_args(table)
        evaluate_classwise.evaluate_classwise(table)
    elif current_service == 'compare_models':
        logger.info(f"Service \x1b[33;1m{current_service}\x1b[0m triggered.")
        table = compare_models.check_table_args(table)
        compare_models.compare_models(table)
    elif current_service is None:
        logger.error("No valid service given. Consult README for choices")
    else:
        logger.error(f"Received unknown chosen service {current_service}")
        return

    logger.info(f"Service \x1b[33;1m {current_service} \x1b[0m finished.")


if __name__ == '__main__':
    main()
