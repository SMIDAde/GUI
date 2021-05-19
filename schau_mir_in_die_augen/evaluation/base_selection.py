import os
import logging
import datetime

import argparse

from schau_mir_in_die_augen.datasets.DatasetBase import DatasetBase
from schau_mir_in_die_augen.datasets.Bioeye import BioEye
from schau_mir_in_die_augen.datasets.DemoDataset import \
    DemoDataset, DemoDatasetUser, DemoDatasetGender, DemoDatasetUserGender

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from schau_mir_in_die_augen.rbfn.Rbfn import Rbfn

from schau_mir_in_die_augen.evaluation.base_evaluation import BaseEvaluation
from schau_mir_in_die_augen.evaluation.evaluation_our_appended import OurEvaluationAppended


def start_logging(general_name: str = None, folder_attachment: str = None):
    """ create folder for log files and set basic log configuration

    :param general_name: will be in the title of folder and the name of the log file
    :param folder_attachment: will only be in the title of the folder
    """

    if not os.path.exists('logs/SMIDA_{name}{attachment}'.format(name=general_name, attachment=folder_attachment)):
        os.makedirs('logs/SMIDA_{name}{attachment}'.format(name=general_name, attachment=folder_attachment))

    logging.basicConfig(filename='logs/SMIDA_{name}{attachment}/{name}_{time}.log'.format(
        name=general_name, attachment=folder_attachment, time=datetime.datetime.now().strftime('%y%m%d_%H%M')),
        filemode='a',
        format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG)


dataset_list = ['bio-tex', 'bio-tex1y', 'bio-ran', 'bio-ran1y',
                'whl', 'gaze_onfaces_user_prediction',
                'rigas-tex', 'rigas-ran',
                'dyslexia', 'BiometricsDS',
                'demo-data', 'demo-user', 'demo-gender', 'demo-user-gender', 'biometricDS_fo_user_ID', 'Gaze_on_faces']


def get_dataset(dataset_name: str, arguments: argparse.Namespace) -> DatasetBase:
    """ Return a Child of DatasetBase by given name.

    See /documentation/DATASETS.md

    :param dataset_name: Dataset to choose
    :param arguments: arguments from parser
    :return: Child of DatasetBase
    """
    logger = logging.getLogger(__name__)

    if dataset_name == 'bio-tex':
        ds = BioEye(BioEye.Subsets.TEX_30min_dv,
                    user_limit=arguments.user_limit,
                    use_valid_data=arguments.use_valid_data)
    elif dataset_name == 'bio-ran':
        ds = BioEye(BioEye.Subsets.RAN_30min_dv,
                    user_limit=arguments.user_limit,
                    use_valid_data=arguments.use_valid_data)
    elif dataset_name == 'bio-tex1y':
        ds = BioEye(BioEye.Subsets.TEX_1year_dv,
                    score_level_eval=arguments.score_level_1y,
                    one_year_train=arguments.score_level_1y_train,
                    user_limit=arguments.user_limit,
                    use_valid_data=arguments.use_valid_data)
    elif dataset_name == 'bio-ran1y':
        ds = BioEye(BioEye.Subsets.RAN_1year_dv,
                    score_level_eval=arguments.score_level_1y,
                    one_year_train=arguments.score_level_1y_train,
                    user_limit=arguments.user_limit,
                    use_valid_data=arguments.use_valid_data)
    elif dataset_name[:4] == 'demo':
        if dataset_name == 'demo-data':
            dsh = DemoDataset
        elif dataset_name == 'demo-user':
            dsh = DemoDatasetUser
        elif dataset_name == 'demo-gender':
            dsh = DemoDatasetGender
        elif dataset_name == 'demo-user-gender':
            dsh = DemoDatasetUserGender
        else:
            raise Exception('No Demo Dataset known with name "{}"'.format(dataset_name))
        ds = dsh(user_limit=arguments.user_limit,
                 seed=arguments.seed)
    else:
        logger.error('Unknown dataset: "{}"'.format(dataset_name))
        raise Exception('Unknown dataset: "{}"'.format(dataset_name))

    logger.debug('Selected Dataset: {ds} with parameters:\n{para}'.format(
        ds=type(ds), para=ds.__dict__))

    return ds


classifier_list = ['rf', 'rbfn', 'svm', 'svm_linear', 'naive_bayes', 'logReg']


def get_classifier(classifier_name: str, arguments: argparse.Namespace):
    """ Return a Classifier by given name.

    See /documentation/CLASSIFIER.md

    :param classifier_name: classifier to choose
    :param arguments: arguments from parser
    :return: classifier
    """

    logger = logging.getLogger(__name__)

    if classifier_name == 'rf':
        if (arguments.max_depth is not None and arguments.max_depth > 0) \
                and (arguments.min_samples_leaf > 0) and (arguments.min_samples_split > 0):
            # todo: Why is it, that rf parameters are only working if there are min_samples_leaf and min_samples_split?
            clf = RandomForestClassifier(max_features=arguments.max_features,
                                         n_estimators=arguments.rf_n_estimators,
                                         n_jobs=-1, max_depth=arguments.max_depth,
                                         bootstrap=arguments.bootstrap,
                                         min_samples_leaf=arguments.min_samples_leaf,
                                         min_samples_split=arguments.min_samples_split)
        else:
            clf = RandomForestClassifier(n_estimators=arguments.rf_n_estimators, n_jobs=-1)

    elif classifier_name == 'rbfn':
        clf = Rbfn(arguments.rbfn_k)

    elif classifier_name == 'naive_bayes':
        clf = GaussianNB()

    elif classifier_name == 'logReg':
        clf = LogisticRegression(solver='lbfgs', multi_class='auto')

    elif classifier_name == 'svm_linear':
        clf = SVC(gamma='scale', probability=True, kernel='linear')

    elif classifier_name == 'svm':
        if arguments.svm_c and arguments.svm_gamma:
            clf = SVC(gamma=arguments.svm_gamma, probability=True, C=arguments.svm_c)
        else:
            clf = SVC(gamma='scale', probability=True)
            
    else:
        logger.error('Unknown classifier: "{}"'.format(classifier_name))
        raise Exception('Unknown classifier: "{}"'.format(classifier_name))

    clf.random_state = arguments.seed

    if arguments.class_weight:
        # works at least for SVC and RF
        clf.class_weight = arguments.class_weight

    logger.debug('Selected Classifier: {clf} with parameters:\n{para}'.format(
        clf=type(clf), para=clf.__dict__))

    return clf


method_list = ['paper-append']


def get_method(method_name: str, classifier, arguments: argparse.Namespace, dataset_name=None) -> BaseEvaluation:
    """ Return a evaluation method by given name.

    See /documentation/METHODS.md

    :param dataset_name:
    :param method_name: method to choose
    :param classifier: classifier
    :param arguments: arguments from parser
    :return: evaluation method (child of BaseEvaluation)
    """

    logger = logging.getLogger(__name__)

    # init the right method
    if method_name == 'paper-append':
        eva = OurEvaluationAppended(classifier,
                                    vel_threshold=arguments.ivt_threshold,
                                    min_fix_duration=arguments.ivt_min_fix_time,
                                    paper_only=True)
    else:
        logger.error('Unknown method: "{}"'.format(method_name))
        raise Exception('Unknown method: "{}"'.format(method_name))

    # general arguments
    eva.plot_confusion_matrix = arguments.plot_confusion_matrix
    eva.dataset_name = dataset_name

    # filter arguments
    eva.preparation_steps['conversion'] = arguments.filter_data
    eva.preparation_steps['filter_type'] = arguments.filter_type
    filter_parameters = dict()
    for idd in range(0, len(arguments.filter_parameter), 2):
        filter_parameters.update({arguments.filter_parameter[idd]: int(arguments.filter_parameter[idd+1])})
        # todo: not wise to convert everything to int, but i have no better idea right now.
    eva.preparation_steps['filter_parameter'] = filter_parameters

    logger.debug('Selected Evaluation: {eva} with parameters:\n{para}'.format(
        eva=type(eva), para=eva.__dict__))

    return eva


def get_default_parser() -> argparse.ArgumentParser:
    """ Returns a Parser for the general features. It will be completed in 'get_conditional_parser'

    :return: parser
    """

    parser = get_parser()

    parser.add_argument('-dh', '--detailed_help', action='store_true',
                        help='will show more detailed help after parsing basic commands.')

    parser = add_dataset_parser(parser)
    parser.add_argument('--test_dataset', choices=dataset_list, required=False)

    parser.add_argument('-clf', '--classifier', choices=classifier_list, required=True)

    parser.add_argument('--method', choices=method_list, required=True)

    parser.add_argument('--svm_gamma', type=float, required=False)
    parser.add_argument('--svm_c', type=float, required=False)
    parser.add_argument('--label', type=str, choices=['user', 'gender', 'label_by_dataset', 'age'], default='user',
                        required=False,
                        help='The label for the classifier to predict')
    # these are Random_forrest only parameters!?
    parser.add_argument('--bootstrap', action='store_true',  default=True, help='Can be used with RF')
    parser.add_argument('--max_depth', type=int, required=False)
    parser.add_argument('--min_samples_leaf', type=int, required=False)
    parser.add_argument('--max_features', default='sqrt', required=False)
    parser.add_argument('--min_samples_split', type=int, required=False)

    parser.add_argument('-mf', '--modelfile', type=str, required=True,
                        help='File path of the trained evaluation.'
                             'Use "[model]and_your_desired_name_here" to use model folder.')
    # todo: [] not possible from command line

    parser.add_argument('-mp', '--more_parameters', type=str, default=None,
                        help='Getting more Parameters for different cases (e.g. training)')

    # eye movement classifier
    # todo: this should be later conditional
    parser = get_parser_eye_movement_classifier('IVT', parser)

    parser.add_argument('--train_top_features', type=int, nargs='*',
                        help='Set flag', default=0)

    parser.add_argument('--use_normalization', action='store_true',
                        help='Can be used to normalize features with mean centralization method')

    return parser


def add_dataset_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ add argument for dataset to parser

    :param parser: parser
    :return: parser
    """
    parser.add_argument('-ds', '--dataset', choices=dataset_list, required=True)
    return parser


def add_seed_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ add argument for dataset to parser

    :param parser: parser
    :return: parser
    """
    parser.add_argument('--seed', type=int, default=42)
    return parser


def get_conditional_parser(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> argparse.ArgumentParser:
    """ completes the default Parser based on already parsed arguments

    :param parser: parser
    :param arguments: already parsed arguments
    :return: parser
    """

    parser = get_parser_dataset(arguments.dataset, parser)
    parser = get_parser_classifier(arguments.classifier, arguments.method, parser)
    parser = get_parser_method(arguments.method, parser)

    if arguments.more_parameters == 'train':
        parser.add_argument('--test_train', action='store_true',
                            help='At the End of Training, run one Evaluation with train-data.')

    return parser


def get_parser(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """ will create and return a parser or pass given parser

    :param parser: parser
    :return: parser
    """

    if parser is None:
        return argparse.ArgumentParser(description='Entry point for evaluations', conflict_handler="resolve")
    else:
        assert isinstance(parser, argparse.ArgumentParser)
        return parser


def get_parser_dataset(dataset_name: str, parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """ return parser for diven dataset.
        Will either create one, pass given parser with additional arguments

    :param dataset_name: name of dataset
    :param parser: parser
    :return: parser
    """
    parser = get_parser(parser)

    parser = add_seed_parser(parser)
    parser.add_argument('-ul', '--user_limit', type=int, default=None)

    if len(dataset_name) >= 3 and dataset_name[:3] == 'bio':
        parser.add_argument('--use_valid_data', action='store_true',
                            help='Can be used with only BioEye dataset for only taking valid data')
    parser.add_argument('--save_eval_res_path', type=str, default=None,
                            help='choose path to save the evaluation results')

    return parser


def get_parser_eye_movement_classifier(emc_name, parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    parser = get_parser(parser)

    if emc_name == 'IVT':
        # Warning: default is used in gui_bokeh!
        #   If you change it here, change it there too!
        parser.add_argument('--ivt_threshold', type=int, default=50,
                            help='Velocity threshold for ivt.')
        parser.add_argument('--ivt_min_fix_time', type=float, default=0.1,
                            help='Minimum duration to count as fixation for ivt.')
    return parser


def get_parser_classifier(classifier_name: str, method_name: str, parser: argparse.ArgumentParser = None) \
        -> argparse.ArgumentParser:
    """ Return parser with arguments for given classifier and method.
        If a parser is given the arguments will be added.

    :param classifier_name:
    :param method_name:
    :param parser:
    :return: parser
    """
    parser = get_parser(parser)

    parser = add_seed_parser(parser)
    parser.add_argument('--class_weight', type=str, default=None,
                        help='Use "balanced" to deal with unbalanced classes.')

    if classifier_name == 'rf':
        parser.add_argument('--rf_n_estimators', type=int, default=200)
    elif classifier_name == 'rbfn':
        parser.add_argument('--rbfn_k', type=int, default=32)
    return parser


def get_parser_method(method_name: str, parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """ Return parser with arguments for given classifier and method.
        If a parser is given the arguments will be added.

    :param method_name:
    :param parser:
    :return: parser
    """
    parser = get_parser(parser)

    parser.add_argument('-pcm', '--plot_confusion_matrix', action='store_true',
                        help='Plot confusion matrix with all labels.')

    parser.add_argument('--filter_data', type=str, default='angle_deg',
                        help='Select the Dataformat to apply the filter. E.g.: "pixel"')
    parser.add_argument('--filter_type', type=str, default='savgol',
                        help='Select Filter to append on data. (There is no other than savgol yet)')
    parser.add_argument('--filter_parameter', nargs='*', default=['frame_size', '15', 'pol_order', '6'],
                        help='Add Parameters for selected filters by "Keyword1 Value1 Keyword2 Value2 ..."')

    return parser


def parse_arguments(arguments) -> argparse.Namespace:
    """ default argument parsing for train and evaluation method

    :param arguments: arguments to parse
    :return: parsed arguments
    """

    # get parser with default parameter
    parser = get_default_parser()

    # check them and get new parser
    preparsed_arguments, _ = parser.parse_known_args(arguments)
    parser = get_conditional_parser(parser, preparsed_arguments)

    if preparsed_arguments.detailed_help:
        arguments += ['--help']

    # pars arguments and return
    return parser.parse_args(arguments)
