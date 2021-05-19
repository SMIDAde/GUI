import sys
import logging
import datetime
import time

sys.path.append('../')

from schau_mir_in_die_augen.evaluation.base_selection import \
    get_method, get_classifier, get_dataset, parse_arguments, start_logging

from schau_mir_in_die_augen.general.io_functions import save_pickle_file

from scripts.inspection import check_python_version

# This is maybe not useful to give direct hints (it will fail before here with at least some python versions),
# but, with the output we maybe collect more running versions
check_python_version()
def main(args):

    ###############
    # PREPARATION #
    ###############
    folder_attachment = '_{data}_{clf}_{method}'.format(data=args.dataset, clf=args.classifier, method=args.method)
    start_logging('Evaluation', folder_attachment=folder_attachment)
    logger = logging.getLogger(__name__)
    logger.info('Start Evaluation with Parameters:\n{}'.format(args))
    time_start = time.time()
    ###############

    # dataset selection
    ds = get_dataset(args.dataset, args)

    # initialize the right classifier
    clf = get_classifier(args.classifier, args)

    # init the right method
    eva = get_method(args.method, clf, args, dataset_name=ds.dataset_name)

    ############
    # LOADING #
    ############
    logger.info('Start Loading (Preparing took {:.3f} seconds).'.format(time.time()-time_start))
    time_loading = time.time()
    start_time = datetime.datetime.now()
    ############

    # data for training
    trajectories_train = ds.load_training_trajectories()
    logger.debug('{num} Cases for Training: {cas}'.format(num=len(trajectories_train),
                                                          cas=trajectories_train.cases))

    trajectories_train = eva.do_data_preparation(trajectories_train)
    logger.debug('Processing of Training: {}'.format(trajectories_train.processing))

    print("data loading time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

    ############
    # ANALYSIS #
    ############
    extract_time = time.time()
    extract_time2 = datetime.datetime.now()  # todo: remove this

    logger.info('Start Extraction (Loading test and train data took {:.3f} seconds).'.format(
        time.time() - time_loading))
    print("Feature extraction for {} training cases".format(len(trajectories_train)))
    labeled_feature_values_train = eva.provide_feature(trajectories=trajectories_train,
                                                       normalize=args.use_normalization,
                                                       label=args.label)
    print("Feature extraction time: ", str(datetime.timedelta(seconds=
                                                              (datetime.datetime.now() - extract_time2).seconds)))

    ############
    # TRAINING #
    ############
    time_training = time.time()
    logger.info('Start Training (Feature Extraction took {:.3f} seconds).'.format(time.time() - extract_time))
    ############

    eva.train(labeled_feature_values=labeled_feature_values_train)

    if args.test_train:
        train_results = eva.evaluation(feature_values=labeled_feature_values_train)
        eva.train_accuracy = train_results
        print('Results with Train data:', train_results)
        logger.debug('Results with Train data: {}'.format(train_results))
    else:
        train_results = None

    save_pickle_file(args.modelfile, eva)

    return train_results


if __name__ == '__main__':
    # make possible to look for specific parameters
    #   see get_conditional_parser
    args = parse_arguments(sys.argv[1:]+['--more_parameters', 'train'])
    print(args)

    main(args)

