import datetime
import logging
import sys
import time
from abc import ABC, abstractmethod
from enum import IntEnum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.append('../')
from schau_mir_in_die_augen.feature_extraction import important_features, trajectory_split_and_feature_cached
from schau_mir_in_die_augen.process.trajectory import Trajectory, Trajectories

from schau_mir_in_die_augen.evaluation.eval_vis import plot_confusion_matrix


class FeatureLabels(IntEnum):
    """ These class provides extra columns to label the feature.

    All labels are deleted when training the data.
    """
    label = 0  # label for training

    user = 11  # name/id of user
    case = 12  # name/id of test case/setup/group

    classifier = 21  # name/id of used classifier


class BaseEvaluation(ABC):
    """ Base for all Evaluation methods.

    Some details are maybe also here:
    https://gitlab.informatik.uni-bremen.de/cgvr/smida2/schau_mir_in_die_augen/-/blob/master/documentation/METHODS.md
    """

    def __init__(self):

        self.clf = None
        self.graph_path = None
        self.feature = []
        self.dataset_name = []
        self.users = []
        self.n_top_features = []        # list of lists with names of features for every classifier
        self.n_ref_top_features = []    # list of lists with names of features for every classifier
        self.binary_classifier_threshold = 0.5
        self.use_optimal_threshold = False  # flag to set whether to use optimal threshold (true) or not (false)
        # number of folds to use k-fold method for cross validation
        #   (default=0 means the method wont be used)
        self.CV_k_fold = 0

        # i am not sure if this is a good solution. But better than in IVT
        #   has to be called manually by do_data_preparation !!!
        # Warning: i use the default values in bokeh! please keep it up to date
        self.preparation_steps = {'conversion': '',
                                  'filter_type': '',
                                  'filter_parameter': dict()}

        self.plot_confusion_matrix = False
        self.info_time = 10  # seconds (how often updates will be print with ETA)

    @abstractmethod
    def trajectory_split_and_feature(self, trajectory: Trajectory) -> list:
        """ Generate feature vectors

        for all saccades and fixations and our saccade, fixation, general features in a trajectory

        :param trajectory:
            2D array of gaze points (x,y) and more
        :return: list
            list of feature vectors for each classifier (each is a 1D ndarray)
        """

    @abstractmethod
    def get_split_parameter(self):
        """ Return specific Parameter for split

        e.g. IVT with velocity threshold 50 deg/s and minimal fixation duration of 0.1 seconds
        """
        split_method = 'IVT'
        split_parameter = {'vel_threshold': 50, 'min_fix_duration': .1}
        clf_parameter = {}

        return split_method, split_parameter, clf_parameter

    # noinspection PyMethodMayBeStatic
    def get_feature_parameter(self):
        """ Return which features should be calculated. """

        omit_our = False
        omit_stats = False

        return omit_our, omit_stats

    def trajectory_split_and_feature_basic(self, trajectory: Trajectory, feature_entries=None, verbose=True):

        split_method, split_parameter, classifier_parameter = self.get_split_parameter()
        omit_our, omit_stats = self.get_feature_parameter()

        features = trajectory_split_and_feature_cached(trajectory=trajectory,
                                                       method_name=split_method, method_parameter=split_parameter,
                                                       classifier_parameter=classifier_parameter,
                                                       feature_entries=feature_entries,
                                                       omit_our=omit_our, omit_stats=omit_stats)

        if verbose and features.shape[0] == 0:
            raise Exception("No Features could be extracted!")

        return features

    def provide_feature(self, trajectories: Trajectories, normalize=False, label: str = 'user'):
        """ Generate a list of feature vectors and labels for multiple trajectories

        :param trajectories: [[Samples x 2] x #labels] list of trajectories (x,y) for every label
        :param normalize: normalizing features
        :param label: property of trajectories to label the data
        :return: list of feature matrices for each classifier and list of label matrices
        """

        label_all = getattr(trajectories, label+'s')

        logger = logging.getLogger(__name__)
        logger.info('Getting features for labels: "{}"'.format(label_all))
        time_info = time.time()

        labeled_feature_values = []

        for ii, trajectory in enumerate(trajectories, 1):

            logger.debug('Getting features for user "{}".'.format(trajectory.user))

            # Get a list of features for each classifier. Each list contains multiple feature vectors.
            # feat_values = [dataframe(fix or sac?), dataframe(sac or fix?)]
            # feat_labels = [[ids],[ids]]

            feat_values = self.trajectory_split_and_feature(trajectory)

            len_feat_values = [data.shape[0] for data in feat_values]
            logger.debug('Got {} features.'.format(len_feat_values))

            if any([length == 0 for length in len_feat_values]):
                print('W: {index}. feature of user "{user}" is empty!'.format(index=len_feat_values.index(0)+1,
                                                                              user=trajectory.user))
                logger.warning('{index}. feature of user "{user}" is empty!'.format(index=len_feat_values.index(0),
                                                                                    user=trajectory.user))
                # adding NaN will not work. Maybe could be evaluated afterwards
                # for length, idd in enumerate(len_feat_values):
                #     if length == 0:
                #         feat_values[idd].loc[0] = [np.nan for _ in range(len(feat_values[idd].columns))]
                # # mark user to skip

            # fist iteration, create
            #   Could be before the for-loop but later, when we don't know the amount of feature sets,
            #   it has to be here.
            if len(labeled_feature_values) == 0:
                # repeat empty lists
                labeled_feature_values = [[] for _ in range(len(feat_values))]

            # merge list of features into
            for jj, f_values in enumerate(feat_values):

                if len(f_values) == 0:
                    continue

                f_values[FeatureLabels.user] = trajectory.user
                f_values[FeatureLabels.case] = trajectory.case
                f_values[FeatureLabels.label] = getattr(trajectory, label)
                f_values[FeatureLabels.classifier] = 'Classifier {}'.format(jj)
                # use only FeatureLabels to be sure, they are dropped later

                labeled_feature_values[jj].append(f_values)  # this list has single entries

            if time.time() - time_info > self.info_time:
                print('Features {act} of {total}. Will finish in about {sec:.0f} seconds. ETA: {eta}'.format(
                    act=ii, total=len(label_all), sec=len(label_all)/ii*(time.time() - time_info),
                    eta=time.strftime('%H:%M:%S', time.localtime(time.time()
                                                                 + len(label_all) / ii
                                                                 * (time.time() - time_info)
                                                                 ))))
                time_info = time.time()

        # concatenate list of DataFrames to a single Dataframe
        for jj in range(len(labeled_feature_values)):
            labeled_feature_values[jj] = pd.concat(labeled_feature_values[jj])

            if normalize:
                # noinspection PyUnresolvedReferences
                #   it is filled with pd.DataFrame
                column_names = labeled_feature_values[jj].columns
                for feature_label in FeatureLabels:
                    column_names = column_names.drop(feature_label)

                logger.debug('normalizing')
                scaler = StandardScaler()  # the range of the data value

                labeled_feature_values[jj][column_names] = scaler.fit_transform(
                    labeled_feature_values[jj][column_names])

        # implement features using overview of subsets
        labeled_feature_values = self.get_second_level_features(labeled_feature_values)

        # Save feature names and Output for logfile (for each classifier)
        #   number of entries and features
        #   Names of features
        self.feature = []
        for jj in range(len(labeled_feature_values)):
            column_names = labeled_feature_values[jj].columns
            for feature_label in FeatureLabels:
                column_names = column_names.drop(feature_label)

            # noinspection PyUnresolvedReferences
            logger.debug('Classifier {jj}: {entries} entries for {feature} feature'.format(
                jj=jj, entries=labeled_feature_values[jj].shape[0],
                feature=labeled_feature_values[jj].shape[1]-2))  # -2 for user and label

            self.feature.append(column_names)
            logger.debug('Names of features: {}'.format(column_names))

        self.users = list(set(trajectories.users))

        logger.info('Features finished.')

        return labeled_feature_values

    # noinspection PyMethodMayBeStatic
    def get_second_level_features(self, labeled_feature_values):
        """ add other features columns depending on all feature rows """
        return labeled_feature_values

    @abstractmethod
    def train(self, labeled_feature_values):
        pass

    @staticmethod
    def load_trajectories(dataset, limit=None):
        """ Load training and test data

        :param dataset: Dataset instance
        :param limit:
            int, default=None, optional

            Limit the number of samples (applied before the split)

        :return:
            list, list, list, list

            train and test trajectories, train and test labels
        """

        # :param partitions: int, default=1, optional
        #  Split individual trajectories into multiple

        x_train, y_train = dataset.load_training()
        x_test, y_test = dataset.load_testing()

        if limit:
            step = len(x_train) // limit
            return x_train[::step], x_test[::step], y_train[::step], y_test[::step]

        return x_train, x_test, y_train, y_test

    def predict_features(self, feature_values_user: list, clfs: list, return_subsets: bool = False) -> list:
        """ Predict a single trajectory for a list of features and associated classifier.

        :param feature_values_user: feature_values for one user
        :param clfs:
            classifiers

            list of trained sklean classifier
            (Some Methods will introduce there own classifier. They have to be passed anyway.)

        :param return_subsets:
            boolean

            will return results for every single row in feature_values_user

        :return:
            list of lists (of lists)

            list of probability vectors for each classifier, each class (and each feature row - 'subset')
        """

        if self.n_top_features:
            # prepare reduced feature_values
            for idd, feature_values_user_clf, top_features in enumerate(zip(feature_values_user, self.n_top_features)):
                feature_values_user[idd] = feature_values_user_clf[:, top_features]

        # calculate_features
        predicted_probs = []
        for clf, feature_values_user_clf in zip(clfs, feature_values_user):
            if len(feature_values_user_clf) > 0:
                # probability for each feature entry for actual classifier
                #   feature_values_user_clf: DataFrame with segment ids (row) and different features (col)
                #   clf_feature_proba: ndarray with (row) for each segment and (col) for each trained user/label
                clf_feature_proba: np.ndarray = clf.predict_proba(feature_values_user_clf)

                if return_subsets:
                    # probability for each possible case [case1, case2, ...] X [row1, row2, ...]
                    predicted_probs.append(clf_feature_proba)
                else:
                    predicted_probs.append(np.mean(clf_feature_proba, axis=0))
            else:
                if return_subsets:
                    predicted_probs.append(np.zeros(clf.classes_.shape[0], feature_values_user_clf.shape[0]))
                else:
                    predicted_probs.append(np.zeros(clf.classes_.shape))

        return predicted_probs

    def get_classifiers_and_names(self) -> tuple:
        """ Will be overwritten from evaluation methods.

        :return: list of classifiers and list of names
        """
        if isinstance(self.clf, list):
            return self.clf, ['Classifier {}'.format(ii) for ii in range(len(self.clf))]
        else:
            return [self.clf], ['SingleClassifier']

    def get_feature_names(self) -> list:
        """ Return a list of features used by evaluation method.

        :return: list of feature names
        """
        if isinstance(self.feature, list):
            return self.feature
        else:
            return [self.feature]

    def evaluation(self, feature_values) -> dict:
        return self.weighted_evaluation(labeled_feature_values=feature_values)

    def weighted_evaluation(self, labeled_feature_values, weights: list = None) -> dict:
        """ weights for different classifier (e.g. one for fixation and one for saccades

        :param labeled_feature_values: pandas.DataFrame with feature values and labels for each row
        :param weights: to weight each classifier
        :return: dict with results
        """

        start_time = datetime.datetime.now()
        logger = logging.getLogger(__name__)
        logger.info('weighted_evaluation started')

        # return pd.DataFrame
        #   index: users
        #   columns: classes to predict
        comb_pred, clfs, ys_predicted_both_clf, classes = self.weighted_predict_proba(labeled_feature_values, weights)
        ys_predicted_sac_clf = ys_predicted_both_clf[:, 0]
        ys_predicted_fix_clf = ys_predicted_both_clf[:, 1]
        
        ys_true = []
        users = sorted(list(set(labeled_feature_values[0][FeatureLabels.user])))
        for user in users:
            labeled_feature_values_user = labeled_feature_values[0].loc[
                labeled_feature_values[0][FeatureLabels.user] == user]
            y_true = set(labeled_feature_values_user[FeatureLabels.label])
            if len(y_true) > 1:
                raise Exception('Should not be')
            ys_true.append(list(y_true)[0])
        assert comb_pred.shape == (len(ys_true), len(clfs[0].classes_))

        # put the true label to the users
        predictions = pd.DataFrame(ys_true, index=users, columns=['truth'])
        if predictions['truth'][0] == 'healthy' or predictions['truth'][0] == 'dyslexic':
            Clas_1 = 'dyslexic'
            Clas_2 = 'healthy'
        elif predictions["truth"][0] == 'M' or predictions["truth"][0] == 'F': #changed
            Clas_1 = 'F'
            Clas_2 = 'M'
        else:
            raise Exception('This should not happen.')
        # optimal threshold code: comment it when I want to use multi class classification
        if self.use_optimal_threshold:
            if len(comb_pred.columns)==2:
                predicted_label_list = []
                for index, row in comb_pred.iterrows():
                    if row[Clas_2] > self.binary_classifier_threshold:   # the optimal threshold for each run
                        # if(row['M'] > threshold):    # the average optimal threshold for all the runs
                        predicted_label_list.append(Clas_2)
                    else:
                        predicted_label_list.append(Clas_1)
                comb_pred_with_threshold = pd.DataFrame(predicted_label_list, list(comb_pred.index.values))
                predictions['predicted'] = comb_pred_with_threshold
            else:
                raise ValueError('Optimal threshold does not work yet with multi-class classification problems.'
                                 'It works only with binary classifier.')
        else:
            predictions['predicted'] = comb_pred.idxmax(axis=1)  ### the max_prob is deciding the class
            ys_predicted_sac_clf = pd.DataFrame(ys_predicted_sac_clf, index=users, columns=classes)
            ys_predicted_fix_clf = pd.DataFrame(ys_predicted_fix_clf, index=users, columns=classes)
            predictions['Sacc predicted'] = ys_predicted_sac_clf.idxmax(axis=1)
            predictions['Fix predicted'] = ys_predicted_fix_clf.idxmax(axis=1)
        ############################################################
        ys_true = list(predictions['truth'])
        ys_predicted = list(predictions['predicted'])
        ys_predicted_sac = list(predictions['Sacc predicted'])
        ys_predicted_fix = list(predictions['Fix predicted'])
        print("ys: true vs. predicted"),
        print(ys_true),
        print(ys_predicted),
        logger.debug('ys_true:\n{}'.format(ys_true)),
        logger.debug('ys_predicted:\n{}'.format(ys_predicted)),
        results = {'Accuracy': accuracy_score(ys_true, ys_predicted),
                    'Accuracy_balanced': balanced_accuracy_score(ys_true, ys_predicted),
                    'Accuracy_sac': accuracy_score(ys_true, ys_predicted_sac),
                    'Accuracy_balanced_sac': balanced_accuracy_score(ys_true, ys_predicted_sac),
                    'Accuracy_fix': accuracy_score(ys_true, ys_predicted_fix),
                    'Accuracy_balanced_fix': accuracy_score(ys_true, ys_predicted_fix)}
        print('Accuracy', balanced_accuracy_score(ys_true, ys_predicted))
        print('Accuracy_sac', balanced_accuracy_score(ys_true, ys_predicted_sac))
        print('Accuracy_fix', balanced_accuracy_score(ys_true, ys_predicted_fix))

        def Find_Optimal_ROC_Cutoff(target, predicted): ##optimal threshold code: comment it when I want to use multi class classification
            """ Find the optimal probability cutoff point for a classification model related to event rate
            Parameters

            target : Matrix with dependent or target data, where rows are observations
            predicted : Matrix with predicted data, where rows are observations
            Returns: list type, with optimal cutoff value
            """
            fpr, tpr, threshold = roc_curve(target, predicted)
            i = np.arange(len(tpr))
            roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
            roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

            return list(roc_t['threshold'])

        # if there are more test samples per user, at this place the predictions were already combined (should be)
        print(results)
        logger.debug('Results: {}'.format(results))

        # Add Prediction to result
        results['True_Lable'] = ys_true
        results['Predicted_Lable'] = ys_predicted
        # only work with binary classification
        mylist = results['True_Lable']  # extract the available labels (for e.g. M or F, dyslexic or healthy) in a list
        myset = set(mylist)  # extract the unique labels in a set

        if len(myset) == 2:   # this should only work if we are using binary classification
            # use the labels to carry the results in the results variable
            f_prob = []
            # print("For debug", comb_pred)
            for i in comb_pred[Clas_1]:    # Clas_1 = 'F' or 'D'
                f_prob.append(i)
            results[Clas_1 + '_prob'] = f_prob

            m_prob = []
            for i in comb_pred[Clas_2]:
                m_prob.append(i)
            results[Clas_2 + '_prob'] = m_prob

            first_label = myset.pop()
            second_label = myset.pop()
            results[first_label] = comb_pred[first_label]   # take probabilities associated with 1st unique label
            results[second_label] = comb_pred[second_label]    # take probabilities associated with 2nd unique label
            #################################################################### comment it when I want to use multi class classification
            ##to calculate precision_score convert, recall_score, f1_score we have to convert  F to 0 and M to 1 in the lists of y_true and y_probas
            y_true_new = []
            y_predict_new = []
            for y in ys_true:  # True_Label
                if y == Clas_1:
                    y = 0
                else:
                    y = 1
                y_true_new.append(y)
            for y in ys_predicted:   # Predicted_Label
                if y == Clas_1:
                    y = 0
                else:
                    y = 1
                y_predict_new.append(y)

            preci_score = precision_score(y_true_new, y_predict_new)
            recal_score = recall_score(y_true_new, y_predict_new)
            f1_scor = f1_score(y_true_new, y_predict_new)

            results['precision_score'] = preci_score
            results['recall_score'] = recal_score
            results['f1_score'] = f1_scor
            results['area_under_ROC_curve'] = roc_auc_score(ys_true, m_prob) # roc_auc_score(y_true_label, predicted_prob_of_class1)
            results['best_threshold_ROC'] = Find_Optimal_ROC_Cutoff(y_true_new, m_prob)
            results['Accuracy'] = accuracy_score(ys_true, ys_predicted)
            results['Accuracy_balanced'] = balanced_accuracy_score(ys_true, ys_predicted)
            results['Accuracy_sac'] = accuracy_score(ys_true, ys_predicted_sac)
            results['Accuracy_balanced_sac'] = balanced_accuracy_score(ys_true, ys_predicted_sac)
            results['Accuracy_fix'] = accuracy_score(ys_true, ys_predicted_fix)
            results['Accuracy_balanced_fix'] = balanced_accuracy_score(ys_true, ys_predicted_fix)
            results['avg_train_accuracy_balanced_fix_sac'] = (results['Accuracy_balanced_fix']+results['Accuracy_balanced_sac'])/2
        ##################################################################################

        print("evaluation time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))
        logger.info('weighted_evaluation finished after {} seconds'.format(datetime.datetime.now() - start_time))

        # plot confusion Matrix
        if self.plot_confusion_matrix:
            ax, plt = plot_confusion_matrix(ys_true, ys_predicted, clfs.classes_, users)
            plt.show()

        return results

    @staticmethod
    def combine_predictions_weighted(ys_predicted: np.ndarray, extra_weights: list) -> np.ndarray:
        """ combine the classifier results: and weight by feature count per classifier

        :param ys_predicted: 3D ndarray with predictions

            0: users
            1: classifiers
            2: classes

        :param extra_weights: value for each classifier (sum up to 1)
        :return: one row prediction (columns=classes)
        """

        number_clfs = ys_predicted.shape[1]
        assert len(extra_weights) == number_clfs

        number_users = ys_predicted.shape[0]

        total_count = number_users * number_clfs
        per_classifier_weight = number_users / total_count

        # weight classifier by number of classifiers
        # still this would reduce the values, because extra_weights does the same
        per_classifier_weight = 1 / number_clfs

        # calculate the sum of weighted predictions of all classifiers
        comb_pred = np.asarray([])
        for ii in range(number_clfs):
            # weighted prediction of the ii-th classifier
            weighted_predictions = extra_weights[ii] * ys_predicted[:, ii]

            if len(comb_pred) == 0:
                comb_pred = weighted_predictions
            else:
                comb_pred += weighted_predictions

        return comb_pred

    def weighted_predict_proba(self, labeled_feature_values, weights=None):
        """ Predict class probabilities with all settings of the evaluation method.

        :param labeled_feature_values: pandas.DataFrame with feature values and labels for each row
        :param weights: to weight the classifiers to each other
        :return:
            ndarray(samples x labels), list of clfs

            Class probability for all samples
        """
        logger = logging.getLogger(__name__)

        clfs, clf_names = self.get_classifiers_and_names()

        # Generate classifier names if necessary
        if clf_names is None:
            if len(clfs) == 1:
                clf_names = ['Classifier']
            else:
                clf_names = []
                for ii in range(len(clfs)):
                    clf_names.append('Classifier {}'.format(ii))

        # Generate equal weights if necessary
        if weights is None:
            weights = [1 / len(clfs)] * len(clfs)

        assert len(clfs) == len(weights)
        assert len(clfs) == len(labeled_feature_values)
        assert len(clf_names) == len(clfs)

        # we need the same classes in all classifiers
        classes = np.asarray(clfs[0].classes_)
        for clf in clfs[1:]:
            if not all(classes == np.asarray(clf.classes_)):
                raise Exception('Expected: {should}\nGot: {reality}'.format(should=classes,
                                                                            reality=np.asarray(
                                                                                clf.classes_)))
        msg = f"Evaluating {len(classes)} classes on {len(labeled_feature_values)} classifiers"
        logger.debug(msg)
        print(msg)

        users = sorted(list(set(labeled_feature_values[0][FeatureLabels.user])))
        # for each test sample (trajectory), ys_predicted contains u ndarrays with shape=kxc elements each.
        # u: number of users
        #   k: number of classifiers for this method
        #   c: number of classes

        # noinspection PyTypeChecker
        ys_predicted = np.asarray(
            [
                self.predict_features(
                    [
                        labeled_feature_values[ii].loc[
                            labeled_feature_values[ii][FeatureLabels.user] == user
                            ].drop(columns=list(FeatureLabels))
                        for ii in range(len(labeled_feature_values))
                    ], clfs)
                for user in users
            ])

        # combine the classifier results: weight by feature count per classifier
        comb_pred = self.combine_predictions_weighted(ys_predicted, weights)

        # comb_pred has now one row for each user and one column for each class
        #   u x k
        comb_pred = pd.DataFrame(comb_pred, index=users, columns=classes)

        return comb_pred, clfs, ys_predicted, classes

    # noinspection PyDefaultArgument
    def top_features(self, labeled_feature_values, seed, n=[8, 8]):
        """ To find the top n features using Random Forrest for any type of features.

        (saccade, fixation and general feature)
        These features can be used to further train any classifier like RF or RBFN.

        :param labeled_feature_values: feature values
        :param seed:
        :param n:
        :return:
          list of lists containing top features for each different features (like saccade, fixation)
          storing it in a class variable
        """

        clf = RandomForestClassifier(random_state=seed)
        print("Training for selecting top features")
        for i, labeled_feature in enumerate(labeled_feature_values):
            if self.__class__.__name__ == 'OurEvaluationOne' or self.__class__.__name__ == 'EvaluationWindowed':
                # clf = sklearn.base.clone(clf)
                n_top = n[0]
            else:
                # clf = sklearn.base.clone(clf)
                n_top = n[i]
            # noinspection PyTypeChecker
            clf.fit(labeled_feature.drop(columns=list(FeatureLabels)),
                    labeled_feature[FeatureLabels.label])
            # noinspection PyTypeChecker
            self.n_top_features.append(important_features(clf, labeled_feature.drop(columns=list(FeatureLabels)),
                                                          n=n_top))
            # important_features: this functions sorts and give the top n features from a trained RF model

    def ref_top_features(self, feature_values, seed, n=None):

        if n is None:
            n = [8, 8]

        xs = []
        ys = []
        for ii in range(2):
            # noinspection PyTypeChecker
            xs.append(feature_values[ii].drop(columns=list(FeatureLabels)))
            ys.append(feature_values[ii][FeatureLabels.label])

        # x = x_train
        # y = y_train
        estimator = RandomForestClassifier(n_estimators=10, random_state=seed)
        selector = RFECV(estimator, step=1, cv=10)
        print("Training for selecting top features")
        for i, (X, y) in enumerate(zip(xs, ys)):
            if self.__class__.__name__ == 'OurEvaluationOne' or self.__class__.__name__ == 'EvaluationWindowed':
                n_top = n[0]

            else:
                n_top = n[i]
            selector = selector.fit(X, y)
            ranks = selector.ranking_
            feature_ranks_with_idx = enumerate(ranks)
            sorted_ranks_with_idx = sorted(feature_ranks_with_idx, key=lambda x: x[1])
            top_col = [idx for idx, rnk in sorted_ranks_with_idx[:n_top]]
            self.n_top_features.append(X.columns[top_col].array)
            # print(i, top_features.shape, selector.n_features_, selector.ranking_, top_feature_names)

    @staticmethod
    def separate_feature_labels(labeled_feature_values: pd.DataFrame) -> tuple:
        """ Separate feature label DataFrame in values and labels and remove NaN from features.

        :param labeled_feature_values: feature values with label columns
        :return: one DataFrame with feature values and one with labels
        """
        feature_values = []
        feature_labels = []
        for ii in range(len(labeled_feature_values)):
            # noinspection PyTypeChecker
            feature_values.append(labeled_feature_values[ii].drop(columns=list(FeatureLabels)))
            feature_labels.append(labeled_feature_values[ii][FeatureLabels.label])

            # remove NaN
            isnan = feature_values[ii].isna().max(axis=1)
            feature_values[ii] = feature_values[ii][~isnan]
            feature_labels[ii] = feature_labels[ii][~isnan]

        return feature_values, feature_labels

    def select_top_features(self, feature_values: pd.DataFrame) -> pd.DataFrame:
        """ Reduce columns of feature_values to top features stored in class

        :param feature_values: pandas.DataFrame with values for different features
        :return: reduced DataFrame (less columns)
        """
        if self.n_top_features is not None:
            # if train_top_features argument is passed with value > 0, then this condition will be runned
            for ii, (features, top_feature_names) in enumerate(zip(feature_values, self.n_top_features)):
                # will select top n feature columns from the training data
                # print(top_f) #top features
                feature_values[ii] = features.loc[:, top_feature_names]

        return feature_values

    def do_data_preparation(self, trajectories: Trajectories) -> Trajectories:

        trajectories.convert_to(self.preparation_steps['conversion'])
        trajectories.apply_filter(filter_type=self.preparation_steps['filter_type'],
                                  **self.preparation_steps['filter_parameter'])

        return trajectories
