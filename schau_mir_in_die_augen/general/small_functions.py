import pandas as pd


def select_probabilities(labeled_predicted_probabilities: dict, user: str, classifier: str):
    """ Select probabilities from different users and specific classifier

    :param labeled_predicted_probabilities: dict with DataFrames of prediction values for each user
    :param user: Name of selected user
    :param classifier: Name of classifier
        'all mean': mean of all probabilities for this user
        'clf mean': mean of the classifier results (differs when classifiers have different numbers of subsets)
        'fix': only fixation (hardcode for now)
        'sac': only saccades (hardcode for now)

    """

    labeled_predicted_probabilities_user = labeled_predicted_probabilities[user]

    # todo: fix this hard code - atm only for saccades/fixations - names should be the same
    clf_hard_names = ['saccade', 'fixation']

    if classifier == 'all mean':
        selected_probabilities = labeled_predicted_probabilities_user.drop(columns=['sample_type'])
    elif classifier == 'clf mean':
        # calculate mean for every classifer
        selected_probabilities = pd.concat([
            labeled_predicted_probabilities_user[labeled_predicted_probabilities_user['sample_type'] == clf_name].mean(
                axis=0)
            for clf_name in clf_hard_names], axis=1).T
        selected_probabilities['sample_type'] = clf_hard_names  # this gets dropped by .mean
        selected_probabilities.set_index(keys='sample_type', inplace=True)
    elif classifier == 'sac':
        selected_probabilities = labeled_predicted_probabilities_user[
            labeled_predicted_probabilities_user['sample_type'] == 'saccade'].drop(columns=['sample_type'])
    elif classifier == 'fix':
        selected_probabilities = labeled_predicted_probabilities_user[
            labeled_predicted_probabilities_user['sample_type'] == 'fixation'].drop(columns=['sample_type'])
    else:
        raise Exception('"{}" is not a valid choice!'.format(classifier))

    return selected_probabilities
