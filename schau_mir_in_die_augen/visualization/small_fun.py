import numpy as np
import pandas as pd

from bokeh.colors import RGB


def find(element_list, n_elements=0):
    """ Find indices of nonzero elements.
    :param element_list: list with elements
    :param n_elements: number of non zero elements to return
    :return: index of non zero elements
    """
    # https://www.mathworks.com/help/matlab/ref/find.html

    non_zero_elements = []

    for counter, element in enumerate(element_list):
        if element:
            non_zero_elements.append(counter)
            if len(non_zero_elements) == n_elements:
                return non_zero_elements

    return non_zero_elements


def rgb2hex(color_rgb: (tuple, list), scaler: float = 1) -> str:
    """ transform 3 floats 0 to 1 to hash string"""
    assert len(color_rgb) == 3
    return '#%02x%02x%02x' % (int(min(color_rgb[0] * 255 * scaler, 255)),
                              int(min(color_rgb[1] * 255 * scaler, 255)),
                              int(min(color_rgb[2] * 255 * scaler, 255)))


def get_next_in_list(elements: list, element):
    if element not in elements:
        raise Exception('"{}" is not in "{}"!'.format(element, elements))
    id_element = elements.index(element)

    if id_element == len(elements) - 1:
        id_element = 0
    else:
        id_element += 1

    return elements[id_element]


def scale_color(color_values, factor: float):
    """ reduce colors to black and white or push them to have nearly only full red, full green, or full blue """

    factor = float(factor * factor)

    red = color_values.view(dtype='uint8')[:, 0::4]
    green = color_values.view(dtype='uint8')[:, 1::4]
    blue = color_values.view(dtype='uint8')[:, 2::4]

    gray_value = (red / 3 + green / 3 + blue / 3)
    gray_max_factor = 255 / (gray_value.max() - gray_value.min()) if gray_value.max() - gray_value.min() > 0 else 1

    gray_value = (gray_value - gray_value.min()) * gray_max_factor * (1 - factor)

    red = red * factor + gray_value
    green = green * factor + gray_value
    blue = blue * factor + gray_value

    red[red < 0] = 0
    green[green < 0] = 0
    blue[blue < 0] = 0

    red[red > 255] = 255
    green[green > 255] = 255
    blue[blue > 255] = 255

    color_values.view(dtype='uint8')[:, 0::4] = np.uint8(red)
    color_values.view(dtype='uint8')[:, 1::4] = np.uint8(green)
    color_values.view(dtype='uint8')[:, 2::4] = np.uint8(blue)

    return color_values


def fade_white(color_values, visibility: float):
    """ fade out to a white image """

    assert 0 <= visibility <= 1

    red = color_values.view(dtype='uint8')[:, 0::4]
    green = color_values.view(dtype='uint8')[:, 1::4]
    blue = color_values.view(dtype='uint8')[:, 2::4]

    red = red * visibility + 255 * (1 - visibility)
    green = green * visibility + 255 * (1 - visibility)
    blue = blue * visibility + 255 * (1 - visibility)

    color_values.view(dtype='uint8')[:, 0::4] = np.uint8(red)
    color_values.view(dtype='uint8')[:, 1::4] = np.uint8(green)
    color_values.view(dtype='uint8')[:, 2::4] = np.uint8(blue)

    return color_values


def alpha_pallet(final_color: tuple, steps: int, opacity: float = 1) -> list:
    """ returns a color pallet from transparent to full color in given steps
    :param final_color: (r,g,b) 0-255
    :param steps: number of steps (n+1 colors will be returned)
    :param opacity: maximal alpha value (0-1]

    :return list with rgba color tuples
    """

    assert steps >= 0
    assert isinstance(steps, int)
    assert 0 < opacity <= 1

    if steps == 0:
        stepsize = 0
    else:
        stepsize = 1 / steps

    color_list = []

    for istep in range(steps + 1):
        color_list.append(RGB(final_color[0], final_color[1], final_color[2], istep * stepsize * opacity))

    return color_list


# dealing with votes #
######################
# todo: this should be somwhere else and be reused

def get_major_vote(labeled_feature_probabilities: pd.DataFrame) -> tuple:
    """ Calculate best performance of columns by row based vote for highest value

    :param labeled_feature_probabilities:
    :return: single winner [str], pd.Series with performance of all
    """

    subset_vote = labeled_feature_probabilities.idxmax('columns').value_counts()

    # add participants with zero votes for completeness
    for participant in set(labeled_feature_probabilities) - set(subset_vote.keys()):
        # brute force, don't know better
        subset_vote[participant] = 0

    return get_winner(subset_vote), subset_vote


def get_mean_vote(labeled_feature_probabilities: pd.DataFrame) -> tuple:
    """ Calculate best performance of columns by mean over rows

    :param labeled_feature_probabilities:
    :return: single winner [str], pd.Series with performance of all
    """

    mean_values = labeled_feature_probabilities.mean().sort_values(ascending=False)
    return get_winner(mean_values), mean_values


def get_winner(sorted_series: pd.DataFrame) -> str:
    """ Select the first entry of list, but check if it is the same with the second """

    if len(sorted_series) > 1 and sorted_series[0] == sorted_series[1]:
        print('W: multiple labels with maximal value! Took first.')

    return sorted_series.index[0]


# todo: move this somewhere usefull
def get_prediction(labeled_feature_values: pd.DataFrame, evaluator,
                   clfs: list, feature_names: list) -> tuple:
    """ calculate prediction for all users
            with given feature values
            on given evaluation method
            with given classifier and selected features

    :param labeled_feature_values: values for features with column "sample_type" for one user
    :param feature_names: list with names of features ... according to clf?
    :param evaluator: trained classifier in some evaluation object?
    :param clfs: also trained classifier? (Necessary since BaseEvaluation doesn't have classifiers)

    :return predicted_probabilities for all users at all segments
    :return updated type_names
    :return valuation_results for specific user at all segments
    """

    # modified type_names (could be "ignored" segments)
    type_names = labeled_feature_values['sample_type']
    # select only valid feature values
    labeled_feature_values = labeled_feature_values[type_names != 'ignored']

    # todo: remove this hard code
    clf_hard_names = ['saccade', 'fixation']

    print('Calculating probabilities for labels.')
    predicted_probs = evaluator.predict_features(
        [labeled_feature_values.loc[
             labeled_feature_values['sample_type'] == clf_hard_names[idd], feature_names[idd]
         ] for idd in range(len(clfs))],
        clfs, return_subsets=True)

    # predicted_probs
    #   list: DataFrame for each classifier (e.g. Saccades and Fixations)
    #       columns: trained labels
    #       rows: subsets (e.g. saccades or fixation)
    # now, combining
    predicted_probs = [pd.DataFrame(
        predicted_probs[idd],
        index=labeled_feature_values.index[
            labeled_feature_values['sample_type'] == clf_hard_names[idd]],
        columns=clfs[0].classes_
    ) for idd in range(len(clfs))]

    # combine dataframes from different classifiers
    predicted_probs = pd.concat(predicted_probs)
    # add sample type to prediction list (order is maintained by pandas)
    predicted_probs['sample_type'] = type_names
    # add unpredicted rows
    predicted_probs = predicted_probs.append(
        type_names[type_names == 'ignored'].to_frame(name='sample_type'),
        sort=False)  # sorting the columns is not necessary

    return predicted_probs, list(type_names)


def calc_prediction_correctness_value(predicted_probs: pd.DataFrame, user: str) -> pd.Series:
    """ Calculate the prediction correctness value for one user at all given segments

    :param predicted_probs: predicted_probabilities for all users at all segments
    :param user: name of user to evaluate
    :return: (result_user - result_best) for every segment as pd.Series
    """
    # calculate results for user and best result of other users for each segment
    result_user = predicted_probs[user]
    result_best = predicted_probs.loc[:, predicted_probs.columns != user].max(axis=1)

    return result_user - result_best


def count_fixation_in_paragraphs(cog_fix_x: list, cog_fix_y: list) -> dict:
    """ Count fixations in different paragraphs of the bio-tex stimulus.

    8 regions (1 header, and 6 paragraphs, outside) are possible

    :param cog_fix_x: x values of the identified fixations
    :param cog_fix_y: y values of the identified fixations
    :return: dict with Paragraph names and number of fixations in it.
    """

    fixation_dict = {'Paragraph_H': 0,
                     'Paragraph_1': 0, 'Paragraph_2': 0, 'Paragraph_3': 0,
                     'Paragraph_4': 0, 'Paragraph_5': 0, 'Paragraph_6': 0,
                     'Outside': 0}

    for cog_x, cog_y in zip(cog_fix_x, cog_fix_y):

        # 490 left side
        # 1175 right side
        # + 60 Margin
        if 490 - 60 > cog_x or cog_x > 1175 + 60:
            fixation_dict['Outside'] += 1
            continue

        if cog_y > 1200:
            fixation_dict['Outside'] += 1

        elif cog_y > 998:
            fixation_dict['Paragraph_H'] += 1

        elif cog_y > 832:
            fixation_dict['Paragraph_1'] += 1

        elif cog_y > 668:
            fixation_dict['Paragraph_2'] += 1

        elif cog_y > 502:
            fixation_dict['Paragraph_3'] += 1

        elif cog_y > 338:
            fixation_dict['Paragraph_4'] += 1

        elif cog_y > 170:
            fixation_dict['Paragraph_5'] += 1

        elif cog_y > -150:
            fixation_dict['Paragraph_6'] += 1

        else:
            fixation_dict['Outside'] += 1

    return fixation_dict
