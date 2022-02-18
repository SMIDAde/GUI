import sys
import os
import webbrowser

import numpy as np
import pandas as pd

from scipy.ndimage.filters import gaussian_filter

from schau_mir_in_die_augen.visualization.delayfun import run_delayed
from schau_mir_in_die_augen.visualization.small_fun import find, rgb2hex, get_next_in_list, scale_color, fade_white, \
    get_major_vote, get_mean_vote, get_prediction, calc_prediction_correctness_value, alpha_pallet, \
    count_fixation_in_paragraphs
from schau_mir_in_die_augen.general.small_functions import select_probabilities
from schau_mir_in_die_augen.general.io_functions import save_parameter_file, replace_local_path, get_local_files, \
    load_pickle_file, load_parameter_file

from schau_mir_in_die_augen.feature_extraction import all_features, calculate_features_cached

from schau_mir_in_die_augen.evaluation.evaluation_general import EvaluationGeneralFixSacNew
from schau_mir_in_die_augen.evaluation.base_selection import method_list, get_method, get_classifier, \
    get_parser_method, get_parser_eye_movement_classifier, get_parser_classifier
from schau_mir_in_die_augen.evaluation.script_helper import call_train, prepare_input
from scripts.inspection import check_scikit

from schau_mir_in_die_augen.datasets.dataset_loader import dataset_list

from schau_mir_in_die_augen.process.trajectory import Trajectories, Trajectory
from schau_mir_in_die_augen.trajectory_classification.trajectory_split import eye_movement_classification, \
    EyeMovementClassifier

# image loading
from PIL import Image

from bokeh.events import Tap, ButtonClick
from bokeh.layouts import layout
from bokeh.models import RangeSlider, MultiSelect, Select, Toggle, Slider, Button, DataTable, \
    TableColumn, ColumnDataSource, Div, NumberFormatter, CDSView
from bokeh.models.widgets import ColorPicker, TextInput, RadioButtonGroup
from bokeh.models.filters import GroupFilter
from bokeh.plotting import figure, curdoc
# noinspection PyUnresolvedReferences
from bokeh.palettes import Category20, brewer

sys.path.append(os.path.join(os.getcwd()))
check_scikit()  # Maybe helpful for loading Classifiers

debugMode = True

################
# Subfunctions #
################

# noinspection SpellCheckingInspection
def load_img_bokeh(fn):
    if isinstance(fn, str):
        if os.path.exists(fn):
            lena_img = Image.open(fn).convert('RGBA')
        else:
            raise Exception('File "{}" to load stimulus does not exist!'.format(fn))
    elif (isinstance(fn, tuple) or isinstance(fn, list)) and len(fn) == 2:
        lena_img = Image.new('RGBA', fn, color='#808080')
    else:
        raise Exception('Bokeh could not load Image. Type "{}" is unknown'.format(type(fn)))

    xdim, ydim = lena_img.size
    # Create an array representation for the image `img`, and an 8-bit "4
    # layer/RGBA" version of it `view`.
    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    # Copy the RGBA image into view, flipping it so it comes right-side up
    # with a lower-left origin
    view[:, :, :] = np.flipud(np.asarray(lena_img))
    return img


def load_data(user: str, case: str, add_duplicate_stats: bool = True) -> Trajectory:
    """ load and modfiy Trajectory of user and specified case

    :param user: name
    :param case: of measurement
    :param add_duplicate_stats: will note the removed duplicates (default=true)
    """

    # todo: order of steps is important - maybe think about this

    # load Trajectory form dataset
    trajectory = dataset.load_trajectory(user, case)

    # scaling
    if data_scaling_slider.value != 1:
        trajectory.scale(np.square(data_scaling_slider.value))

    # centering
    if data_center_toggle.active:
        trajectory.center()

    # interpolating
    if data_interpolate_toggle.active:
        trajectory.interpolate()

    # smooth data
    if smooth_toggle.active:
        trajectory.convert_to('angle_deg')
        trajectory.apply_filter('savgol', frame_size=savgol_filter_frame_slider.value,
                                pol_order=savgol_filter_poly_slider.value)

    # remove duplicates
    if duplicate_toggle.active:
        remove_stats = trajectory.remove_duplicates(duplicate_slider.value)
        if add_duplicate_stats:
            update_duplicate_toggle(remove_stats)

    # bokeh will use pixel most of the time
    # the necessary y-inversion is be done, when pulling the data.
    trajectory.convert_to('pixel')

    return trajectory


def get_colors(n_colors):
    """ Get an array of n colors """
    if n_colors > 11:
        return Category20[n_colors]
    elif n_colors > len(user_colors) or not user_color_toggle.active:
        # make sure the first colors stay constant
        n_stable_colors = max(n_colors, len(user_colors))
        return brewer['Spectral'][n_stable_colors][:n_colors]
    else:
        return user_colors[:n_colors]


def get_user_color(user_id=None):
    """color for selected users"""
    u_color = get_colors(len(user_select.value))
    if user_id is None:
        return u_color
    else:
        return u_color[user_id]


def get_red2green_color(value: float) -> str:
    """ Transform value -1 to 1 to color in hex format to scale from red over white to green.

    Everything smaller or created will be counted as -1 or 1.
    """
    # todo: this could also be done by a colormap. Should be faster.

    if value >= 0:
        # reduce everything, except green
        color_rgb = np.array([1, 1, 1]) - min(value, 1) * np.array([1, 0, 1])
    elif value < 0:
        # reduce everything, except red (value negative)
        color_rgb = np.array([1, 1, 1]) - min(-value, 1) * np.array([0, 1, 1])
    else:
        # NaN case (blau)
        return '#0000ff'

    return rgb2hex(color_rgb)


def get_pcv_color(pcv_values: list) -> list:
    """ calculate the color for the prediction_correctness visualization

    Use prediction_color_slider to overscale.
    """

    return [get_red2green_color(prediction_result * prediction_color_slider.value)
            for prediction_result in pcv_values]


def get_user_name(user_id=None):
    """user name of selected users"""
    user_names = user_select.value
    if user_id is None:
        return user_names
    else:
        return user_names[user_id]


def get_trajectory_info(segments: list) -> dict:
    """ Return array with center of gravity, extant and duration for given segments
    :param segments: array [[[xa1,xa2,...],[ya1,ya2,...]],[[xb1,xb2,...],[yb1,yb2,...]],...]
    :return dictionary with count, cog_x, cog_y, size and duration """

    count_i = []
    cog_x = []
    cog_y = []
    size_s = []
    duration_d = []

    for ii in range(len(segments)):
        if len(segments[ii][:, 0]):
            count_i.append(ii)
            cog_x.append(np.nanmean(segments[ii][:, 0]))
            cog_y.append(np.nanmean(segments[ii][:, 1]))
            max_distance_to_cog = np.sqrt(np.max(np.square(segments[ii][:, 0] - cog_x[ii])
                                                 + np.square(segments[ii][:, 1] - cog_y[ii])))
            size_s.append(max(max_distance_to_cog, marker_minsize))
            duration_d.append(len(segments[ii][:, 0]))

    # print(str(cog_x)+" and "+str(cog_y)+" with Size "+str(size_s))

    return {'count': count_i, 'cog_x': cog_x, 'cog_y': cog_y, 'size': size_s, 'duration': np.array(duration_d)}


def generate_prediction_plot_data(trajectory: Trajectory,
                                  evaluated_eye_movements: tuple,
                                  trajectory_slice: slice) -> dict:
    """ Generate dictionary for plotting.

    :param trajectory: Trajectory of Current participants.
    :param evaluated_eye_movements: segmentized and evaluate eye movements.
    :param trajectory_slice: part of active Trajectory.
    :return:
    """

    # get subset of start_samples, prediction_results and trajectory
    #   start_samples: sample number where a sample starts
    #   type_names: type of a sample
    #   prediction_results: prediction correctness value
    start_samples, type_names, prediction_results = evaluated_eye_movements
    xy = trajectory.get_trajectory('pixel_image')

    assert start_samples[0] <= trajectory_slice.start
    assert trajectory_slice.start < trajectory_slice.stop
    assert start_samples[-1] >= trajectory_slice.stop - 1

    # brute force
    #   this could be very slow
    xys = []
    prediction_results_new = []
    type_names_new = []
    start_samples_new = [trajectory_slice.start]
    # First sample index
    #   caseA1: start_samples[0] < trajectory_slice.start start_samples_new[0] should be trajectory_slice.start
    #   caseA2: start_samples[0] = trajectory_slice.start start_samples_new[0] should be trajectory_slice.start
    #   caseA3: start_samples[0] > trajectory_slice.start not possible (see assert)
    for s_id, start_next_sample in enumerate(start_samples[1:]):

        if start_next_sample <= trajectory_slice.start:
            # samples before start of trajectory are ignored
            continue

        if start_next_sample < trajectory_slice.stop - 1:
            # caseB0:
            prediction_results_new.append(prediction_results[s_id])
            start_samples_new.append(start_next_sample)
            if predictor_type_select.active:
                type_names_new.append(type_names[s_id])
            else:
                type_names_new.append('fixation')
                # This is a pretty bad hack, to show only fixation.
                # Instead of adapting the filter, I adapt the data ... naughty...
            xys.append(xy[start_samples_new[-2]:start_samples_new[-1]])

        else:
            # caseB1: start_next_sample >= trajectory_slice.stop - 1
            #   this is okay. start_sample define the index of the velocity! (one shorter than trajectory)
            prediction_results_new.append(prediction_results[s_id])
            start_samples_new.append(trajectory_slice.stop - 1)
            type_names_new.append(type_names[s_id])
            xys.append(xy[start_samples_new[-2]:start_samples_new[-1]])
            break

    new_prediction_data = {'xs': [xy[:, 0] for xy in xys],
                           'ys': [xy[:, 1] for xy in xys],
                           'line_color': get_pcv_color(prediction_results_new),
                           'line_type': type_names_new}

    return new_prediction_data


def crop_trajectory(xy, t_id):
    """ Return a copy with all values outside the image set to NaN. """

    v = np.array(xy)

    if not clip_toggle.active:
        return v

    h, w = stimulus.shape

    # todo: ignore warning that NaN (or Inf) is in the data
    # np.warnings.filterwarnings('ignore')
    # doesn't work, cause default: throws a lot of warnings ...

    # left of zero
    nan_bool = v[:, 0] < 0 + x_offset_slider.value
    number_nan = sum(nan_bool)
    v[nan_bool] = np.NaN

    # right of width
    nan_bool = v[:, 0] > w + x_offset_slider.value
    # noinspection PyTypeChecker
    number_nan += sum(nan_bool)
    v[nan_bool] = np.NaN

    # below zero
    nan_bool = v[:, 1] < 0 + y_offset_slider.value
    number_nan += sum(nan_bool)
    v[nan_bool] = np.NaN

    # higher than height
    nan_bool = v[:, 1] > h + y_offset_slider.value
    # noinspection PyTypeChecker
    number_nan += sum(nan_bool)
    v[nan_bool] = np.NaN

    update_clip_toggle(number_nan, t_id)

    # np.warnings.filterwarnings('default')
    return v


#############
# Parameter #
#############

# global data - I should encapsulate the plot into a class

# get dict with all datasets
datasets = dict()
for dataset in dataset_list:
    datasets.update({dataset.dataset_name: dataset})

# load default dataset
dataset = datasets[dataset_list[0].dataset_name]

# load backup dataset
if not dataset.data_exists:
    print('Default Dataset is not available. Load DemoData instead!')
    dataset = datasets['Random Data']  # Name of DemoDataset()

cases = dataset.get_cases()
users = list(dataset.get_users())

stimulus = load_img_bokeh(dataset.get_stimulus(cases[0]))


# load default data
# we support multiple lines so make it a list
#: Storage for all Trajectories
trajectories = Trajectories()
trajectories_active = [np.asarray([[0], [0]]).T]

# load this a dummy predictor
#   todo: would be better to have this more general
predictor = EvaluationGeneralFixSacNew()

# todo: make dynamic delay according to update time
delay_time_ms = 15

ivt_threshold = 100
ivt_min_dur = 0.1
savgol_frame_size = 7
savgol_poly_order = 1
remove_duplicates_threshold = 5
ivt_slider_delay_velo_ms = 15
ivt_slider_delay_stim_ms = 250

# current subsets for each selected user
#   start_samples, type_names, prediction_results = eye_movements4trajectories[trajectory_id]
eye_movements4trajectories = [([], [], [])]
labeled_predicted_probabilities = dict()
def update_classification():
    global eye_movements4trajectories, labeled_predicted_probabilities

    # update data above eye movements
    labeled_predicted_probabilities = dict()
    eye_movements4trajectories = [evaluate_trajectory(trajectory) for trajectory in trajectories]

    # Update classification selections
    if len(labeled_predicted_probabilities) > 0:
        predictor_classification_select.options = list(labeled_predicted_probabilities)
        predictor_classification_select.value = predictor_classification_select.options[0]
    else:
        predictor_classification_select.options = ['N/A']
        predictor_classification_select.value = 'N/A'

    auto_scale_prediction_color()       # auto scale color
    update_predictor_classifier_text()
    update_bar_source()                 # update bars in velocity plot
    update_fixation_marker_source()     # update marker in stimulus plot
    update_time()


def evaluate_trajectory(trajectory, lazy: bool = True):
    """ Calculates start_samples (frames) and type_names (saccades, fixation, ... ) for a given trajectory.
    
    If necessary, (pcv) prediction is also calculated.

    :param trajectory: Trajectory object.
    :param lazy: will not calculate prediction, when prediction is not visible
    :return: start_samples, type_names, prediction_correctness_values
    """

    start_samples, type_names = eye_movement_classification(
        trajectory,
        EyeMovementClassifier('IVT',
                              {'vel_threshold': ivt_threshold,
                               'min_fix_duration': ivt_min_dur,
                               'sample_rate': dataset.sample_rate},
                              mark_nan=mark_nan_toggle.active))

    type_occurrence = dict(zip(*np.unique(type_names, return_counts=True)))
    p_vel_info.text = '<b>Segmented Parts</b><br>{user}:{segments}'.format(user=trajectory.user,
                                                                           segments=type_occurrence)

    # create prediction_correctness_values
    if (lazy and not p_stim_evaluation.visible) or predictor_select.value == 'None':
        prediction_correctness_values = [0] * len(type_names)
    elif predictor_select.value == 'Test':
        prediction_correctness_values = np.random.rand(len(type_names)) * 2 - 1
    else:
        prediction_correctness_values, updated_type_names = calc_pcv(trajectory, start_samples, type_names)
        type_names = updated_type_names  # explicit to make it important

    return start_samples, type_names, prediction_correctness_values


def calc_pcv(trajectory: Trajectory, start_samples: list, type_names: list) -> tuple:
    """ Calculate the prediction correctness value for one user on multiple segments

    :param trajectory: Trajectory for selected user
    :param start_samples: begin of segments
    :param type_names: type of segments
    :return: list of pcv values for the given segments, list of updated type_names
    """

    global labeled_predicted_probabilities

    if predictor is None:
        raise Exception('No Predictor loaded')

    elif predictor.dataset_name is not None \
            and dataset.dataset_name is not None \
            and predictor.dataset_name != dataset.dataset_name:
        print('W: Selected dataset is "{selected}" but Predictor was trained on "{train}"'.format(
            selected=dataset.dataset_name, train=predictor.dataset_name))

    # get classifiers and features
    clfs, clf_names = predictor.get_classifiers_and_names()
    feature_names = predictor.get_feature_names()

    for (clf, clf_name) in zip(clfs, clf_names):
        if trajectory.user not in clf.classes_:
            # clf.classes_ should be the same for each classifier
            print('Classifier "{clf}" can not predict "{user_gui}". It is only trained for "{user_clf}"'.format(
                clf=clf_name, user_gui=trajectory.user, user_clf=clf.classes_))
            clfs = None
            break

        # todo: Maybe this has to be addapted for gender ... it is not about the user, its about the prediciton
        #   To try to predict a user outside the training data makes no sense.
        #   But to predict a gender outside the training data is completely fine.

    if clfs is None:
        print('Predictor is empty!')
        prediction_results = [0] * len(type_names)
        updated_type_names = type_names

    else:
        print('Predicting features for user "{}".'.format(trajectory.user))

        # load feature values
        labeled_feature_values = calculate_features_cached(trajectory, start_samples, type_names,
                                                           cached=predictor_classifier_cache_toggle.active)

        # predict labels
        #    predicted_probs: DataFrame
        #      row: segments
        #      col: user/participants/labels + type_names
        #      val: probability
        #    type_names: list
        #      type_names with new information (e.g. "ignored" because of NaN or to short or whatever)
        predicted_probs, updated_type_names = get_prediction(
            labeled_feature_values, predictor, clfs, feature_names)
        predicted_probs.sort_index(inplace=True)  # necessary to have same order like start_samples

        # calculate result for one user and all segments
        prediction_results = list(calc_prediction_correctness_value(predicted_probs, trajectory.user))

        # update list of probabilities
        labeled_predicted_probabilities.update({trajectory.user: predicted_probs})

    return prediction_results, updated_type_names


time_bar = []

# Layout
layout_margin = 10
width_three = 1024
width_one = (width_three - 2 * layout_margin) // 3
width_half = (width_one - layout_margin) // 2
width_two = width_one * 2
width_full = width_three + width_two

#########################
# Main visible elements #
#########################


# user color #
##############
def user_color_cb(_attrname, _old, _new):
    global user_colors
    user_colors = [color_picker.color for color_picker in user_color_picker]
    initialize_plotdata()


# colors for user
user_color_toggle = Toggle(label="Use these colors", active=True, width=width_half)
user_color_toggle.on_change('active', user_color_cb)

user_color_picker = []
user_colors = ['yellow', 'limegreen', 'darkorange', 'violet', 'pink', 'cyan', 'skyblue']
for idd in range(len(user_colors)):
    user_color_picker.append(ColorPicker(title="User {}".format(idd), color=user_colors[idd], width=width_half))
    user_color_picker[idd].on_change('color', user_color_cb)


# stimulus plot #
#################
# create a plot and style its properties
#: Stimulus Plot
p_stim = figure(tools="hover,wheel_zoom,pan,save,box_zoom,help,reset,tap",
                match_aspect=True,
                width=width_three, height=width_three, margin=layout_margin // 2)

p_stim.outline_line_color = None
p_stim.grid.grid_line_color = None
# p.xaxis.visible = False
# p.yaxis.visible = False
p_stim.toolbar.autohide = False

# original image as background
p_stim_background = p_stim.image_rgba([stimulus], x=0, y=0, dw=stimulus.shape[1], dh=stimulus.shape[0])

# marker
user_parallel_maximum = 10
p_stim_marker_source = ColumnDataSource(data=get_trajectory_info(
    trajectories.get_trajectories('pixel_image')))
p_stim_fixation_marker_source_single = {'cog_x': [], 'cog_y': [], 'count': [],
                                        'duration': [],
                                        'size': [], 'marker_alpha': [],
                                        'text_size': [], 'text_color': []}
p_stim_fixation_marker_source = [ColumnDataSource(p_stim_fixation_marker_source_single)
                                 for _ in range(user_parallel_maximum)]

p_stim_fixation_mark = []
p_stim_saccades_mark = []

for idd in range(user_parallel_maximum):
    p_stim_fixation_mark.append(p_stim.circle(x='cog_x', y='cog_y',
                                              radius='size', fill_alpha='marker_alpha',
                                              source=p_stim_fixation_marker_source[idd],
                                              line_width=2, visible=False))
    p_stim_fixation_mark[idd].selection_glyph = p_stim_fixation_mark[idd].glyph
    p_stim_fixation_mark[idd].nonselection_glyph = p_stim_fixation_mark[idd].glyph

for idd in range(user_parallel_maximum):
    p_stim_saccades_mark.append(p_stim.line(x='cog_x', y='cog_y',
                                            source=p_stim_fixation_marker_source[idd],
                                            line_width=2,
                                            line_dash="dashed", visible=False))
    p_stim_saccades_mark[idd].selection_glyph = p_stim_saccades_mark[idd].glyph
    p_stim_saccades_mark[idd].nonselection_glyph = p_stim_saccades_mark[idd].glyph

# plot
p_stim_trajectories = p_stim.multi_line(xs=[], ys=[], line_color=[], line_width=2)
p_stim_trajectories.selection_glyph = p_stim_trajectories.glyph
p_stim_trajectories.nonselection_glyph = p_stim_trajectories.glyph
p_stim_trajectory_marker = p_stim.asterisk(x=[], y=[], line_color=[], size=4)
# plot prediction correctness
# filter to devide between fixation and all (for now)
p_stim_prediction_source = ColumnDataSource({'xs': [], 'ys': [], 'line_color': [], 'type_name': []})
p_stim_prediction_view = CDSView(source=p_stim_prediction_source, filters=[GroupFilter(column_name='line_type',
                                                                                       group='fixation')])
p_stim_evaluation = p_stim.multi_line(xs='xs', ys='ys', line_color='line_color', line_width=3, visible=False,
                                      source=p_stim_prediction_source, view=p_stim_prediction_view)

# text
p_stim_fixation_count = []
for idd in range(user_parallel_maximum):
    p_stim_fixation_count.append(p_stim.text(x='cog_x', y='cog_y', text="count",
                                             text_font_size="text_size", text_color='text_color',
                                             source=p_stim_fixation_marker_source[idd],
                                             text_align='center', text_baseline="middle",
                                             visible=False))
    p_stim_fixation_count[idd].selection_glyph = p_stim_fixation_count[idd].glyph
    p_stim_fixation_count[idd].nonselection_glyph = p_stim_fixation_count[idd].glyph

# Feature table #
#################
table_columns = [TableColumn(field="feat_names", title="Features")]
table_source = ColumnDataSource()
#: Feature Table
feature_table = DataTable(source=table_source, columns=table_columns, width=width_two, height=width_three)

# control visibility of feature table
feature_table_toggle = Toggle(label='Feature Table Visible', active=True, width=width_half)


def feature_table_toggle_cb(_event, _old, new):
    feature_table.visible = new
    remark_table.visible = new  # title defined at the bottom
    update_table_toggle.visible = new
    if new:
        headline_table.width = width_one
    else:
        headline_table.width = width_one + width_half
        update_table_toggle.active = False  # turn of updating
    update_feature_table_description()


feature_table_toggle.on_change('active', feature_table_toggle_cb)




# velocity plot #
#################
tooltips = [("type", "@user_name @type @id"),
            ("range", "(@start, @end)"),
            ("(time,velocity)", "($x, $y)")]

ivt_threshold_source = ColumnDataSource({'scope': [], 'threshold': [], 'user_name': [], 'type': [], 'id': []})
name_source = ColumnDataSource({'user_name': [], 'user_color': [], 'bar_y': []})
bar_source_keys = ['user_name', 'user_color', 'bar_y', 'bar_height', 'start', 'end', 'color', 'type', 'id',
                   'color_evaluation']
bar_source = ColumnDataSource(dict.fromkeys(bar_source_keys, [None]))

#: Velocity Plot
p_vel = figure(y_range=(0, 1000),
               tools="hover,wheel_zoom,pan,save,box_zoom,help,reset",  # active_scroll="wheel_zoom",
               width=width_full, height=400, margin=layout_margin // 2, tooltips=tooltips)
p_vel.xaxis.axis_label = 'Time [#sample]'
p_vel.yaxis.axis_label = 'Velocity [deg/s]'
p_vel.outline_line_color = None
p_vel.toolbar.autohide = True

# add eye movement classification bars
p_vel.hbar(y="bar_y", height="bar_height", fill_alpha=0.5,
           left='start', right='end', fill_color="color",
           source=bar_source)
# add evaluation result bars
p_vel_evaluation = p_vel.hbar(y="bar_y", height=10,
                              left='start', right='end', fill_color="color_evaluation",
                              source=bar_source, visible=False)
# add names
p_vel.text(x=-10, y='bar_y', text='user_name', text_color='user_color',
           source=name_source, text_align='right', text_baseline="middle")

# add ivt threshold
p_vel.line(x='scope', y='threshold', source=ivt_threshold_source,
           line_color='white', line_dash='dashed', line_width=2)

p_vel_velocities = p_vel.multi_line(xs=[[0]], ys=[[0]], line_color=get_colors(1), line_width=1)

p_vel_info = Div(width=width_one)


def update_bar_source():

    user_names = []
    user_color = []
    bar_y = []
    segments = []
    for user_id, (sample_starts, type_names, prediction_results) in enumerate(eye_movements4trajectories):
        len_samples = len(sample_starts) - 1

        color_value = []
        indices = []
        index_fix = 0
        index_sac = 0

        if predictor_toggle.active:
            color_pcv = get_pcv_color(prediction_results)
        else:
            color_pcv = ['#ffffff'] * len(prediction_results)

        for typ in type_names:
            if typ == 'saccade':
                color_value.append("#FFFFFF")
                indices.append(index_sac)
                index_sac += 1
            elif typ == 'fixation':
                color_value.append("#718dbf")
                indices.append(index_fix)
                index_fix += 1
            else:
                color_value.append("#808080")
                indices.append(None)

        user_names.append(get_user_name(user_id))
        user_color.append(get_user_color(user_id))
        bar_y.append((user_id + 0.5) * ivt_threshold)

        # order has to be like bar_source_keys
        segment = list(zip([get_user_name(user_id)] * len_samples,              # user_name
                           [get_user_color(user_id)] * len_samples,             # user_color
                           [(user_id + 0.5) * ivt_threshold] * len_samples,     # bar_y
                           [ivt_threshold] * len_samples,                       # bar_height
                           sample_starts[:-1],                                  # start
                           sample_starts[1:],                                   # end
                           color_value,                                         # color
                           type_names,                                          # type
                           indices,                                             # id
                           color_pcv,                                           # quality of prediction
                           ))
        segment.sort(key=lambda x: x[0])
        segments += segment

    new_data = {}

    for ii in range(len(bar_source_keys)):
        new_data[bar_source_keys[ii]] = [t[ii] for t in segments]

    bar_source.data = new_data
    name_source.data = {'user_name': ['IVT - - -'] + user_names,
                        'user_color': ['white'] + user_color,
                        'bar_y': [ivt_thresh_slider.value] + bar_y}
    ivt_threshold_source.data = {'scope': [0, time_slider.end],
                                 'threshold': [ivt_thresh_slider.value, ivt_thresh_slider.value],
                                 'user_name': ['IVT']*2,            # only for tooltip
                                 'type': ["velocity threshold"]*2,  # only for tooltip
                                 'id': [""]*2}                      # only for tooltip


# Velocity Time Bar
update_velocity_toggle = Toggle(label="Update Time Bar", active=True, width=width_half)
def update_velocity_toggle_cb(_attrname, _old, _new):
    update_time_bar_source()
update_velocity_toggle.on_change('active', update_velocity_toggle_cb)

# white box for time
time_bar_source = ColumnDataSource()
def update_time_bar_source():
    if not update_velocity_toggle.active:
        return
    time_bar_source.data = {'start': [time_slider.value[0]],
                            'end': [time_slider.value[1]],
                            'user_name': ['Time'],  # only for tooltip
                            'type': ["range"],      # only for tooltip
                            'id': [""]}             # only for tooltip
p_vel_time = p_vel.hbar(y=500, height=1000, left='start', right='end', source=time_bar_source,
                        fill_color="#ffffff", fill_alpha=0.1)

def velocity_tap(event):
    bar_height = ivt_threshold
    # Select user based on click position in y. See update_velocity_plot.
    user_idx = int(event.y // bar_height)
    if user_idx < 0 or user_idx >= len(eye_movements4trajectories):
        if len(eye_movements4trajectories) == 1:
            user_idx = max(min(user_idx, len(eye_movements4trajectories) - 1), 0)
        else:
            print("unknown user clicked: ", user_idx)
            return
    sequence = None
    # linear search all eye movements
    for t_id in range(len((eye_movements4trajectories[user_idx][0])) - 1):
        if eye_movements4trajectories[user_idx][0][t_id] <= event.x < eye_movements4trajectories[user_idx][0][t_id + 1]:
            sequence = (eye_movements4trajectories[user_idx][0][t_id],
                        eye_movements4trajectories[user_idx][0][t_id + 1] - 1)
            break
    if sequence:
        time_slider.value = sequence
        # time_slider_cb('value', time_slider.value, time_slider.value)
p_vel.on_event(Tap, velocity_tap)


####################
# Update functions #
####################

def animate_update():
    # cycle the time time_slider
    if hold_left_time_toggle.active:
        # move only right range
        if time_slider.value[1] + animation_step_slider.value < time_slider.end:
            # normal step
            time_slider.value = (time_slider.value[0],
                                 time_slider.value[1] + animation_step_slider.value)
        else:
            # repeat or do final step (and break)
            if time_slider.value[1] == time_slider.end:
                # repeat
                time_slider.value = (time_slider.value[0],
                                     time_slider.value[0] + 1)
            else:
                # final step
                time_slider.value = (time_slider.value[0],
                                     time_slider.end)
                if stop_right_time_toggle.active:
                    # break
                    play_button.active = False

    else:
        # move left and right range together
        if time_slider.value[1] + animation_step_slider.value < time_slider.end:
            # normal step
            time_slider.value = (time_slider.value[0] + animation_step_slider.value,
                                 time_slider.value[1] + animation_step_slider.value)
        else:
            # repeat or do final step (and break)
            if time_slider.value[1] == time_slider.end:
                # repeat
                if time_slider.value[0] == 0:
                    # No Movement possible, assume moving only the right side is wanted
                    hold_left_time_toggle.active = True
                else:
                    time_slider.value = (0,
                                         time_slider.value[1] - time_slider.value[0])
            else:
                # final step
                time_slider.value = (time_slider.value[0] + time_slider.end - time_slider.value[1],
                                     time_slider.end)
                if stop_right_time_toggle.active:
                    # break
                    play_button.active = False

def update_time():
    # BEST PRACTICE --- update .data in one step with a new dict
    global trajectories_active

    # x,y for all selected users
    #   sometimes the slider is slightly of integer (e.g. .00000000000006)
    trajectory_slice = slice(int(time_slider.value[0]), int(time_slider.value[1]))
    trajectories_active = [xys[trajectory_slice, :]
                           for xys in trajectories.get_trajectories('pixel_image')]
    subset_plot = [crop_trajectory(xy, t_id) for (t_id, xy) in enumerate(trajectories_active)]

    new_plot_data = {'xs': [s[:, 0] for s in subset_plot],
                     'ys': [s[:, 1] for s in subset_plot],
                     'line_color': get_colors(len(trajectories_active))}

    # only for first user
    new_prediction_data = generate_prediction_plot_data(trajectory=trajectories[0],
                                                        evaluated_eye_movements=eye_movements4trajectories[0],
                                                        trajectory_slice=trajectory_slice)

    # send data to renderer
    p_stim_trajectories.data_source.data = new_plot_data
    update_trajectory_marker_source()
    p_stim_evaluation.data_source.data = new_prediction_data
    update_marker_source(subset_plot)
    update_fixation_marker_source()
    update_time_bar_source()
    update_feature_table_data()

def time_slider_full():
    time_slider.value = (0, time_slider.end)

def time_slider_fix():
    """ Correct Slider Values while resetting"""
    if time_slider.value[1] > time_slider.end:
        time_slider.value = (max(0, time_slider.end - (time_slider.value[1] - time_slider.value[0])),
                             time_slider.end)

def initialize_plotdata():
    """update trajectories and dataset"""
    # we update the global plot data here
    global trajectories, removed_nans, interpolated_values
    # delete old data
    trajectories = Trajectories()
    usernames = user_select.value
    removed_nans = [0 for _ in usernames]
    interpolated_values = [0 for _ in usernames]
    # add all selected users
    max_trajectory_length = 0
    for (u_id, username) in enumerate(usernames):
        trajectory = load_data(username, stimuli_select.value)
        update_data_interpolate_toggle(sum(trajectory.interpolated), u_id)
        trajectories.append(trajectory)
        max_trajectory_length = max(max_trajectory_length, len(trajectory))
    time_slider.end = max_trajectory_length
    time_slider_fix()

    p_vel_velocities.data_source.data = {'xs': trajectories.sample_ids, 'ys': trajectories.get_velocities('angle_deg'),
                                         'line_color': get_colors(len(trajectories)),
                                         'user_name': trajectories.users,   # only for tooltip
                                         'type': trajectories.cases,        # only for tooltip
                                         'id': trajectories.genders}        # only for tooltip

    update_classification()  # update the saccades and fixations
    update_time()

    return trajectories.velocities

marker_minsize = 5

# noinspection PyTypeChecker
#   [str] * int
def update_fixation_marker_source(active: bool = False) -> None:
    """ Update marks showing all points of interest (fixation)

        get_trajectory_info provides everything for p_stim_marker_source:
            count, cog_x, cog_y, size and duration
        This function extends the information for fixation marker and saccade lines with:
            duration_scaled, marker_alpha/color, line_color, text_size/color

    :param active: force update (otherwise only done, when one of components is visible.
    """
    # Note: It is not possible to have element wise marker_colors / line_colors

    global p_stim_fixation_marker_source

    # todo: [optional] show only fixations in Time-Range

    # update fixation only if necessary
    if not active \
            and not marker_fixation_circle_toggle.active \
            and not marker_fixation_count_toggle.active \
            and not marker_saccades_toggle.active:
        return

    # check active user (up to limit)
    for u_id in range(user_parallel_maximum):

        if u_id >= len(user_select.value):
            p_stim_fixation_marker_source[u_id].data = p_stim_fixation_marker_source_single
            continue

        start_samples, type_names, prediction_results = eye_movements4trajectories[u_id]

        index_fixation = find([typ == 'fixation' for typ in type_names])

        data = get_trajectory_info(
            [trajectories[u_id].get_trajectory('pixel_image')[
             range(start_samples[index_fixation[i_fix]],
                   start_samples[index_fixation[i_fix] + 1]), :]
             for i_fix in range(len(index_fixation))])

        # scale duration to show duration by marker line width
        # noinspection PyArgumentList
        #   doesn't understand ...
        data["marker_alpha"] = data["duration"] / data["duration"].max() * marker_duration_slider_alpha.value

        # colors
        data["text_color"] = [get_user_color(u_id)] * len(index_fixation)

        # calculate text_size
        if marker_fixation_text_toggle.active:
            # noinspection PyTypeChecker
            data["text_size"] = ["{}pt".format(element * marker_duration_slider_text.value)
                                 for element in data["duration"]]
        else:
            # noinspection PyTypeChecker
            data["text_size"] = ["{}pt".format(50 * marker_duration_slider_text.value)] * len(data["duration"])

        p_stim_fixation_marker_source[u_id].data = data

    update_fixation_marker_color()

# marker trajectory last element
marker_trajectory_last_toggle = Toggle(label="Mark only Last Element", active=True, width=width_half)
def marker_last_toggle_cb(_attrname, _old, _new):
    update_marker_source()
marker_trajectory_last_toggle.on_change('active', marker_last_toggle_cb)

# marker trajectory
marker_trajectory_toggle = Toggle(label="Mark Trajectory ([0,0] if NaN)", active=False, width=width_half)
def marker_toggle_cb(_attrname, _old, new):
    p_stim_position.visible = new
    update_marker_source()
marker_trajectory_toggle.on_change('active', marker_toggle_cb)

def set_marker_size(marker_size):
    p_stim_marker_source.data['size'] = marker_size

# ping marker
marker_trajectory_ping_button = Button(label="Ping Marker", width=width_half)
def marker_trajectory_ping_button_cb(_event):
    if not p_stim_position.visible or len(p_stim_marker_source.data['size']) < 1:
        return
    marker_size = p_stim_marker_source.data['size']
    iterations = 10
    frame_rate = 25
    for it in range(1, iterations + 1):
        run_delayed(frame_rate * it, 'scaling' + str(it), set_marker_size,
                    [int(np.ceil(elem * 1.5 ** it)) for elem in marker_size])
    run_delayed(frame_rate * (iterations + 2), 'scalingBack', set_marker_size, marker_size)
marker_trajectory_ping_button.on_event(ButtonClick, marker_trajectory_ping_button_cb)

p_stim_position = p_stim.circle('cog_x', 'cog_y', radius='size', source=p_stim_marker_source,
                                fill_alpha=0.1, line_color='red', visible=marker_trajectory_toggle.active)

def update_marker_source(subset=None, active=False):
    # update marker
    if not marker_trajectory_toggle.active and not active:
        return

    if not subset:
        subset = trajectories.get_trajectories('pixel_image')

    if marker_trajectory_last_toggle.active:
        # show only last position
        cog_x = [0 if np.isnan(s[-1, 0]) else s[-1, 0] for s in subset]
        cog_y = [0 if np.isnan(s[-1, 1]) else s[-1, 1] for s in subset]

        p_stim_marker_source.data = {'cog_x': cog_x,
                                     'cog_y': cog_y,
                                     'size': [marker_minsize] * len(subset)}

    else:
        p_stim_marker_source.data = get_trajectory_info(subset)

def update_fixation_marker_color():

    for s_id in range(min(user_parallel_maximum, len(user_select.value))):
        if marker_fixation_number_color_toggle.active:
            p_stim_fixation_count[s_id].glyph.text_color = marker_fixation_number_color_picker.color
        else:
            p_stim_fixation_count[s_id].glyph.text_color = get_user_color(s_id)
        if marker_fixation_marker_color_toggle.active:
            p_stim_fixation_mark[s_id].glyph.line_color = marker_fixation_marker_color_picker.color
            p_stim_fixation_mark[s_id].glyph.fill_color = marker_fixation_marker_color_picker.color
            p_stim_saccades_mark[s_id].glyph.line_color = marker_fixation_marker_color_picker.color
        else:
            p_stim_fixation_mark[s_id].glyph.line_color = get_user_color(s_id)
            p_stim_fixation_mark[s_id].glyph.fill_color = get_user_color(s_id)
            p_stim_saccades_mark[s_id].glyph.line_color = get_user_color(s_id)

def marker_fixation_circle_color_picker_cb(_attrname, _old, _new):
    change_fixation_marker()

marker_fixation_number_color_picker = ColorPicker(title="Fixation Number:", color="blue",
                                                  width=width_half)
marker_fixation_number_color_toggle = Toggle(label="Use this color", active=True,
                                             width=width_half)
marker_fixation_marker_color_picker = ColorPicker(title="Fixation Marker:", color="fuchsia",
                                                  width=width_half)
marker_fixation_marker_color_toggle = Toggle(label="Use this color", active=False,
                                             width=width_half)

marker_fixation_number_color_picker.on_change("color", marker_fixation_circle_color_picker_cb)
marker_fixation_marker_color_picker.on_change("color", marker_fixation_circle_color_picker_cb)
marker_fixation_number_color_toggle.on_change("active", marker_fixation_circle_color_picker_cb)
marker_fixation_marker_color_toggle.on_change("active", marker_fixation_circle_color_picker_cb)

# mark fixations
def fixation_toggle_cb(_attrname, _old, _new):
    update_fixation_marker()
marker_fixation_circle_toggle = Toggle(label="Mark Fixation", active=False, width=width_half)
marker_fixation_circle_toggle.on_change('active', fixation_toggle_cb)
marker_fixation_text_toggle = Toggle(label="Count Size: Duration", active=True, width=width_half)
marker_fixation_text_toggle.on_change('active', fixation_toggle_cb)
marker_fixation_count_toggle = Toggle(label="Count Fixation", active=False, width=width_half)
marker_fixation_count_toggle.on_change('active', fixation_toggle_cb)
marker_saccades_toggle = Toggle(label="Mark Saccades", active=False, width=width_half)
marker_saccades_toggle.on_change('active', fixation_toggle_cb)

# color for trajectory marker

def update_trajectory_marker_source():
    """ Update marker color """

    if not p_stim_trajectory_marker_toggle.active:
        return

    new_plot_data = p_stim_trajectories.data_source.data

    if trajectory_marker_color_toggle.active:
        color = trajectory_marker_color_picker.color
    else:
        color = new_plot_data['line_color'][0]

    p_stim_trajectory_marker.data_source.data = {
        'x': new_plot_data['xs'][0],
        'y': new_plot_data['ys'][0],
        'line_color': [color] * len(new_plot_data['xs'][0])}

def trajectory_marker_cb(_attrname, _old, _new):
    update_trajectory_marker_source()

trajectory_marker_color_picker = ColorPicker(title="Trajectory Marker:", color="black",
                                             width=width_half)
trajectory_marker_color_toggle = Toggle(label="Use this color", active=True,
                                        width=width_half)
trajectory_marker_color_picker.on_change("color", trajectory_marker_cb)
trajectory_marker_color_toggle.on_change("active", trajectory_marker_cb)

#### Stimulus tap ####

def tap_cb(_event):

    # find which is tapped
    index = None
    user_id = None

    for (idu, source) in enumerate(p_stim_fixation_marker_source):
        if len(source.selected.indices) > 0:
            index = source.selected.indices[0]
            user_id = idu

    if index is None:
        time_slider_full()
        return

    bool_user = [names == get_user_name(user_id) for names in bar_source.data['user_name']]
    bool_type = [types == 'fixation' for types in bar_source.data['type']]  # todo: how to find saccades?
    bool_index = [ids == index for ids in bar_source.data['id']]

    boolean_mask = bool_user and bool_type and bool_index
    show_id = boolean_mask.index(True)
    time_slider.value = bar_source.data['start'][show_id], bar_source.data['end'][show_id]

p_stim.on_event(Tap, tap_cb)


# Slider for Duration
marker_duration_slider_text = Slider(title="Duration*Size Factor", start=0.05, end=1, step=0.05, value=0.25,
                                     width=width_one)
def marker_duration_slider_cb(_attrname, _old, _new):
    update_fixation_marker_source()
marker_duration_slider_text.on_change('value', marker_duration_slider_cb)

marker_duration_slider_alpha = Slider(title="Duration*Alpha Factor", start=0.05, end=2, step=0.05, value=1,
                                      width=width_one)
marker_duration_slider_alpha.on_change('value', marker_duration_slider_cb)

def change_fixation_marker():  # todo: can the calls update/change be separated? No, then delete this
    update_fixation_marker_source()

def update_fixation_marker():

    update_fixation_marker_source()

    for u_id in range(user_parallel_maximum):
        p_stim_saccades_mark[u_id].visible = marker_saccades_toggle.active
        p_stim_fixation_mark[u_id].visible = marker_fixation_circle_toggle.active
        p_stim_fixation_count[u_id].visible = marker_fixation_count_toggle.active

    if marker_fixation_text_toggle.active:
        marker_fixation_text_toggle.label = "Count Size: Duration"
    else:
        marker_fixation_text_toggle.label = "Count Size: 50pt"


# Update Table
update_table_toggle = Toggle(label="Update Feature Table", active=False, width=width_half)


def update_table_toggle_cb(_attrname, _old, _new):
    update_feature_table_description()
    update_feature_table_data()


update_table_toggle.on_change('active', update_table_toggle_cb)


def update_feature_table_description():
    if update_table_toggle.active:
        headline_table.text = '<h2>Feature Table</h2>'
    elif feature_table.visible:
        headline_table.text = '<h2>Feature Table <font color="#e84d60">(No automatic update)</font></h2>'
    else:
        headline_table.text = '<h2>Feature Table (Hidden)</h2>'


def update_feature_table_data():
    # calculate feature and update table

    if not update_table_toggle.active:
        return

    # calculate the features
    features = []
    for xy in trajectories_active:
        if xy.shape[0] < 2:
            break
        trajectory = Trajectory(xy, kind='pixel_image', sample_rate=dataset.sample_rate,
                                **Trajectory.screen_params_converter(dataset.get_screen_params()))
        features.append(all_features(trajectory, omit_our=False))

    if len(features) < 1:
        return

    feat = {str(i): f.values[0] for i, f in enumerate(features)}
    feat["feat_names"] = features[0].columns

    # update the table
    feature_table.source.data = feat
    feature_table.columns = [
                                TableColumn(field="feat_names", title="Features")
                            ] + [
                                TableColumn(field=str(i), title=get_user_name(i),
                                            formatter=NumberFormatter(format='0,0.000', text_align="right"))
                                for i in range(len(trajectories_active))
                            ]
    # todo: show color of users.
    # text_color=get_user_color(i) could be used to color feature values but it is to bright so read anything.
    # We could use a cell Formatter to color cells, or maybe HTML Formatter to color header,
    #   but only one Formatter can be used!


##########################
# other visible elements #
##########################

# Datasets #
############
dataset_names = list(datasets.keys())
dataset_select = Select(title="Dataset", value=None, options=dataset_names, width=width_one)
update_in_progress = False
def dataset_select_cb(_attrname, _old, new):
    # we update the global plot data here
    global dataset, cases, users, update_in_progress

    dataset = datasets[new]
    print('selected: ', new)
    sample_rate_button.label = "Tracking Rate: " + str(dataset.sample_rate)

    cases = dataset.get_cases()
    cases.sort()
    if not len(cases):
        raise Exception("Dataset doesn't return cases! (get_cases)")

    update_in_progress = True  # prevent early user update
    stimuli_select.options = cases
    stimuli_select.value = cases[0]
    update_in_progress = False

    users = list(dataset.get_users())
    if not len(cases):
        raise Exception("Dataset doesn't return users! (get_users")
    user_select.options = sorted(users)
    user_select.value = [users[0]]
    training_user_limit_slider.end = len(users)

    time_slider_full()
    predictor_parameter_changed()
dataset_select.on_change('value', dataset_select_cb)


# Stimuli #
###########
stimuli_select = Select(title="Case / Stimulus", value=None, options=cases, width=width_one)
def stimuli_select_cb(_attrname, _old, new):
    # we update the global plot data here
    global stimulus, user_alignment, cog_fix_all, cog_sac_all, cog_fix_eval, cog_sac_eval

    stimulus = load_img_bokeh(dataset.get_stimulus(new))

    update_image_data()
    p_stim_background.glyph.dh, p_stim_background.glyph.dw = stimulus.shape

    user_alignment = dataset.load_alignment(stimuli_select.value)

    # reset heatmap data
    cog_fix_all = None
    cog_sac_all = None
    cog_fix_eval = None
    cog_sac_eval = None

    if not update_in_progress:
        user_select_cb('value', [], user_select.value)
stimuli_select.on_change('value', stimuli_select_cb)


# Users #
#########
user_select = MultiSelect(title="Agents / Users", value=[], options=users, width=width_one, height=110)
# noinspection SpellCheckingInspection
def user_select_cb(_attrname, _old, new):
    if alignment_auto_toggle.active:
        get_alignment()
    predictor_user_select.value = new
    predictor_user_button_cb(ButtonClick)
    initialize_plotdata()  # update trajectories and velocities
user_select.on_change('value', user_select_cb)

# Mark NaN #
############
mark_nan_toggle = Toggle(label="List NaN in Classification", active=False, width=width_one)
def mark_nan_toggle_cb(_attrname, _old, _new):
    update_classification()
mark_nan_toggle.on_change('active', mark_nan_toggle_cb)

# Interpolate NaN #
###################
data_interpolate_toggle = Toggle(label="Interpolate NaN while loading", active=False, width=width_one)
def data_interpolate_toggle_cb(_attrname, _old, _new):
    # update the actual plot
    initialize_plotdata()
    if not data_interpolate_toggle.active:
        data_interpolate_toggle.label = "Interpolate NaN while loading"
data_interpolate_toggle.on_change('active', data_interpolate_toggle_cb)
interpolated_values = []
def update_data_interpolate_toggle(number_nan, v_id):
    global interpolated_values
    interpolated_values[v_id] = number_nan
    data_interpolate_toggle.label = "Interpolate {} NaN while loading".format(interpolated_values)

# Clip to image #
#################
clip_toggle = Toggle(label="Crop trajectory to image (only visually)", active=False, width=width_one)
def clip_toggle_cb(_attrname, _old, _new):
    # update the actual plot
    update_time()
    if not clip_toggle.active:
        clip_toggle.label = "Crop trajectory to image (only visually)"
clip_toggle.on_change('active', clip_toggle_cb)
removed_nans = []
def update_clip_toggle(removed_nan, s_idd):
    global removed_nans
    removed_nans[s_idd] = removed_nan
    clip_toggle.label = "Cropped {} samples out of image (only visually)".format(removed_nans)

# center data #
###############
data_center_toggle = Toggle(label="Center data while loading", active=False, width=width_one)
def center_data_toggle_cb(_attrname, _old, _new):
    initialize_plotdata()
data_center_toggle.on_change('active', center_data_toggle_cb)

# Remove duplicates #
#####################
duplicate_toggle = Toggle(label="Remove duplicates from trajectory", active=False, width=width_one)
def duplicate_toggle_cb(_attrname, _old, _new):
    # update the actual plot
    run_delayed(delay_time_ms, "plot_data", initialize_plotdata)
    if not duplicate_toggle.active:
        duplicate_toggle.label = "Remove duplicates from trajectory"
duplicate_toggle.on_change('active', duplicate_toggle_cb)
duplicate_slider = Slider(title="Ignore repetitions", start=0, end=100, step=1, value=remove_duplicates_threshold,
                          width=width_one)
duplicate_slider.on_change('value', duplicate_toggle_cb)
def update_duplicate_toggle(removed_stats):
    # removed_stats: number of duplicates total, number of sets of duplicates
    duplicate_toggle.label = "Removed {} duplicates in {} sets from trajectory".format(
        removed_stats[0], removed_stats[1])

# Trajectory visible #
######################
trajectory_visible_toggle = Toggle(label="Show Trajectory", active=True, width=width_half)
def trajectory_visible_toggle_cb(_attrname, _old, new):
    p_stim_trajectories.visible = new
trajectory_visible_toggle.on_change('active', trajectory_visible_toggle_cb)

p_stim_trajectory_marker_toggle = Toggle(label="Show Nodes [U1]", active=False, width=width_half)
def p_stim_trajectory_marker_toggle_cb(_attrname, _old, new):
    p_stim_trajectory_marker.visible = new
    if new:
        update_time()
p_stim_trajectory_marker_toggle.on_change('active', p_stim_trajectory_marker_toggle_cb)

trajectory_width_slider = Slider(start=1, end=10, step=1, value=2, title="Width Trajectory", width=width_one)
def trajectory_width_slider_cb(_attrname, _old, new):
    p_stim_trajectories.glyph.line_width = new
    p_stim_trajectory_marker.glyph.line_width = new * 2
    for u_id in range(user_parallel_maximum):
        p_stim_fixation_mark[u_id].glyph.line_width = new
        p_stim_saccades_mark[u_id].glyph.line_width = new
trajectory_width_slider.on_change('value', trajectory_width_slider_cb)

# Objects without Callback #
############################
hold_left_time_toggle = Toggle(label="Hold left range", active=False, width=width_half)
stop_right_time_toggle = Toggle(label="Stop at the end", active=False, width=width_half)
sample_rate_button = Button(label="Tracking Rate", width=width_half)

def get_animation_speed():
    return animation_step_slider.value / (animation_interval_slider_x1000.value / 1000) / dataset.sample_rate

# animation slider #
####################
animation_step_slider = Slider(title="Step size [frames]", start=1, end=100, step=1, value=1,
                               width=width_one)
def animation_step_slider_cb(_attrname, _old, new):
    # print("Step:"+str(new))
    if new >= 1 and not np.isclose(10 ** animation_speed_slider_log.value,
                                   new / (animation_interval_slider_x1000.value / 1000) / dataset.sample_rate):
        # print(" Speed is:" + str(10**animation_speed_slider_log.value))
        # print(" Speed should:" + str(new / (animation_interval_slider_x1000.value / 1000) / dataset.sample_rate))
        # update speed to correct value
        animation_speed_slider_log.value = np.log10(
            new / (animation_interval_slider_x1000.value / 1000) / dataset.sample_rate)
    elif new < 1:
        raise Exception('frame can not be below 1')

animation_step_slider.on_change('value', animation_step_slider_cb)

animation_interval_slider_x1000 = Slider(title="Update Interval [ms]", start=10, end=500, step=10, value=100,
                                         width=width_one)
def animation_interval_slider_cb(_attrname, _old, new):
    # print("Interval:" + str(new))
    if not np.isclose(10 ** animation_speed_slider_log.value,
                      animation_step_slider.value / (new / 1000) / dataset.sample_rate):
        # print(" Speed is:" + str(10 ** new))
        # print(" Speed should:" + str(get_animation_speed()))
        new_step_value = int(dataset.sample_rate * 10 ** animation_speed_slider_log.value * new / 1000)
        if new_step_value >= 1:
            animation_step_slider.value = new_step_value
        else:
            animation_speed_slider_log.value = np.log10(
                animation_step_slider.value / (new / 1000) / dataset.sample_rate)
animation_interval_slider_x1000.on_change('value', animation_interval_slider_cb)

# animation speed slider #
##########################
animation_speed_slider_log = Slider(title="Speed ratio (logarithmic slider)", start=-2, end=0, step=0.01,
                                    value=np.log10(get_animation_speed()), width=width_one)
def animation_speed_slider_cb(_attrname, _old, new):
    # print("Speed:" + str(10**new))
    animation_speed_slider_log.title = "Speed ratio {0:0.2f} (log slider)".format(10 ** new)
    if not np.isclose(10 ** new, get_animation_speed()):
        # print(" Speed is:" + str(10**new))
        # print(" Speed should:" + str(get_animation_speed()))
        new_step_value = int(dataset.sample_rate * 10 ** new * animation_interval_slider_x1000.value / 1000)
        if new_step_value >= 1:
            animation_step_slider.value = new_step_value
        else:
            animation_interval_slider_x1000.value = animation_step_slider.value * 1000 / (
                    dataset.sample_rate * (10 ** animation_speed_slider_log.value))
animation_speed_slider_log.on_change('value', animation_speed_slider_cb)

# Changing Data #
##################

next_dataset_button = Button(label="Next Dataset", width=width_half)
def next_dataset_button_cb():
    dataset_select.value = get_next_in_list(dataset_select.options, dataset_select.value)
next_dataset_button.on_event(ButtonClick, next_dataset_button_cb)
next_stimulus_button = Button(label="Next Stimulus", width=width_half)
def next_stimulus_button_cb():
    stimuli_select.value = get_next_in_list(stimuli_select.options, stimuli_select.value)
next_stimulus_button.on_event(ButtonClick, next_stimulus_button_cb)
next_user_button = Button(label="Next User", width=width_half)
def next_user_button_cb():
    user_select.value = [get_next_in_list(user_select.options, user_select.value[0])]
next_user_button.on_event(ButtonClick, next_user_button_cb)

# Image Color #
###############

def image_color_toggle_cb(_attrname, _old, _new):
    run_delayed(delay_time_ms, 'image', update_image_data)

def update_image_data():
    p_stim_background.data_source.data = {'image': [
        fade_white(scale_color(stimulus.copy(), image_color_slider.value), image_fade_slider.value)]}


image_color_slider = Slider(start=0, end=3, step=0.05, value=1, title="Color of Image",
                            width=width_one)
image_color_slider.on_change('value', image_color_toggle_cb)

image_fade_slider = Slider(start=0, end=1, step=0.05, value=1, title="Image Fade out",
                           width=width_one)
image_fade_slider.on_change('value', image_color_toggle_cb)


# data scaling slider #
###################
data_scaling_slider = Slider(start=0.1, end=10, step=0.01, value=1, title="data-scaling (squared)",
                             width=width_two)


def data_scaling_slider_cb(_attrname, _old, new):
    run_delayed(delay_time_ms, "plot_data", initialize_plotdata)
    # updating title with used number (squared value).
    data_scaling_slider.title = "{:.3} data-scaling (squared)".format(np.square(new))


data_scaling_slider.on_change('value', data_scaling_slider_cb)

# x/y-offset slider #
###################
x_offset_slider = Slider(start=-500, end=500, step=1, value=0, title="x-offset", width=width_two)
def x_offset_slider_cb(_attrname, _old, new):
    p_stim_background.glyph.x = new
x_offset_slider.on_change('value', x_offset_slider_cb)

y_offset_slider = Slider(start=-500, end=500, step=1, value=0, title="y-offset", width=width_two)
def y_offset_slider_cb(_attrname, _old, new):
    p_stim_background.glyph.y = new
y_offset_slider.on_change('value', y_offset_slider_cb)

# saving / loading Offsets #
############################

# see stimuli_select_cb for loading of user_alignment
user_alignment = pd.DataFrame()

alignment_auto_toggle = Toggle(label='Load Alignment, when user changes', active=True, width=width_one)
alignment_load_button = Button(label='Load Alignment (first user)', width=width_half)
def alignment_load_button_cb(_event):
    get_alignment()
alignment_load_button.on_event(ButtonClick, alignment_load_button_cb)
alignment_save_button = Button(label='Save Alignment', width=width_half)


# button to reset the xy sliders to 0 and the data_scaling slider to 1
reset_values_button = Button(label='Reset Offset and Scaling', width=width_one)


def reset_values_button_cb(_event):

    x_offset_slider.value = 0
    y_offset_slider.value = 0
    data_scaling_slider.value = 1


reset_values_button.on_event(ButtonClick, reset_values_button_cb)


def alignment_save_button_cb(_event):
    save_alignment()
alignment_save_button.on_event(ButtonClick, alignment_save_button_cb)

def get_alignment():
    # todo: alignment is fixt not in data, instead image is moved.
    #   We have to modify the data somehow...
    #   A good Idea could be, to play with this alignment and make a function to transfer this to a deeper dataset.
    #   Other dataset should affect the trajectory while loading.
    #   There also can be a dataset offset (mean of all users) so all individual offsets are smaller.
    if user_select.value[0] in user_alignment.index:
        x_offset_slider.value = user_alignment.loc[user_select.value[0], 'x']
        y_offset_slider.value = user_alignment.loc[user_select.value[0], 'y']

def save_alignment():
    global user_alignment

    for user in user_select.value:
        user_alignment.loc[user] = [x_offset_slider.value, y_offset_slider.value]

    dataset.save_alignment(stimuli_select.value, user_alignment)
    print('Alignment [{},{}] saved for'.format(x_offset_slider.value, y_offset_slider.value), user_select.value)


# Smoothing #
#############
smooth_toggle = Toggle(label="Smooth Values", active=True, width=width_one)
savgol_filter_frame_slider = Slider(start=1, end=55, step=2, value=savgol_frame_size, title="Savgol frame_size",
                                    width=width_one)
savgol_filter_poly_slider = Slider(start=1, end=9, step=1, value=savgol_poly_order, title="Savgol pol_order",
                                   width=width_one)  # 0=1, and 2=3 for savgol_poly !?
def smooth_toggle_cb(_attrname, _old, _new):
    savgol_filter_poly_slider.end = savgol_filter_frame_slider.value - 1
    savgol_filter_poly_slider.value = min(savgol_filter_poly_slider.value, savgol_filter_poly_slider.end)
    predictor_parameter_changed()
    run_delayed(delay_time_ms, "plot_data", initialize_plotdata)
smooth_toggle.on_change('active', smooth_toggle_cb)
savgol_filter_frame_slider.on_change('value', smooth_toggle_cb)
savgol_filter_poly_slider.on_change('value', smooth_toggle_cb)


# Time slider #
###############
time_slider = RangeSlider(start=0, end=1, step=1, value=(0, 1), title="Time range", width=width_three)
# noinspection PyUnusedLocal,PyUnusedLocal
def time_slider_cb(_attrname, _old, new):
    # make shure range is not zero
    if new[1] - new[0] < 1:
        if _old[0] - new[0]:
            # first element changed
            time_slider.value = (new[0] - 1, new[1])
        else:
            # last element changed
            time_slider.value = (new[0], new[1] + 1)
        return

    run_delayed(delay_time_ms, "time", update_time)
time_slider.on_change('value', time_slider_cb)

time_slider_reset_button = Button(label="Reset Time", width=width_half)
def time_slider_reset_button_cb(_event):
    time_slider_full()
time_slider_reset_button.on_event(ButtonClick, time_slider_reset_button_cb)


# Play #
########
callback_id = None
play_button = Toggle(label="â–º Play", button_type="success", active=False, width=width_half)
def animate():
    global callback_id
    if play_button.active:
        play_button.label = 'â�šâ�š Pause'

        # (re)start animation
        if callback_id is not None:
            curdoc().remove_periodic_callback(callback_id)
        callback_id = curdoc().add_periodic_callback(animate_update, animation_interval_slider_x1000.value)

    else:
        play_button.label = 'â–º Play'

        # stop animation
        if callback_id is not None:
            curdoc().remove_periodic_callback(callback_id)
            callback_id = None

def animate_cb(_attrname, _old, _new):
    # ignore input and run animation (todo: make this function unnecessary - there has to be a better way
    animate()
play_button.on_change('active', animate_cb)
animation_interval_slider_x1000.on_change('value', animate_cb)


# IVT threshold time_slider #
#############################
ivt_thresh_slider = Slider(start=0, end=500, step=1, value=ivt_threshold, title="IVT velocity threshold",
                           width=width_three)
# callback_policy="throttle", callback_throttle=ivt_slider_delay_velo_ms) doesn't work :(
# todo: remove ivt_threshold? => slider should be enough
def ivt_thresh_slider_cb(_attrname, _old, new):
    global ivt_threshold
    ivt_threshold = new
    predictor_parameter_changed()
    # update the velocity plot, when there is no further change)
    run_delayed(ivt_slider_delay_velo_ms, "segments", update_classification)
ivt_thresh_slider.on_change('value', ivt_thresh_slider_cb)


# IVT min duration time_slider #
################################
ivt_min_dur_slider = Slider(start=0, end=2, step=.01, value=ivt_min_dur, title="IVT minimal duraton",
                            width=width_two)
# callback_policy="throttle", callback_throttle=ivt_slider_delay_velo_ms) doesn't work :(
# todo: remove ivt_min_dur? => slider should be enough
def ivt_min_dur_slider_cb(_attrname, _old, new):
    global ivt_min_dur
    ivt_min_dur = new
    predictor_parameter_changed()
    # update the velocity plot, when there is no further change)
    run_delayed(ivt_slider_delay_velo_ms, "segments", update_classification)
ivt_min_dur_slider.on_change('value', ivt_min_dur_slider_cb)

# Heat Map #
############

heat_map_div = Div(text="<b>Settings for Heatmap</b>")

heat_map_show_button = Button(label="Show Heat Map", width=width_half, button_type='default')


def heat_map_show_button_cb(_event):
    heat_map_show_button.button_type = 'danger'
    run_delayed(100, 'Heat Map', show_heat_map)


def show_heat_map():
    eval_heat()
    heat_map_show_button.button_type = 'success'


heat_map_show_button.on_event(ButtonClick, heat_map_show_button_cb)

# store all maps
map_image = []
map_remove_button = Button(label='Remove Maps', width=width_one)

def turn_map_image_invisible():
    """ bad hack to remove the old images """
    global map_image

    for mapi in map_image:  # bad hack to remove the old images
        mapi.visible = False

def map_remove_button_cb(_event):
    turn_map_image_invisible()


map_remove_button.on_click(map_remove_button_cb)


def eval_heat():

    global cog_fix_all, cog_sac_all, map_image

    if clf_map_show_select.active == 0:
        cog_eval = cog_fix_all
    else:
        cog_eval = cog_sac_all

    if cog_eval is None:
        calculate_cog('all')
        eval_heat()
        return

    turn_map_image_invisible()

    # noinspection PyUnresolvedReferences
    map_image.append(show_heatmap(cog_eval['cog_x'], cog_eval['cog_y'], color='yellow'))

    return


# Map Information #
###############

map_information_div = Div(text="Run Heat Map or Prediction Map to collect Information", width=width_one)


def update_map_information(pcv_fix_r, pcv_sac_r) -> None:
    """ Calculate statistics of Fixations and Saccades

    :param pcv_fix_r: prediction correctness vales for fixations (zeros for heatmap - only to count them)
    :param pcv_sac_r: prediction correctness vales for saccades (zeros for heatmap - only to count them)
    """
    pos_pcv_fix_r = [val for val in pcv_fix_r if val > 0]
    neg_pcv_fix_r = [val for val in pcv_fix_r if val < 0]
    pos_pcv_sac_r = [val for val in pcv_sac_r if val > 0]
    neg_pcv_sac_r = [val for val in pcv_sac_r if val < 0]

    map_information_div.text = \
        '{len_fix} Fixations / Saccades {len_sac}<br>' \
        '{sum_pcv_fix:0.3f} sum of pcv {sum_pcv_sac:0.3f}<br>' \
        '{sum_pos_pcv_fix:0.3f} sum positive pcv {sum_pos_pcv_sac:0.3f}<br>' \
        '{sum_neg_pcv_fix:0.3f} sum negative pcv {sum_neg_pcv_sac:0.3f}<br>' \
        '{len_pos_pcv_fix} ({per_pos_pcv_fix:.1f}%) len positive pcv {len_pos_pcv_sac} ({per_pos_pcv_sac:.1f}%)<br>' \
        '{len_neg_pcv_fix} ({per_neg_pcv_fix:.1f}%) len negative pcv {len_neg_pcv_sac} ({per_neg_pcv_sac:.1f}%)<br>' \
        ''.format(
            len_fix=len(pcv_fix_r), len_sac=len(pcv_sac_r),
            sum_pcv_fix=sum(pcv_fix_r), sum_pcv_sac=sum(pcv_sac_r),
            sum_pos_pcv_fix=sum(pos_pcv_fix_r), sum_pos_pcv_sac=sum(pos_pcv_sac_r),
            sum_neg_pcv_fix=sum(neg_pcv_fix_r), sum_neg_pcv_sac=sum(neg_pcv_sac_r),
            len_pos_pcv_fix=len(pos_pcv_fix_r), len_pos_pcv_sac=len(pos_pcv_sac_r),
            per_pos_pcv_fix=len(pos_pcv_fix_r)/len(pcv_fix_r)*100,
            per_pos_pcv_sac=len(pos_pcv_sac_r)/len(pcv_sac_r)*100,
            len_neg_pcv_fix=len(neg_pcv_fix_r), len_neg_pcv_sac=len(neg_pcv_sac_r),
            per_neg_pcv_fix=len(neg_pcv_fix_r)/len(pcv_fix_r)*100,
            per_neg_pcv_sac=len(neg_pcv_sac_r)/len(pcv_sac_r)*100)
    return


# Prediction #
##############
predictor_select = Select(title="Predictor ([model_user]=User Doc, [model]=Repo)",
                          value='None', options=['None', 'Test'], width=width_one)
def predictor_select_cb(_attrname, _old, new):
    set_predictor(new)
    update_classification()
    update_time()
predictor_select.on_change('value', predictor_select_cb)

predictor_scan_button = Button(label="Scan for Predictors", width=width_half)
def predictor_scan_button_cb(_event):
    predictor_select.options = ['None', 'Test'] + method_list \
                               + get_local_files('[model]') + get_local_files('[model_user]')
predictor_scan_button.on_event(ButtonClick, predictor_scan_button_cb)
predictor_scan_button_cb(ButtonClick)

predictor_folder_button = Button(label="Open Model Folder", width=width_half)
def predictor_folder_button_cb(_event):
    webbrowser.open(replace_local_path('[model]'))
    webbrowser.open(replace_local_path('[model_user]'))
predictor_folder_button.on_event(ButtonClick, predictor_folder_button_cb)

predictor_toggle = Toggle(label="Show Prediction Result", active=False, width=width_one)
def predictor_toggle_cb(_attrname, _old, new):
    if new:
        trajectory_visible_toggle.active = False
    p_stim_evaluation.visible = new
    p_vel_evaluation.visible = new
    if new:
        update_classification()
predictor_toggle.on_change('active', predictor_toggle_cb)

predictor_type_select = Toggle(label="Only fixation", active=False, width=width_one)
def predictor_type_select_cb(_attrname, _old, _new):
    update_time()
predictor_type_select.on_change('active', predictor_type_select_cb)


prediction_color_slider = Slider(start=1, end=10, step=.1, value=1, title="Color Scaling", width=width_one)
def prediction_color_slider_cb(_attrname, _old, _new):
    run_delayed(delay_time_ms, "time_source", update_bar_source)
    run_delayed(delay_time_ms, "time", update_time)
prediction_color_slider.on_change('value', prediction_color_slider_cb)


prediction_color_text = Div(text="Color is calculated by<br>"
                                 "\"<i>probability for correct guess</i> - <i>other highest probability</i>\".<br> "
                                 " Visualization is from -1 (red) over 0 (white) to 1 (green).<br>"
                                 "The Scaler enhances visibility.", width=width_one)

prediction_color_auto_div = Div(text="AutoScale Prediction", width=width_one)
prediction_color_auto_radio = RadioButtonGroup(labels=["No", "Q", "1", "max", "90%",  "q3", "mean", "median", "q1"],
                                               active=4, width=width_one)
def prediction_color_auto_radio_cb(_attrname, _old, _new):
    auto_scale_prediction_color()
prediction_color_auto_radio.on_change('active', prediction_color_auto_radio_cb)

prediction_color_quartile_slider = Slider(start=0, end=1, step=0.01, value=1, title="Quartile Value", width=width_one)
def prediction_color_quartile_slider_cb(_attrname, _old, _new):
    if prediction_color_auto_radio.active != 1:
        prediction_color_auto_radio.active = 1  # will trigger auto_scale_prediction_color()
    else:
        run_delayed(100, 'color_scale', auto_scale_prediction_color)
prediction_color_quartile_slider.on_change('value', prediction_color_quartile_slider_cb)


def auto_scale_prediction_color():

    # get predictions for first user (others are not shown)
    _, _, prediction_results = eye_movements4trajectories[0]
    abs_values = [abs(x) for x in prediction_results]
    pre_max = max(abs_values)
    pre_mean = np.nanmean(abs_values)

    # note maximum on AutoScale
    prediction_color_auto_div.text = "AutoScale (first User max={max:.3f}, mean={mean:.3f})".format(
        max=pre_max, mean=pre_mean)

    if prediction_color_auto_radio.active == 2:
        prediction_color_slider.value = 1
    elif pre_max == 0:
        # no values: do nothing
        return
    elif prediction_color_auto_radio.active == 1:  # Q
        prediction_color_slider.value = 1 / np.nanquantile(abs_values, prediction_color_quartile_slider.value)
    elif prediction_color_auto_radio.active == 3:  # max
        prediction_color_quartile_slider.value = 1
    elif prediction_color_auto_radio.active == 4:  # 90 %
        prediction_color_quartile_slider.value = 0.9
    elif prediction_color_auto_radio.active == 5:  # q3
        prediction_color_quartile_slider.value = 0.75
    elif prediction_color_auto_radio.active == 6:  # mean
        prediction_color_slider.value = 1 / pre_mean
    elif prediction_color_auto_radio.active == 7:  # median / q2
        prediction_color_quartile_slider.value = 0.5
    elif prediction_color_auto_radio.active == 8:  # q1
        prediction_color_quartile_slider.value = 0.25
    # else: nothing to do
    return


prediction_width_slider = Slider(start=1, end=10, step=1, value=2, title="Width Prediction", width=width_one)
def prediction_width_slider_cb(_attrname, _old, new):
    p_stim_evaluation.glyph.line_width = new
prediction_width_slider.on_change('value', prediction_width_slider_cb)

def set_predictor(predictor_name: str = None) -> None:
    """ predictor is method from base_selection (e.g. 'paper-append')
    In the predictor a classifier could been saved.

    """
    global predictor

    predictor_copy_button.button_type = 'default'

    if predictor_name is None or predictor_name == 'None':
        update_predictor_text('No Predictor selected')
        predictor = None
    elif predictor_name in method_list:
        try:
            classifier_name = 'rf'  # dummy
            parser = get_parser_eye_movement_classifier('IVT')
            parser = get_parser_method(predictor_name, parser)
            parser = get_parser_classifier(classifier_name, predictor_name, parser)
            args = parser.parse_args([])
            classifier = get_classifier(classifier_name, arguments=args)
            predictor = get_method(predictor_name, classifier, arguments=args)
        except:
            update_predictor_text('Predictor could not been loaded!'.format(predictor_name))
            predictor = None

    else:
        predictor = load_pickle_file(predictor_name)
        update_predictor_text()
        predictor_user_select.value = user_select.value
        predictor_user_button_cb(ButtonClick)
        predictor_parameter_changed()

predictor_info = Div(text="", width=width_one)

def update_predictor_text(text: str = None) -> None:
    """ set the predictor_info text """
    if text is None:
        _, clf_names = predictor.get_classifiers_and_names()  # i think this would be better as attirbute
        predictor_info.text = '<h4>Predictor on dataset: "{dataset}"</h4>\n' \
                              '<div><b>Dataset Preparation:</b><br>' \
                              'Conversion: {convert}<br>' \
                              'Filtering: {filter}</div><br>' \
                              '<div><b>Movement Separator: IVT</b><br>' \
                              'vel_threshold = {vel}<br>min_fix_duration = {minfix}</div><br>' \
                              '<div><b>Label Classifier: {clf_names}</b></div>'.format(
            dataset=predictor.dataset_name,
            convert=predictor.preparation_steps['conversion'],
            filter=predictor.preparation_steps['filter_parameter'],
            vel=predictor.vel_threshold, minfix=predictor.min_fix_duration,
            clf_names=clf_names, users=sorted(predictor.users))
        # IVT will later be flexible
        update_predictor_user_select()
        predictor_classifier_select.options = ['all mean', 'clf mean'] + clf_names
    else:
        predictor_info.text = text

def predictor_do_copy():
    dataset_select.value = predictor.dataset_name

    # hardcode 'angel_deg'
    # hardcode 'savgol'
    savgol_filter_frame_slider.value = predictor.preparation_steps['filter_parameter']['frame_size']
    savgol_filter_poly_slider.value = predictor.preparation_steps['filter_parameter']['pol_order']

    ivt_min_dur_slider.value = predictor.min_fix_duration
    ivt_thresh_slider.value = predictor.vel_threshold

    training_user_limit_slider.value = len(predictor.users)

    predictor_parameter_changed()


predictor_copy_button = Button(label="Use Parameters of classifier", width=width_one)
def predictor_copy_button_cb(_event):
    predictor_copy_button.button_type = 'warning'
    run_delayed(100, 'predictor', predictor_do_copy)
predictor_copy_button.on_event(ButtonClick, predictor_copy_button_cb)

def predictor_parameter_changed():
    if predictor is None:
        return
    if dataset.dataset_name == predictor.dataset_name \
            and savgol_filter_frame_slider.value == predictor.preparation_steps['filter_parameter']['frame_size'] \
            and savgol_filter_poly_slider.value == predictor.preparation_steps['filter_parameter']['pol_order'] \
            and ivt_min_dur_slider.value == predictor.min_fix_duration \
            and ivt_thresh_slider.value == predictor.vel_threshold:
        predictor_copy_button.button_type = 'success'
    else:
        predictor_copy_button.button_type = 'danger'
    update_training_button()

predictor_user_button = Button(label="User Info Button", width=width_one)
def predictor_user_button_cb(_event):
    if user_select.value[0] in predictor.users:
        predictor_user_button.button_type = 'success'
        predictor_user_button.label = 'Selected User {} is in Training Data of Predictor'.format(
            user_select.value[0])
    else:
        predictor_user_button.button_type = 'warning'
        predictor_user_button.label = 'Selected User {} is not in Training Data of Predictor'.format(
            user_select.value[0])
predictor_user_button.on_event(ButtonClick, predictor_user_button_cb)

predictor_user_select = MultiSelect(title="Trained Users", value=['Others Selected'],
                                    options=['Others Selected'], width=width_one, height=180)
def predictor_user_select_cb(_attrname, _old, new):
    if 'Others Selected' in new:
        return
    if any([user not in predictor.users for user in new]):
        predictor_user_select.value = ['Others Selected']
    else:
        user_select.value = new
predictor_user_select.on_change('value', predictor_user_select_cb)

def update_predictor_user_select():
    predictor_user_select.options = ['Others Selected'] + sorted(predictor.users)


predictor_classifier_select = Select(title="Prediction by Type of Segments", value='clf mean',
                                     options=['all mean', 'clf mean'], width=width_one)

predictor_classification_select = Select(title="Classified Users", value='N/A',
                                         options=['N/A'], width=width_one)
def predictor_classification_select_cb(_attrname, _old, _new):
    update_predictor_classifier_text()
    # todo: this should also select the user to show the prediction.
predictor_classification_select.on_change('value', predictor_classification_select_cb)


def predictor_classifier_select_cb(_attrname, _old, _new):
    update_predictor_classifier_text()
predictor_classifier_select.on_change('value', predictor_classifier_select_cb)

predictor_classifier_cache_toggle = Toggle(label="Use Cache", active=True, width=width_half)

predictor_classifier_info_overview = Div(text="", width=width_one)
predictor_classifier_info_detail = Div(text="", width=width_one)
def update_predictor_classifier_text() -> None:
    """ set the predictor_classifier_info_detail text
    Uses labeled_predicted_probabilities (DataFrame) with subsets x classes+1.
    Columns are the class names + sample_type (defining wich classifer()
    """

    if len(labeled_predicted_probabilities) == 0:
        predictor_classifier_info_detail.text = ''
        predictor_classifier_info_overview.text = ''
        return

    user = predictor_classification_select.value

    classifiers = ['all mean', 'clf mean', 'fix', 'sac']

    # Brute Force
    text = '<div><b>Overview of Segment Prediction of "{user}"</b></div>'.format(user=user)

    for classifier in classifiers:

        selected_probabilities = select_probabilities(labeled_predicted_probabilities,
                                                      user,  # user
                                                      classifier)

        first_by_vote, votes = get_major_vote(selected_probabilities)
        first_by_value, values = get_mean_vote(selected_probabilities)

        win = '<font color="green"><b>{}</b></font>'
        lose = '<font color="red"><b>{}</b></font>'

        second_vote = [votes[1] if len(votes) > 1 else 0][0]
        second_value = values[1]  # can not be single, like voting
        win_major = [win.format('WIN') if user == first_by_vote else lose.format('LOSE')][0]
        diff_major = [votes[user] - second_vote if user == first_by_vote
                      else votes[0] - votes[user]][0]
        win_value = [win.format('WIN') if user == first_by_value else lose.format('LOSE')][0]
        diff_value = [values[user] - second_value if user == first_by_value
                      else values[0] - values[user]][0]

        text = text + '<br>' \
                      '<div><b>"{clf}"</b><br>' \
                      '<div style="text-align:left;">Votes: {prob_votes:.3f} = {user_votes} / {all_votes}' \
                      '&nbsp;&nbsp;&nbsp;&nbsp;<span style="float:right;">' \
                      '{win_major} by {diff_major} &#8793; {prop_major:.3f}</span></div>' \
                      '<div style="text-align:left;">Mean: {user_value:.3f}' \
                      '&nbsp;&nbsp;&nbsp;&nbsp;<span style="float:right;">' \
                      '{win_value} by {diff_value:.3f}</span></div></div>'.format(
            clf=classifier,
            user_votes=votes[user], all_votes=sum(votes),
            prob_votes=votes[user] / sum(votes), user_value=values[user],
            win_major=win_major, diff_major=diff_major, prop_major=diff_major / sum(votes),
            win_value=win_value, diff_value=diff_value,
        )

        first_by_vote_styled = [win if user == first_by_vote else lose][0].format(first_by_vote)
        first_by_value_styled = [win if user == first_by_value else lose][0].format(first_by_value)

        if classifier == predictor_classifier_select.value:
            predictor_classifier_info_detail.text = \
                '<div><b>"{clf}" for user {user}</b></div><br>' \
                '<div>{vote_major} by Vote:<br>{values_major}</div><br>' \
                '<div>{vote_mean} by Mean:<br>{values_mean}</div>'.format(
                    user=user, clf=classifier,
                    vote_major=first_by_vote_styled,
                    vote_mean=first_by_value_styled,
                    values_major=votes,
                    values_mean=values)

    predictor_classifier_info_overview.text = text

    return

predictor_next_user_button = Button(label="Next trained User", width=width_half)
def predictor_next_user_button_cb(_event):
    predictor_user_select.value = [get_next_in_list(predictor_user_select.options, predictor_user_select.value[0])]
predictor_next_user_button.on_event(ButtonClick, predictor_next_user_button_cb)

cog_fix_all = None
cog_sac_all = None
cog_fix_eval = None
cog_sac_eval = None


def prepare_cog_segments(start_samples: list, type_names: list, prediction_correctness_values: list,
                         complete_trajectory: list, segment_type: str) -> tuple:
    """ Calculate all center of gravities for specific segment_type

    :param start_samples: index of sample beginning in trajectory
    :param type_names: type of segment
    :param prediction_correctness_values: pcv for segment
    :param complete_trajectory: trajectory with coordinates [x, y]?
    :param segment_type: Segment type to isolate
    :return: cog_x, cog_y, pcv_r

    todo: it is maybe necessary to return full segments to better describe saccades.
          The handling than has to be later.
          Maybe have sacc start and sacc end

    """
    index_fixation = find([typ == segment_type for typ in type_names])
    pcvs_fix = [result for (result, type_name) in zip(prediction_correctness_values, type_names)
                if type_name == segment_type]

    segments = [complete_trajectory[
                range(start_samples[i_fix],
                      start_samples[i_fix + 1]), :]
                for i_fix in index_fixation]

    cog_x = []
    cog_y = []
    pcv_r = []

    for (segment, eval_res) in zip(segments, pcvs_fix):
        cog_x.append(np.nanmean(segment[:, 0]))
        cog_y.append(np.nanmean(segment[:, 1]))
        pcv_r.append(eval_res)

    return cog_x, cog_y, pcv_r


def calculate_cog(user_selection: str):
    """ Go through all trained user and get center of gravities. Save them in global cog variables.

    :param user_selection: 'all' every user, 'trained' only user which where trained

    todo: maybe necessary to do modifications to trajectory
    """

    print('Mapping user')

    # Hardcode to count fixations in bio-tex paragraphs
    if dataset_select.value.startswith("BioEyeTex"):
        count_biotex_fixations = True
    else:
        count_biotex_fixations = False
    biotex_fixation_list = []

    global cog_fix_eval, cog_sac_eval, cog_fix_all, cog_sac_all
    cog_fix_x = []
    cog_fix_y = []
    pcv_fix_r = []
    cog_sac_x = []
    cog_sac_y = []
    pcv_sac_r = []

    if user_selection == 'all':
        user_list = user_select.options
    elif user_selection == 'trained':
        user_list = predictor_user_select.options[1:]
    else:
        raise Exception()

    # todo: could be done parallel?
    for user in user_list:  # first option is "others selected" maybe this should be removed

        print(" Calculating: ", user)

        trajectory = load_data(user, stimuli_select.value, add_duplicate_stats=False)

        # if possible, correct alignment
        if user in user_alignment.index:
            trajectory.offset(-user_alignment.loc[user, 'x'],
                              user_alignment.loc[user, 'y'])  # y is inverted
        complete_trajectory = trajectory.get_trajectory('pixel_image')

        start_samples, type_names, prediction_correctness_values = evaluate_trajectory(trajectory, lazy=False)

        # Prepare Segments
        cog_fix_x_new, cog_fix_y_new, pcv_fix_r_new = prepare_cog_segments(start_samples, type_names,
                                                                           prediction_correctness_values,
                                                                           complete_trajectory, 'fixation')
        # Hardcode Bio-Tex: creating a dictionary to record the fixation regions
        if count_biotex_fixations:
            biotex_fixation_list.append(
                {'Participant': user,
                 **count_fixation_in_paragraphs(cog_fix_x_new, cog_fix_y_new)})
        
        cog_sac_x_new, cog_sac_y_new, pcv_sac_r_new = prepare_cog_segments(start_samples, type_names,
                                                                           prediction_correctness_values,
                                                                           complete_trajectory, 'saccade')

        cog_fix_x.extend(cog_fix_x_new)
        cog_fix_y.extend(cog_fix_y_new)
        pcv_fix_r.extend(pcv_fix_r_new)
        cog_sac_x.extend(cog_sac_x_new)
        cog_sac_y.extend(cog_sac_y_new)
        pcv_sac_r.extend(pcv_sac_r_new)

    update_map_information(pcv_fix_r, pcv_sac_r)

    if user_selection == 'all':
        cog_fix_all = {'cog_x': np.asarray(cog_fix_x), 'cog_y': np.asarray(cog_fix_y), 'pcv': np.asarray(pcv_fix_r)}
        cog_sac_all = {'cog_x': np.asarray(cog_sac_x), 'cog_y': np.asarray(cog_sac_y), 'pcv': np.asarray(pcv_sac_r)}
    elif user_selection == 'trained':
        cog_fix_eval = {'cog_x': np.asarray(cog_fix_x), 'cog_y': np.asarray(cog_fix_y), 'pcv': np.asarray(pcv_fix_r)}
        cog_sac_eval = {'cog_x': np.asarray(cog_sac_x), 'cog_y': np.asarray(cog_sac_y), 'pcv': np.asarray(pcv_sac_r)}

    # Hardcode Bio-Tex: create a 'fixation_count.csv' file containing regions for all the users in the dataset
    if count_biotex_fixations:
        count_frame = pd.DataFrame(biotex_fixation_list)
        count_frame.set_index('Participant', drop=True, inplace=True)
        print("Mean Number of Fixations for Paragraphs:\n{}".format(count_frame.mean(axis=0)))
        count_frame.to_csv(os.path.join(os.path.join(os.path.dirname(dataset.align_folder)),
                                        '{set} Stimulus {sti}.csv'.format(
                                            set=dataset_select.value, sti=stimuli_select.value)))

    print('Finished mapping user')

    return


predictor_user_map_div = Div(text="<b>Settings for Prediction Correctness Mapp</b>")
predictor_user_map_select = RadioButtonGroup(labels=["full", "full-sum", "full-max", "full-len",
                                                     "good", "bad", "both"], active=0,
                                             width=width_one)
map_bins_slider = Slider(title="Bins", value=500, start=1, end=1000, step=1, width=width_one)
map_gauss_slider = Slider(title="Sigma (for GaussFilter)", value=5, start=0, end=10, step=1,
                          width=width_one)
map_opacity_slider = Slider(title="Maximal Opacity", value=0.9, start=0.05, end=1, step=0.05, width=width_one)
clf_map_show_select = RadioButtonGroup(labels=["Fixations", "Saccades"], active=0, width=width_half)
predictor_user_map_show_button = Button(label="Show Prediction Map", width=width_half)
def predictor_user_map_show_button_cb(_event):
    calculate_pch()
predictor_user_map_show_button.on_event(ButtonClick, predictor_user_map_show_button_cb)


# noinspection PyUnresolvedReferences
def calculate_pch():
    """ Calculate the prediction correctness heatmap and update map_image.

    Uses the global center of gravity variables cog_fix_eval, cog_sac_eval.
    """

    global cog_fix_eval, cog_sac_eval, map_image

    if clf_map_show_select.active == 0:
        cog_eval = cog_fix_eval
    else:
        cog_eval = cog_sac_eval

    if cog_eval is None:
        calculate_cog('trained')
        calculate_pch()
        return

    turn_map_image_invisible()

    if 0 <= predictor_user_map_select.active <= 3:
        # full

        cog_acc_x = cog_eval['cog_x']
        cog_acc_y = cog_eval['cog_y']
        pcv_res = cog_eval['pcv']
        # making weights of good and bad similar
        if predictor_user_map_select.active == 1:
            # normalized by sum
            pcv_res[pcv_res > 0] /= sum(pcv_res[pcv_res > 0])
            pcv_res[pcv_res < 0] /= - sum(pcv_res[pcv_res < 0])
        elif predictor_user_map_select.active == 2:
            # normalized by max
            pcv_res[pcv_res > 0] /= max(pcv_res[pcv_res > 0])
            pcv_res[pcv_res < 0] /= - min(pcv_res[pcv_res < 0])
        elif predictor_user_map_select.active == 3:
            # normalized by len
            pcv_res[pcv_res > 0] /= len(pcv_res[pcv_res > 0])
            pcv_res[pcv_res < 0] /= len(pcv_res[pcv_res < 0])
        map_image.append(show_heatmap(cog_acc_x, cog_acc_y, pcv_res))

    if predictor_user_map_select.active == 4 \
            or predictor_user_map_select.active == 6:
        # good / both
        selection_bool = cog_eval['pcv'] > 0
        cog_acc_x = cog_eval['cog_x'][selection_bool]
        cog_acc_y = cog_eval['cog_y'][selection_bool]
        pcv_res = cog_eval['pcv'][selection_bool]
        map_image.append(show_heatmap(cog_acc_x, cog_acc_y, pcv_res))

    if predictor_user_map_select.active == 5 \
            or predictor_user_map_select.active == 6:
        # bad / both
        selection_bool = cog_eval['pcv'] < 0
        cog_acc_x = cog_eval['cog_x'][selection_bool]
        cog_acc_y = cog_eval['cog_y'][selection_bool]
        pcv_res = cog_eval['pcv'][selection_bool]
        map_image.append(show_heatmap(cog_acc_x, cog_acc_y, pcv_res))

    return


def show_heatmap(coordinates_x: list, coordinates_y: list, weights: list = None, color: str = None):
    """ Add a alpha (green) Heatmap to trajectory plot.
        Negative weights make it red.

        :return: handle to image
    """

    if weights is None:
        weights = [1] * len(coordinates_x)

    if color is None or color == 'green':
        # red green
        color_low = (255, 0, 0)
        color_high = (0, 255, 0)
    elif color == 'yellow':
        # blue yellow
        color_low = (0, 0, 255)
        color_high = (255, 255, 0)
    else:
        raise Exception

    # find min max coordinates to make bins symmetrical
    diff_x = max(coordinates_x) - min(coordinates_x)
    diff_y = max(coordinates_y) - min(coordinates_y)

    # histogram2d(y,x), bins are twisted!
    if diff_x > diff_y:
        other_bins = int(map_bins_slider.value * diff_x / diff_y)
        bins = [map_bins_slider.value, other_bins]
    else:
        other_bins = int(map_bins_slider.value * diff_y / diff_x)
        bins = [other_bins, map_bins_slider.value]

    # calculate heatmap
    histo_array, yedges, xedges = np.histogram2d(coordinates_y, coordinates_x, bins=bins, weights=weights)

    shading_steps = 200

    # noinspection PyArgumentList
    factor = -histo_array.min() / histo_array.max()
    if abs(factor) > 1:
        low_color = color_low
        low_steps = shading_steps
        if factor < 0:
            high_color = (0, 0, 0)
            high_steps = 0
        else:
            high_color = tuple([c * factor for c in color_high])
            high_steps = int(shading_steps / factor)
    else:
        # minimum closer to 0 than maximum (minimum could be 0)
        high_color = color_high
        high_steps = shading_steps
        if factor < 0:
            low_color = (0, 0, 0)
            low_steps = 0
        else:
            low_color = tuple([c * factor for c in color_low])
            low_steps = int(shading_steps * factor)


    # bluring
    histo_array = gaussian_filter(histo_array, sigma=map_gauss_slider.value)

    return p_stim.image(image=[histo_array], x=xedges[0], y=yedges[0],
                        dw=xedges[-1] - xedges[0], dh=yedges[-1] - yedges[0],
                        palette=alpha_pallet(low_color, low_steps, map_opacity_slider.value)[::-1] +
                                alpha_pallet(high_color, high_steps, map_opacity_slider.value)[1:])


# Training #
############

def training_prepare_arguments() -> dict:
    arguments = {'dataset': dataset.dataset_key,
                 'classifier': training_classifier_textbox.value,
                 'method': training_method_textbox.value,
                 'modelfile': training_filename_textbox.value,
                 'label': training_label_textbox.value}

    if training_seed_textbox.value:
        arguments['seed'] = int(training_seed_textbox.value)

    if training_user_limit_slider.value != training_user_limit_slider.end:
        arguments['user_limit'] = training_user_limit_slider.value

    # todo: these is hardcode, but i doesn't want the parameters to appear if not important...
    #   maybe let them appear always or create a default value in base_selection, which we can access.
    if ivt_thresh_slider.value != 50:
        arguments['ivt_threshold'] = ivt_thresh_slider.value
    if ivt_min_dur_slider.value != 0.1:
        arguments['ivt_min_fix_time'] = ivt_min_dur_slider.value

    # todo: these is hardcode, it have to be implemented someday
    #   maybe let them appear always or create a default value in base_selection, which we can access.
    if savgol_filter_frame_slider.value != 15 or savgol_filter_poly_slider.value != 6:
        arguments['filter_parameter'] = ['frame_size', str(savgol_filter_frame_slider.value),
                                         'pol_order', str(savgol_filter_poly_slider.value)]

    return arguments

def update_training_button():
    training_button.button_type = 'default'
    training_parameter_textbox.value = " ".join(prepare_input(**training_prepare_arguments()))
    training_button.label = "Run Training"

def update_training_button_cb(_attrname, _old, _new):
    update_training_button()

def training_button_cb(_event):

    training_button.button_type = 'danger'
    training_button.label = 'Running'
    run_delayed(100, 'training', do_training)

def do_training():

    call_train(**training_prepare_arguments())

    training_button.button_type = 'success'
    training_button.label = 'Finished'
    predictor_scan_button_cb(ButtonClick)
    predictor_select.value = training_filename_textbox.value

def training_parameter_textbox_cb(_attrname, _old, _new):
    update_training_button()

# todo: read arguments and try to parse ... that would be really nice XD
training_parameter_textbox = TextInput(
    title='Parameter Line (read only; execution may take long; Predictor will be loaded when finished))',
    value='', width=width_full)
training_parameter_textbox.on_change('value', training_parameter_textbox_cb)

training_button = Button(label='Run Training', width=width_half, button_type='warning')
training_button.on_event(ButtonClick, training_button_cb)

training_classifier_textbox = TextInput(title='--classifier', value='rf', width=width_half)
training_classifier_textbox.on_change('value', update_training_button_cb)

training_method_textbox = TextInput(title='--method', value='paper-append', width=width_half)
training_method_textbox.on_change('value', update_training_button_cb)

training_filename_textbox = TextInput(title='--modelfile', value='[model]testmodel', width=width_half)
training_filename_textbox.on_change('value', update_training_button_cb)

training_seed_textbox = TextInput(title='--seed', value='', width=width_half)
training_seed_textbox.on_change('value', update_training_button_cb)

training_label_textbox = TextInput(title='--label', value='user', width=width_half)
training_label_textbox.on_change('value', update_training_button_cb)

training_user_limit_slider = Slider(title='--user_limit', value=10, start=2, end=len(user_select.options),
                                    width=width_one)
training_user_limit_slider.on_change('value', update_training_button_cb)


# Save/load Settings #
######################

# this list give the names an order of activation of settings.
settings_list = [
    'dataset_select', 'stimuli_select', 'user_select', 'time_slider',
    'user_color_toggle',  # colors are separated below
    'trajectory_visible_toggle', 'p_stim_trajectory_marker_toggle',
    'marker_fixation_circle_toggle', 'marker_saccades_toggle',
    'marker_fixation_count_toggle', 'marker_fixation_text_toggle',
    'marker_duration_slider_text', 'marker_duration_slider_alpha',
    'hold_left_time_toggle', 'stop_right_time_toggle',
    'animation_step_slider', 'animation_speed_slider_log', 'animation_interval_slider_x1000',
    'marker_trajectory_toggle', 'marker_trajectory_last_toggle',
    'trajectory_marker_color_picker', 'trajectory_marker_color_toggle',
    'marker_fixation_number_color_picker', 'marker_fixation_number_color_toggle',
    'marker_fixation_marker_color_picker', 'marker_fixation_marker_color_toggle',
    'image_color_slider', 'image_fade_slider',
    'update_table_toggle',
    'mark_nan_toggle', 'data_interpolate_toggle',
    'clip_toggle', 'data_center_toggle',
    'smooth_toggle', 'savgol_filter_frame_slider', 'savgol_filter_poly_slider',
    'duplicate_toggle', 'duplicate_slider',
    'data_scaling_slider', 'x_offset_slider', 'y_offset_slider',
    'ivt_thresh_slider', 'ivt_min_dur_slider',
    'prediction_color_slider', 'prediction_width_slider', 'trajectory_width_slider',
]


def save_settings(filename: str):
    """ Store every Setting in an list and save them in a pickle file."""

    # Settings for sliders, button and pickers, which can be automated very easy.
    settings_from_name = dict.fromkeys(settings_list)

    for setting in settings_from_name:
        instance = getattr(sys.modules[__name__], setting)
        if hasattr(instance, 'value'):
            status = instance.value
        elif hasattr(instance, 'active'):
            status = instance.active
        elif hasattr(instance, 'color'):
            status = instance.color
        else:
            raise Exception('Type is unknown')

        if isinstance(status, list):
            settings_from_name.update({setting: status[:]})  # make a unique list, somehow necessary for pickle (Martin)
        else:
            settings_from_name.update({setting: status})

    # other settings
    settings_user_color = [cp.color for cp in user_color_picker]
    settings_view = {'tra_x': [p_stim.x_range.start, p_stim.x_range.end],
                     'tra_y': [p_stim.y_range.start, p_stim.y_range.end],
                     'vel_x': [p_vel.x_range.start, p_vel.x_range.end],
                     'vel_y': [p_vel.y_range.start, p_vel.y_range.end]}

    other_settings = {'settings_user_color': settings_user_color,
                      'settings_view': settings_view}

    save_parameter_file(filename, {'settings_from_name': settings_from_name,
                                   'other_settings': other_settings})

    print('Saved values of:', list(settings_from_name), ', views, and user colors.')


def load_settings(filename: str,
                  load_simple: bool = False,
                  load_color: bool = False,
                  load_view: bool = False):
    """ Activate values found in settings file on hard drive. """
    parameter_dict = load_parameter_file(filename, ['settings_from_name', 'other_settings'])

    if not load_simple and not load_color and not load_view:
        # if nothing specified, do all
        load_simple = True
        load_color = True
        load_view = True

    if load_simple:
        load_settings_simple(parameter_dict['settings_from_name'])

    if load_color:
        if isinstance(parameter_dict['other_settings'], dict):
            # todo: backwards compatibility, remove when transferred everything
            if hasattr(parameter_dict['other_settings'], 'settings_user_color'):
                load_settings_color(parameter_dict['other_settings']['settings_user_color'])
            else:
                print("settings_user_color couldn't be loaded")
        else:
            load_settings_color(parameter_dict['other_settings'])

    if load_view and isinstance(parameter_dict['other_settings'], dict):
        # todo: backwards compatibility, remove when transferred everything
        load_settings_view(parameter_dict['other_settings']['settings_view'])


def load_settings_simple(settings_from_name):
    for setting in settings_list:
        if setting not in settings_from_name:
            print(setting, "could not ben loaded")
            continue

        instance = getattr(sys.modules[__name__], setting)
        if hasattr(instance, 'value'):
            instance.value = settings_from_name[setting]
        elif hasattr(instance, 'active'):
            instance.active = settings_from_name[setting]
        elif hasattr(instance, 'color'):
            instance.color = settings_from_name[setting]
        else:
            raise Exception('Type is unknown')

    if len(set(settings_from_name) - set(settings_list)) > 0:
        print('Ignored: ', set(settings_from_name) - set(settings_list))

    print('Loaded values of:', set(settings_from_name) & set(settings_list))


def load_settings_color(settings_user_color):
    if len(user_color_picker) != len(settings_user_color):
        print("Colors can't be loaded! Size is different")
    else:
        for (cp, color) in zip(user_color_picker, settings_user_color):
            cp.color = color
        print('Loaded user colors: ', settings_user_color)


def load_settings_view(settings_view):
    p_stim.x_range.start, p_stim.x_range.end = settings_view['tra_x']
    p_stim.y_range.start, p_stim.y_range.end = settings_view['tra_y']
    p_vel.x_range.start, p_vel.x_range.end = settings_view['vel_x']
    p_vel.y_range.start, p_vel.y_range.end = settings_view['vel_y']
    print('Loaded views: ', settings_view)


settings_select = Select(title="Select Settings", value='None', options=['None'], width=width_one)
settings_select2 = TextInput(title="Name ([bus]=User Documents, [bet]=Repository)", value='[bus]default.pickle',
                             width=width_one)
def settings_select_cb(_attr, _old_, new):
    settings_select2.value = new
settings_select.on_change('value', settings_select_cb)

settings_load_button_all = Button(label="load all", width=width_half)
def settings_load_button_all_cb(_event):
    load_settings(settings_select2.value)
settings_load_button_all.on_event(ButtonClick, settings_load_button_all_cb)

settings_load_button_view = Button(label="load view", width=width_half)
def settings_load_button_view_cb(_event):
    load_settings(settings_select2.value, load_view=True)
settings_load_button_view.on_event(ButtonClick, settings_load_button_view_cb)

settings_save_button = Button(label="save", width=width_half)
def settings_save_button_cb(_event):
    save_settings(settings_select2.value)
    settings_scan_button_cb(ButtonClick)
settings_save_button.on_event(ButtonClick, settings_save_button_cb)

settings_scan_button = Button(label="Scan for Settings", width=width_half)
def settings_scan_button_cb(_event):
    settings_select.options = ['None'] + get_local_files('[bus]') + get_local_files('[bet]')
settings_scan_button.on_event(ButtonClick, settings_scan_button_cb)
settings_scan_button_cb(ButtonClick)

settings_folder_button = Button(label="Open Setting Folders", width=width_half)
def settings_folder_button_cb(_event):
    webbrowser.open(replace_local_path('[bus]'))
    webbrowser.open(replace_local_path('[bet]'))
settings_folder_button.on_event(ButtonClick, settings_folder_button_cb)

# Debugging #
#############

command_text = TextInput(title="Command", value="'Hello World'", width=width_one, visible=debugMode)

command_button_run = Button(label="Run Command", width=width_half, visible=debugMode)
def command_button_run_cb(_event):
    if debugMode:
        exec(command_text.value)
command_button_run.on_event(ButtonClick, command_button_run_cb)

command_button_print = Button(label="Print Variable", width=width_half, visible=debugMode)
def command_button_print_cb(_event):
    if debugMode:
        exec("print(" + command_text.value.partition("=")[0] + ")")


command_button_print.on_event(ButtonClick, command_button_print_cb)


##########
# Layout #
##########

doc_link = 'https://gitlab.informatik.uni-bremen.de/cgvr/smida2/schau_mir_in_die_augen/' \
           '-/blob/master/documentation/VISUALIZATION.md#'

# Map Control
map_layout = [map_bins_slider, map_gauss_slider, map_opacity_slider]

# Heat Map Group
heat_map_layout = [heat_map_div,
                   [map_layout],
                   [clf_map_show_select, heat_map_show_button]]

# Prediction Map Group
prediction_map_layout = [predictor_user_map_div,
                         [map_layout],
                         predictor_user_map_select,
                         [clf_map_show_select, predictor_user_map_show_button]]

# Configs #
###########

#: Layout for Configuration
layout_config = layout([
    [[[settings_scan_button,
       settings_folder_button],
      settings_select],
     [[settings_load_button_all, settings_load_button_view, settings_save_button],
      settings_select2],
     [[command_button_run, command_button_print],
      command_text]]
])

# Settings #
############

headline_Settings = Div(text="<h2>Settings</h2>", width=width_three + width_two)
remark_data = Div(text="<h3>Select Data</h3>", width=width_one)
remark_annotation = Div(text="<h3>Select Annotation</h3>", width=width_one)
remark_animation = Div(text="<h3>Control Animation</h3>", width=width_one + width_one)
remark_customization = Div(text="<h3>Customization</h3>", width=width_one)

#: Layout for Settings
layout_top_row = layout([  # strange, should work without double brackets
    #    headline_Settings,
    [remark_data, remark_annotation, remark_animation, remark_customization],
    [
        [dataset_select, stimuli_select, user_select],  # Select Database, Stimulus and User
        [
            [trajectory_visible_toggle, p_stim_trajectory_marker_toggle],
            [marker_fixation_circle_toggle, marker_saccades_toggle],
            [marker_fixation_count_toggle, marker_fixation_text_toggle],
            marker_duration_slider_text,
            marker_duration_slider_alpha
        ],
        [
            [play_button, sample_rate_button],
            [hold_left_time_toggle, stop_right_time_toggle],
            [update_velocity_toggle, update_table_toggle],
            [next_dataset_button],
            [next_stimulus_button],
            [next_user_button],
        ],
        [animation_step_slider,
         animation_speed_slider_log,
         animation_interval_slider_x1000,
         [time_slider_reset_button, marker_trajectory_ping_button],
         [marker_trajectory_toggle, marker_trajectory_last_toggle]],
        [[trajectory_marker_color_picker, trajectory_marker_color_toggle],
         [marker_fixation_number_color_picker, marker_fixation_number_color_toggle],
         [marker_fixation_marker_color_picker, marker_fixation_marker_color_toggle],
         image_color_slider,
         image_fade_slider]
    ]])

# Stimuli Plot #
################

headline_trajectory = Div(text="<h2>Trajectory Plot</h2>", width=width_three)
headline_table = Div(text="<h2>Feature Table</h2>", width=width_one)
remark_table = Div(text="""<br>Features computed on the selected part of the trajectory.
                     NaNs are interpolated but it is <b>not</b> clipped to the image.""",
                   width=width_two)

#: Layout for Stimuli Plot
layout_main_stimulus = layout(
    [headline_trajectory, headline_table, update_table_toggle, feature_table_toggle],
    [time_slider, remark_table],
    [p_stim, feature_table, [
        Div(text='<b>Heat Map / Prediction Map / Prediction Value<b>', width=width_one),
        map_remove_button,
        [map_layout],
        Div(width=width_one),
        [heat_map_show_button, clf_map_show_select],
        Div(width=width_one),
        map_information_div,
        Div(width=width_one),
        [predictor_user_map_show_button, clf_map_show_select], predictor_user_map_select,
        Div(width=width_one),
        predictor_toggle,
        predictor_type_select,
        prediction_color_auto_div, prediction_color_auto_radio, prediction_color_quartile_slider,
        prediction_color_slider,
        prediction_color_text,
        prediction_width_slider,
        trajectory_width_slider,
        predictor_next_user_button,
    ]],
    [[[mark_nan_toggle, data_interpolate_toggle, update_table_toggle, play_button],
      [clip_toggle, data_center_toggle],
      [smooth_toggle, savgol_filter_frame_slider, savgol_filter_poly_slider],
      [duplicate_toggle, duplicate_slider]
      ],
     [data_scaling_slider,
      x_offset_slider,
      y_offset_slider,
      [alignment_load_button, alignment_auto_toggle, alignment_save_button],
      reset_values_button]
     ])

# Velocity Plot #
#################

headline_velocities = Div(text="""
     <h2>Velocity Plot</h2>
     Click on a box
     (<font color="#DDDDDD">Saccades:white</font>
     or <font color="#718dbf">Fixations:blue</font>)
     to select the time range.""", width=width_three)

#: Layout for Veclocity Plot
layout_separation_graphs = layout(
    [headline_velocities],
    [ivt_thresh_slider, ivt_min_dur_slider],
    [p_vel, p_vel_info],
    [time_slider, update_velocity_toggle])

# Prediction #
##############

headline_prediction = Div(text='<h2><a href="{}#Prediction", target="_blank">Prediction</a></h2>'.format(doc_link),
                          width=width_three)

#: Layout for Prediction
layout_prediction = layout(
    [headline_prediction],
    [[[predictor_scan_button, predictor_folder_button], predictor_select, predictor_info],
     [predictor_toggle, predictor_classifier_cache_toggle, predictor_copy_button,
      predictor_user_button, predictor_user_select],
     [predictor_classification_select, predictor_classifier_select, predictor_classifier_info_detail],
     predictor_classifier_info_overview,
     prediction_map_layout,
     ])

# Training #
##############

headline_training = Div(text="""
     <h2>Training</h2>
     Train with actual Parameters.</br>
     Use "[model]" as prefix for filename to save in the "SMIDA/model" folder.""", width=width_three)

#: Layout for Training
layout_training = layout(
    [headline_training],
    [training_classifier_textbox,
     training_method_textbox,
     training_filename_textbox, training_seed_textbox,
     training_user_limit_slider, training_label_textbox],
    [training_parameter_textbox],
    [training_button])


# Overall Layout #
##################

headline_SMIDA = Div(text='<h2><a href="{}", target="_blank" >SMIDA Visualization</a></h2>'.format(doc_link),
                     width=width_one)

curdoc().title = "SMIDA Visualization Tool"
curdoc().theme = 'dark_minimal'
curdoc().add_root(layout([
    [headline_SMIDA, user_color_toggle, [user_color_picker]],
    layout_config,
    layout_top_row,
    layout_main_stimulus,
    layout_separation_graphs,
    layout_prediction,
    layout_training,
], margin=layout_margin))

##################
# Initialization #
##################

# load initial data ("where humans look" as default - is prepared above)
#   will not load if not triggered
#   todo: local settings would be nice.
dataset_select.value = dataset_select.options[dataset_list.index(dataset)]

# hide Feature table
feature_table_toggle.active = False
