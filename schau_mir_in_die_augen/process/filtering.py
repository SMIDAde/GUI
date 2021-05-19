import numpy as np
from scipy.signal import savgol_filter


def savgol_filter_trajectory(xy: np.asarray, frame_size: int, pol_order: int):
    """ filter the data using savitzky_Golay

    :param xy: 2D array with equal length filtered: angle_x(Xorg), angle_y(Yorg)
    :param frame_size: used for savitzky_Golay filter
    :param pol_order: polynomial order used for savitzky_Golay filter

    :return: filtered x and y vectors
    """
    xf = savgol_filter(xy[:, 0], frame_size, pol_order)
    yf = savgol_filter(xy[:, 1], frame_size, pol_order)

    return np.asarray([xf, yf]).T


def remove_duplicates(xy, threshold: int = 0, verbose=False):
    """ replacing repetitive values with NaN

    @:param xy: ndarray [[x1,y1],[x2,y2],... ]
    @:param threshold [int]: ignore repetitions smaller than this
    @:return xy:

        with NaN at repetitions

        stats: (number of deleted samples, number of deleted sets)

    No change in data is not possible while assuming existence of noise."""

    assert len(xy.shape) == 2
    assert xy.shape[1] == 2

    removed_samples = 0
    removed_sets = 0

    def remove_duplicate(start, end):
        nonlocal xy, removed_samples, removed_sets
        end = end + 1
        if end - start <= threshold:
            if verbose:
                print(" - Ignored {number} samples ({start} to {end})".format(number=end - start,
                                                                              start=duplicate, end=end))
        else:
            xy[start:end + 1] = (np.nan, np.nan)
            removed_samples += end - start
            removed_sets += 1
            if verbose:
                print(" - Removed {number} samples ({start} to {end})".format(number=end - start,
                                                                              start=duplicate, end=end))

    if verbose:
        print("RUN: remove_duplicates")

    duplicate = None
    for idd, values in enumerate(xy[1:]):
        if not duplicate:
            if all(xy[idd] == values):
                # found first duplicate
                duplicate = idd+1
        elif any(xy[duplicate] != values):
            # found last duplicate
            remove_duplicate(duplicate, idd)
            duplicate = None

    if duplicate:
        # everything at the end is duplicate
        remove_duplicate(duplicate, len(xy)-1)

    factor_removed = removed_samples/xy.shape[0]
    if factor_removed > 0.5:
        print("W: {} % of samples removed".format(factor_removed*100))

    if verbose:
        print(" - Removed {number} samples ({percent} % ) in total".format(number=removed_samples,
                                                                           percent=factor_removed*100))
        print("END: remove_duplicates")

    return xy, (removed_samples, removed_sets)


# this filter is no in use
def smooth_velocities(xf, yf, sample_rate):
    """ generate smooth velocities
    input:
    filtered x and y vectors
    sample_rate: Sensor sample rate in Hz

    Return:
    1D np.arrays of smoothed velocities
    """

    # calculate x velocity
    diff_x = xf[:-1] - xf[1:]
    vel_x = diff_x * sample_rate

    # calculate y velocity
    diff_y = yf[:-1] - yf[1:]
    vel_y = diff_y * sample_rate

    smoothed_velocities = np.asarray([vel_x, vel_y]).T
    norm_our_velocities = np.linalg.norm(smoothed_velocities, axis=1)

    return norm_our_velocities
