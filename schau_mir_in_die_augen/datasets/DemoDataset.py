""" implement a DemoDataset for testing """

import numpy as np
import random

from schau_mir_in_die_augen.datasets.DatasetBase import DatasetBase
from schau_mir_in_die_augen.datasets.DatasetFolder import DatasetFolder
from schau_mir_in_die_augen.process.conversion import convert_angles_deg_to_shifted_pixel_coordinates


# default parameters
default_none = None
default_number_user = 100
default_number_cases = 6
default_length_datasets = 1000
default_length_datasets_fluctuation = 0
default_male_ratio = 0.5
default_scale_male = 1
default_dataset_scaling = 250
default_sample_rate = 100
default_number_nan_parts = 0
default_max_lengths_nan_parts = 0


class DemoDataset(DatasetBase, DatasetFolder):
    """ Dummy dataset for Testing

        Default User: "User 1", "User 2", ...
        Default Case: "Case 1", "Case 2", ...

    """

    def __init__(self,
                 seed=default_none,
                 number_users: int = default_number_user,
                 user_limit: int = default_none,
                 sample_rate: float = default_sample_rate,
                 number_cases: int = default_number_cases,
                 training_cases: list = default_none,
                 testing_cases: list = default_none,
                 length_datasets: int = default_length_datasets,
                 length_datasets_fluctuation: int = default_length_datasets_fluctuation,
                 male_ratio: float = default_male_ratio,
                 scale_male: float = default_scale_male,
                 change_user_filter_frame: int = default_none,
                 change_user_filter_poly: int = default_none,
                 dataset_scaling: int = default_dataset_scaling,
                 number_nan_parts: int = default_number_nan_parts,
                 max_length_nan_parts: int = default_max_lengths_nan_parts):
        """ Artifical Dataset

        :param seed:
        :param number_users:
        :param user_limit:
        :param sample_rate:
        :param number_cases:
        :param training_cases:
        :param testing_cases:
        :param length_datasets: sample length of a dataset
        :param length_datasets_fluctuation: number of samples a dataset is possibly longer
        :param male_ratio:
        :param scale_male:
        :param change_user_filter_frame:
        :param change_user_filter_poly:
        :param dataset_scaling:
        :param number_nan_parts: Create number of blocks which are NaN
        :param max_length_nan_parts: Maximum sample length of NaN Parts
        """

        DatasetBase.__init__(self,
                             seed=seed, user_limit=user_limit,
                             sample_rate=sample_rate,
                             data_kind='angle_deg')

        DatasetFolder.__init__(self,
                               data_folder='[smida]data/fake_path/DATA/',
                               stim_folder='[smida]data/fake_path/STIMULI/',
                               align_folder='[smida]data/fake_path/SMIDA_alignment/')

        self.dataset_name = 'Random Data'
        self.dataset_key = 'demo-data'

        self.number_users = number_users
        self.number_cases = number_cases
        if training_cases is None:
            self.training_cases = ['Case {}'.format(idd) for idd in range(int(np.ceil(self.number_cases / 2)))]
        else:
            self.training_cases = training_cases
        if testing_cases is None:
            self.testing_cases = ['Case {}'.format(idd) for idd in range(int(np.ceil(self.number_cases / 2)),
                                                                         self.number_cases)]
            if not len(self.testing_cases):
                self.testing_cases = training_cases
        else:
            self.testing_cases = testing_cases

        self.male_ratio = male_ratio  # gender ratio m/(m+f)
        self.scale_male = scale_male

        self.change_user_filter_frame = change_user_filter_frame
        self.change_user_filter_poly = change_user_filter_poly

        self.length_datasets = length_datasets
        self.length_datasets_fluctuation = length_datasets_fluctuation
        self.dataset_scaling = dataset_scaling

        self.number_nan_parts = number_nan_parts
        self.length_nan_parts = max_length_nan_parts
        if number_nan_parts > 0:
            # Add NaN Tag to name to make it different
            self.dataset_name += ' NaN({},{})'.format(number_nan_parts, max_length_nan_parts)

        self.user_data = None
        self.user_gender = None
        self.user_filter = None

    def get_screen_params(self):
        return {'pix_per_mm': np.asarray([10, 10]),
                'screen_dist_mm': 500,
                'screen_res': np.asarray([2000, 1500])}

    def get_users(self):
        """ Unique list of users

        :return: list of users
        """

        random.seed(self.seed)

        pot = str(int(np.ceil(np.log10(self.number_users+1))))

        users = [('User {:0' + pot + 'g}').format(idd+1) for idd in range(self.number_users)]

        if self.user_limit is None:
            return users
        else:
            return random.sample(users, self.user_limit)

    def get_cases(self):
        """ List of recordings per user

        :return: list of recordings per user
        """
        # necessary for visualization
        return ['Case {}'.format(idd) for idd in range(self.number_cases)]

    def store_data(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.user_data = [[self.generate_random_sample()
                           for _ in range(self.number_cases)]
                          for _ in range(self.number_users)]

        self.user_gender = ['M' if rand <= self.male_ratio else 'F' for rand in np.random.rand(self.number_users)]

        def get_savgol_parameter():
            frame = (np.random.randint(self.change_user_filter_frame) // 2) * 2 + 1
            poly = np.random.randint(min(self.change_user_filter_poly, frame))
            return frame, poly

        if self.change_user_filter_frame:
            self.user_filter = [get_savgol_parameter() for _ in range(self.number_users)]

    def load_data(self, user, case, convert=True):
        """ Get x, y array with ordinates for recording $case from $user

        :param user: str
            user id to load (e.g. 'ID_001')
        :param case: str ('1' or '2' for Bioeye)
            identifier for filename to divide training and testing
        :param convert: boolean (default=True)
            will convert array to Pixel

        :return: np.ndarray
            2D np.arrays with same length and x, y as components

            pixel:
            x should be 0 to width from left to right
            y should be 0 to height from top to bottom

            angle:
            x should be around 0 from left to right?
            y should be around 0 from top to bottom?
        """

        if self.user_data is None:
            self.store_data()

        # get coordinates
        #   user id starts at 1
        xy = self.user_data[int(user[4:])-1][int(case[4:])]

        if convert:
            screen_parameter = self.get_screen_params()
            xy = convert_angles_deg_to_shifted_pixel_coordinates([(s[0], -s[1]) for s in xy],
                                                                 screen_parameter['pix_per_mm'],
                                                                 screen_parameter['screen_dist_mm'],
                                                                 screen_parameter['screen_res'])

        return xy

    def modify_trajectory(self, trajectory):
        """ Change loaded trajectory depending on Dataset.
        Will be called for every "load_trajectory"
        """

        # smooth data
        trajectory.apply_filter('savgol', frame_size=59, pol_order=0)   # try to create plateaus
        trajectory.apply_filter('savgol', frame_size=29, pol_order=0)   # no fast changes
        trajectory.apply_filter('savgol', frame_size=3, pol_order=0)    # no peaks

        if trajectory.gender == 'M':
            trajectory.scale(self.scale_male)

        if self.user_filter is not None:
            user_id = int(trajectory.user[4:])
            trajectory.apply_filter('savgol', frame_size=self.user_filter[user_id][0],
                                    pol_order=self.user_filter[user_id][1])

        return trajectory

    def get_gender(self, user: str):

        if self.user_gender is None:
            self.store_data()

        #   user id starts at 1
        return self.user_gender[int(user[4:])-1]

    def get_cases_training(self) -> list:
        return self.training_cases

    def get_cases_testing(self) -> list:
        return self.testing_cases

    def get_stimulus(self, case=None):
        """ Relative filename of stimulus

        :param case: case from get_cases
        :return: string
           Relative path to stimulus
        """
        # necessary for visualization
        screen_parameter = self.get_screen_params()

        return [param for param in screen_parameter['screen_res']]

    def generate_random_sample(self) -> list:

        # create continuous xy coordinates around zero
        random_sample = (np.random.rand(
            int(self.length_datasets + np.random.randint(self.length_datasets_fluctuation + 1)), 2)
         - 0.5) * self.dataset_scaling

        # insert nan parts if requested
        for _ in range(self.number_nan_parts):
            random_sample = self.apply_nan_block(random_sample, self.length_nan_parts)

        return random_sample

    @staticmethod
    def apply_nan_block(xy: list, max_length_nan_part: int) -> list:
        """ insert a nan block with max_length_nan_part samples for a 2d coordinates list

        :param xy: 2d coordinates [(x1, y1), (x2, y2), ... ]
        :param max_length_nan_part: maximal number of consecutive samples to insert NaN
        """

        # todo: is this affected by random seed from outside?

        start_sample = np.random.randint(len(xy) - max_length_nan_part)

        xy[start_sample:start_sample + np.random.randint(max_length_nan_part) + 1] = (np.nan, np.nan)

        return xy


# define defaults

user_change_user_filter_frame = 50
user_change_user_filter_poly = 10
gender_scale_male = 1.2


class DemoDatasetUser(DemoDataset):
    """ Dummy dataset for Testing with user specific differences """

    def __init__(self,
                 seed=default_none,
                 number_users: int = default_number_user,
                 user_limit: int = default_none,
                 sample_rate: float = default_sample_rate,
                 number_cases: int = default_number_cases,
                 training_cases: list = default_none,
                 testing_cases: list = default_none,
                 length_datasets: int = default_length_datasets,
                 length_datasets_fluctuation: int = default_length_datasets_fluctuation,
                 male_ratio: float = default_male_ratio,
                 dataset_scaling: int = default_dataset_scaling,
                 number_nan_parts: int = default_number_nan_parts,
                 max_length_nan_parts: int = default_max_lengths_nan_parts):
        super().__init__(seed=seed, number_users=number_users, user_limit=user_limit, sample_rate=sample_rate,
                         number_cases=number_cases, training_cases=training_cases, testing_cases=testing_cases,
                         length_datasets=length_datasets, length_datasets_fluctuation=length_datasets_fluctuation,
                         male_ratio=male_ratio,
                         change_user_filter_frame=user_change_user_filter_frame,
                         change_user_filter_poly=user_change_user_filter_poly,
                         dataset_scaling=dataset_scaling,
                         number_nan_parts=number_nan_parts, max_length_nan_parts=max_length_nan_parts)

        self.dataset_name = self.dataset_name + ' User'
        self.dataset_key = 'demo-user'


class DemoDatasetGender(DemoDataset):
    """ Dummy dataset for Testing with user specific differences """

    def __init__(self,
                 seed=default_none,
                 number_users: int = default_number_user,
                 user_limit: int = default_none,
                 sample_rate: float = default_sample_rate,
                 number_cases: int = default_number_cases,
                 training_cases: list = default_none,
                 testing_cases: list = default_none,
                 length_datasets: int = default_length_datasets,
                 length_datasets_fluctuation: int = default_length_datasets_fluctuation,
                 male_ratio: float = default_male_ratio,
                 dataset_scaling: int = default_dataset_scaling,
                 number_nan_parts: int = default_number_nan_parts,
                 max_length_nan_parts: int = default_max_lengths_nan_parts):
        super().__init__(seed=seed, number_users=number_users, user_limit=user_limit, sample_rate=sample_rate,
                         number_cases=number_cases, training_cases=training_cases, testing_cases=testing_cases,
                         length_datasets=length_datasets, length_datasets_fluctuation=length_datasets_fluctuation,
                         male_ratio=male_ratio,
                         change_user_filter_frame=user_change_user_filter_frame,
                         change_user_filter_poly=user_change_user_filter_poly,
                         dataset_scaling=dataset_scaling,
                         number_nan_parts=number_nan_parts, max_length_nan_parts=max_length_nan_parts)

        self.dataset_name = self.dataset_name + ' Gender'
        self.dataset_key = 'demo-gender'


class DemoDatasetUserGender(DemoDataset):
    """ Dummy dataset for Testing with user specific differences """

    def __init__(self,
                 seed=default_none,
                 number_users: int = default_number_user,
                 user_limit: int = default_none,
                 sample_rate: float = default_sample_rate,
                 number_cases: int = default_number_cases,
                 training_cases: list = default_none,
                 testing_cases: list = default_none,
                 length_datasets: int = default_length_datasets,
                 length_datasets_fluctuation: int = default_length_datasets_fluctuation,
                 male_ratio: float = default_male_ratio,
                 dataset_scaling: int = default_dataset_scaling,
                 number_nan_parts: int = default_number_nan_parts,
                 max_length_nan_parts: int = default_max_lengths_nan_parts):
        super().__init__(seed=seed, number_users=number_users, user_limit=user_limit, sample_rate=sample_rate,
                         number_cases=number_cases, training_cases=training_cases, testing_cases=testing_cases,
                         length_datasets=length_datasets, length_datasets_fluctuation=length_datasets_fluctuation,
                         male_ratio=male_ratio,
                         change_user_filter_frame=user_change_user_filter_frame,
                         change_user_filter_poly=user_change_user_filter_poly,
                         dataset_scaling=dataset_scaling,
                         number_nan_parts=number_nan_parts, max_length_nan_parts=max_length_nan_parts)

        self.dataset_name = self.dataset_name + ' User&Gender'
        self.dataset_key = 'demo-user-gender'
