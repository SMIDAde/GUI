from enum import Enum, auto

import numpy as np
from os.path import basename, join as pjoin
import os
import random
from schau_mir_in_die_augen.process.conversion import convert_angles_deg_to_shifted_pixel_coordinates
from schau_mir_in_die_augen.datasets.DatasetBase import DatasetBase
from schau_mir_in_die_augen.datasets.DatasetFolder import DatasetFolder


class BioEye(DatasetBase, DatasetFolder):
    class Subsets(Enum):
        RAN_30min_dv = auto()
        TEX_30min_dv = auto()
        RAN_1year_dv = auto()
        TEX_1year_dv = auto()

    def __init__(self, subset=Subsets.TEX_30min_dv, score_level_eval=False, one_year_train=False, user_limit=None,
                 seed=42, use_valid_data=False):
        """

        :param subset: string
        one of: ran30, ran1y, tex30, tex1y
        :param score_level_eval: bool
        see load_testing: evaluation for 1year like in score level paper
        """
        DatasetBase.__init__(self, seed=seed, user_limit=user_limit,
                             sample_rate=250,
                             data_kind='angle_deg')

        DatasetFolder.__init__(self,
                               data_folder='[smida]data/BioEye2015_DevSets/{}/'.format(subset.name),
                               stim_folder='[smida]data/BioEye2015_DevSets/Visual_Stimuli/',
                               align_folder='[smida]data/BioEye2015_DevSets/SMIDA_alignment/{}'.format(subset.name))

        self.one_year_train = one_year_train
        self.score_level_eval = score_level_eval
        if subset == self.Subsets.RAN_30min_dv:
            self.dataset_name = 'BioEyeRan 30min'
            self.dataset_key = 'bio-ran'
        elif subset == self.Subsets.TEX_30min_dv:
            self.dataset_name = 'BioEyeTex 30min'
            self.dataset_key = 'bio-tex'
        elif subset == self.Subsets.RAN_1year_dv:
            self.dataset_name = 'BioEyeRan 1year'
            self.dataset_key = 'bio-ran1y'
        elif subset == self.Subsets.TEX_1year_dv:
            self.dataset_name = 'BioEyeTex 1year'
            self.dataset_key = 'bio-tex1y'
        else:
            raise Exception('Subset "{}" is unkown!'.format(subset))
        self.subset = subset

        self.testing_id = '3' if '1year' in subset.name else '1'
        self.training_id = '1' if '1year' in subset.name else '2'
        if one_year_train:
            self.training_id = '3'
            self.testing_id = '1'

        # Screen dimensions (w × h): 474 × 297 mm
        self.screen_size_mm = np.asarray([474, 297])
        # Screen resolution (w × h): 1680 × 1050 pixels
        self.screen_res = np.asarray([1680, 1050])
        self.pix_per_mm = self.screen_res / self.screen_size_mm
        # Subject's distance from screen: 550 mm
        self.screen_dist_mm = 550
        self.is_valid = use_valid_data

    def get_screen_params(self):
        return {'pix_per_mm': self.pix_per_mm,
                'screen_dist_mm': self.screen_dist_mm,
                'screen_res': self.screen_res}

    def get_users(self):
        # unique users
        users = list(sorted(set([basename(f.path)[:6]
                                 for f in os.scandir(self.data_folder)
                                 if f.is_file()])))

        # randomly select some users, if there are more than user_limit
        if self.user_limit:
            random.seed(self.seed)
            random.shuffle(users)
            users = users[:self.user_limit]

        return users

    def get_cases(self):
        return [self.training_id, self.testing_id]

    def get_stimulus(self, case="1"):
        if self.subset in {BioEye.Subsets.RAN_30min_dv, BioEye.Subsets.RAN_1year_dv}:
            return os.path.join(self.stim_folder, 'RAN_Stimulus.png')
        else:
            return os.path.join(self.stim_folder, 'TEXT_Stimulus_Session_{}.png'.format(case))

    def load_data(self, user='ID_003', case='1', convert=True):
        """ Get x-org, y-org array with ordinates for recording $case from $user

        :param user: str
            user id to load (e.g. 'ID_001')
        :param case: str ('1' or '2')
            identifier for filename to divide training and testing
        :param convert: boolean (default=True)
            will convert array to Pixel

        :return: np.ndarray
            2D np.arrays with equal length x, y as components
        """
        filename = '{}_{}.txt'.format(user, case)
        xy = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=[1, 2])
        # todo: loading takes very long

        if self.is_valid:
            valid = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=[3])
            xy = xy[valid == 1]

        # y-axis seems to be reverted!
        if convert:
            # convert angles to pixel

            xy = convert_angles_deg_to_shifted_pixel_coordinates([(s[0], -s[1]) for s in xy],
                                                                 self.pix_per_mm, self.screen_dist_mm, self.screen_res)
        else:
            xy = np.asarray([(s[0], -s[1]) for s in xy])

        return xy

    def get_cases_training(self):
        return [self.training_id]

    def get_cases_testing(self):
        # it seems like the a score level authors used both sessions for evaluation - even though the first session is
        # the same as the first from the first recording session
        if self.score_level_eval:
            assert self.testing_id != '1'
            return [self.testing_id,'1']
        return [self.testing_id]