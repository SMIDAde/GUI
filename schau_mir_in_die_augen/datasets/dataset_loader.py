""" Load a list of all datasets

Is used for the visualization class to have a selection.
"""

from schau_mir_in_die_augen.datasets.Bioeye import BioEye
from schau_mir_in_die_augen.datasets.DemoDataset import DemoDataset, DemoDatasetUser

dataset_list = [BioEye(subset=BioEye.Subsets.TEX_30min_dv), BioEye(subset=BioEye.Subsets.TEX_1year_dv),
                BioEye(subset=BioEye.Subsets.RAN_30min_dv), BioEye(subset=BioEye.Subsets.RAN_1year_dv),
                DemoDataset(), DemoDatasetUser()]
