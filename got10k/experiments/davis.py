import os

from got10k.datasets.davis import DAVIS
from got10k.experiments import ExperimentOTB


class ExperimentDAVIS(ExperimentOTB):
    def __init__(self, root_dir, result_dir='results', report_dir='reports', start_idx=0, end_idx=None, version="2017"):
        self.dataset = DAVIS(root_dir, version)
        self.result_dir = os.path.join(result_dir, 'OTB' + str(version))
        self.report_dir = os.path.join(report_dir, 'OTB' + str(version))
        self.start_idx = start_idx
        self.end_idx = end_idx

    def report(self, tracker_names, plot_curves=True):
        pass  # not implemented (yet) ...
