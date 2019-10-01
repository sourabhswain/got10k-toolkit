import os

from got10k.datasets.davis import DAVIS
from got10k.experiments import ExperimentOTB


class ExperimentDAVISLike(ExperimentOTB):
    def __init__(self):
        self.use_confs = True

    def _record(self, record_file, boxes, times, confs):
        super()._record(record_file, boxes, times)
        # convert confs to string
        lines = ['%.4f' % c for c in confs]
        lines[0] = '99999.99'
        conf_file = record_file.replace(".txt", "_confidence.value")
        with open(conf_file, 'w') as f:
            f.write(str.join('\n', lines))

    def report(self, tracker_names, plot_curves=True):
        pass  # not implemented (yet) ...


class ExperimentDAVIS(ExperimentDAVISLike):
    def __init__(self, root_dir, result_dir='results', report_dir='reports', start_idx=0, end_idx=None,
                 version="2017_val"):
        self.dataset = DAVIS(root_dir, version)
        self.result_dir = os.path.join(result_dir, 'DAVIS' + str(version))
        self.report_dir = os.path.join(report_dir, 'DAVIS' + str(version))
        self.start_idx = start_idx
        self.end_idx = end_idx
        super().__init__()
