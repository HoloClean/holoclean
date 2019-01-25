import pandas as pd

from .detector import Detector


class ErrorsLoaderDetector(Detector):
    """
    Detector that loads a table of constant errors from csv.
    """
    def __init__(self, name, fpath):
        """
        :param name: (str) name of the detector
        :param fpath: (str) path to csv file to load errors from.
        """
        super(ErrorsLoaderDetector, self).__init__(name)
        self.errors_df = pd.read_csv(fpath, dtype=str, encoding='utf-8')
        if list(self.errors_df) != ['_tid_', 'attribute']:
            raise Exception('Invalid input file for ErrorsLoaderDetector  %s' % name)

    def setup(self, dataset=None, env=None):
        self.ds = dataset
        self.env = env

    def detect_noisy_cells(self):
        """
        Returns a pandas.DataFrame containing loaded errors from a csv file.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute in violation
        """
        return self.errors_df

