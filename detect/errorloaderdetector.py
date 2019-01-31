import pandas as pd

from .detector import Detector


class ErrorsLoaderDetector(Detector):
    """
    Detector that loads a table of constant errors from csv with the columns:
        _tid_: entity ID
        attribute: attribute in violation
        in the format _tid_, attribute
    """
    def __init__(self, fpath, name='ErrorLoaderDetector'):
        """
        :param name: (str) name of the detector
        :param fpath: (str) path to csv file to load errors from.
        """
        super(ErrorsLoaderDetector, self).__init__(name)
        self.errors_df = pd.read_csv(fpath,
                                     dtype={'_tid_': int, 'attribute': str},
                                     encoding='utf-8')
        if list(self.errors_df) != ['_tid_', 'attribute']:
            raise Exception('Invalid input file for ErrorsLoaderDetector: {}'.format(fpath))

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

