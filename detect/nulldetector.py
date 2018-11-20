import pandas as pd
from .detector import Detector

class NullDetector(Detector):
    def __init__(self, name='NullDetector'):
        super(NullDetector, self).__init__(name)

    def setup(self, dataset, env):
        self.ds = dataset
        self.env = env
        self.df = self.ds.get_raw_data()

    def detect_noisy_cells(self):
        attributes = self.ds.get_attributes()
        errors = []
        for attr in attributes:
            # TODO: test if isnull() resolve null issue
            tmp_df = self.df[self.df[attr].isnull()]['_tid_'].to_frame()
            tmp_df.insert(1, "attribute", attr)
            errors.append(tmp_df)
        errors_df = pd.concat(errors, ignore_index=True).drop_duplicates().reset_index(drop=True)
        return errors_df

