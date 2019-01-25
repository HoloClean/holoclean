import pandas as pd
from .detector import Detector
from dataset import Table

class FixedDetector(Detector):
    def __init__(self, name, src, fpath=None, 
                df=None, db_conn=None, table_query=None, db_engine=None):
        super(FixedDetector, self).__init__(name)
        self._errors_table = Table(name, src, na_values=None, 
            exclude_attr_cols=[], fpath=fpath, df=df, db_conn=db_conn, 
            table_query=table_query, db_engine=db_engine)
        if set(list(self._errors_table.df)) != set(['_tid_', 'attribute']):
            raise Exception('Invalid FixedDetector table for %s' % name)

    def setup(self, dataset=None, env=None):
        self.ds = dataset
        self.env = env

    def detect_noisy_cells(self):
        """
        Returns a pandas.DataFrame containing fixed violations 
        loaded from self._errors_table

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute in violation
        """
        return self._errors_table.df

