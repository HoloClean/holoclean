import pandas as pd

from dataset.table import Table, Source
from .detector import Detector


class ErrorsLoaderDetector(Detector):
    """
    Detector that loads a table of constant errors with the columns:
        id_col: entity ID
        attr_col: attribute in violation
        in the format id_col, attr_col
    Can load these erros from a csv file, a relational table, or a pandas 
    dataframe with the same format.
    """
    def __init__(self, fpath=None, df=None,
                 db_engine=None, table_name=None, schema_name=None,
                 id_col="_tid_", attr_col="attribute", 
                 name="ErrorLoaderDetector"):
        """
        :param fpath: (str) Path to source csv file to load errors
        :param df: (DataFrame) datarame containing the errors
        :param db_engine: (DBEngine) Database engine object
        :param table_name: (str) Relational table considered for loading errors
        :param schema_name: (str) Schema in which :param table_name: exists
        :param id_col: (str) ID column name
        :param attr_col: (str) Attribute column name
        :param name: (str) name of the detector

        To load from csv file, :param fpath: must be specified.
        To load from a relational table, :param db_engine:, and 
        :param table_name: must be specified, optionally specifying :param schema_name:.
        """
        super(ErrorsLoaderDetector, self).__init__(name)
        src = None
        dataset_name = None
        if fpath is not None:
            dataset_name = "errors_file"
            src = Source.FILE
        elif df is not None:
            dataset_name = "errors_df"
            src = Source.DF
        elif (db_engine is not None) and (table_name is not None):
            dataset_name = table_name
            src = Source.DB
        else:
            raise Exception("ERROR while intializing ErrorsLoaderDetector. Please provide (<fpath>), (<db_engine> and <table_name>), OR <df>")

        self.errors_table = Table(dataset_name, src, 
                                  exclude_attr_cols=[attr_col],
                                  fpath=fpath, df=df,
                                  schema_name=schema_name, db_engine=db_engine)
                                
        expected_schema = [id_col, attr_col]
        if list(self.errors_table.df.columns) != expected_schema:
            raise Exception("ERROR while intializing ErrorsLoaderDetector: The loaded errors table does not match the expected schema of {}".format(expected_schema))
        
        self.errors_table.df = self.errors_table.df.astype({
            id_col: int,
            attr_col: str
        })


    def setup(self, dataset=None, env=None):
        self.ds = dataset
        self.env = env

    def detect_noisy_cells(self):
        """
        Returns a pandas.DataFrame containing loaded errors from a source.

        :return: pandas.DataFrame with columns:
            id_col: entity ID
            attr_col: attribute in violation
        """
        return self.errors_table.df
