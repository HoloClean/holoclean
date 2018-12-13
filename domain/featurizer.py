from abc import ABCMeta, abstractmethod

class Estimator:
    """
    Estimator is an abstract class for posterior estimators that estimate
    the posterior of p(value | other values) for the purpose of domain generation
    and weak labelling.
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_df):
        """
        :param data_df: (pandas.DataFrame) raw/initial data to use to infer values
        """
        self.data_df = data_df
        # TODO: make a copy of the data in Postgres

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, row):
        raise NotImplementedError
