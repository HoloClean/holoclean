from abc import ABCMeta, abstractmethod


class Estimator:
    """
    Estimator is an abstract class for posterior estimators that estimate
    the posterior of p(value | other values) for the purpose of domain generation
    and weak labelling.
    """
    __metaclass__ = ABCMeta

    def __init__(self, env, dataset):
        """
        :param env: (dict) dict containing environment/parameters settings.
        :param dataset: (Dataset)
        """
        self.env = env
        self.ds = dataset
        self.attrs = self.ds.get_attributes()

    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_pp(self, row, attr, values):
        """
        :param row: (namedtuple) current values of the target row.
        :param attr: (str) attribute of row (i.e. cell) to generate posteriors for.
        :param values: (list[str]) list of values (for this attr) to generate posteriors for.

        :return: iterator of tuples (value, proba) for each value in :param values:
        """
        raise NotImplementedError

    @abstractmethod
    def predict_pp_batch(self):
        """
        predict_pp_batch is like predict_pp but with a batch of cells.

        :return: iterator of iterator of tuples (value, proba) (one iterator per cell/row in cell_domain_rows)
        """
        raise NotImplementedError
