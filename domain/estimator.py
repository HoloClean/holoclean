from abc import ABCMeta, abstractmethod

class Estimator:
    """
    Estimator is an abstract class for posterior estimators that estimate
    the posterior of p(value | other values) for the purpose of domain generation
    and weak labelling.
    """
    __metaclass__ = ABCMeta

    def __init__(self, dataset, pruned_domain, active_attrs):
        """
        :param dataset: (Dataset)
        :param pruned_domain: (dict[dict[list[str]]]) dictionary that maps
        :param active_attrs: (list[str]) list of attributes with random variables (for training samples)
        _tid_ (str) --> attr (str) --> (pruned) domain values (list[str]).
        """
        self.ds = dataset
        self.dom = pruned_domain
        self.attrs = self.ds.get_attributes()
        self.active_attrs = active_attrs

    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_pp(self, row, attr, values):
        """
        :param row: (namedtuple) current values of the target row.
        :param attr: (str) attribute of row (i.e. cell) to generate posteriors for.
        :param values: (list[str]) list of values (for this attr) to generate posteriors for.

        :return: list of tuples (value, proba) for each value in :param values:
        """
        raise NotImplementedError
