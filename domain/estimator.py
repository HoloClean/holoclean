from abc import ABCMeta, abstractmethod


class Estimator:
    """
    Estimator is an abstract class for posterior estimators that estimate
    the posterior of p(value | other values) for the purpose of domain generation
    and weak labelling.
    """
    __metaclass__ = ABCMeta

    def __init__(self, env, dataset, domain_df):
        """
        :param env: (dict) dict containing environment/parameters settings.
        :param dataset: (Dataset)
        """
        self.env = env
        self.ds = dataset
        self.domain_df = domain_df.sort_values('_vid_')
        self.attrs = self.ds.get_attributes()

    @abstractmethod
    def train(self, num_epochs, batch_size):
        raise NotImplementedError

    @abstractmethod
    def predict_pp_batch(self):
        """
        predict_pp_batch predicts the probabilities for a batch of cells corresponding
        to the cells specified in domain_df.

        :return: iterator of (vid, is categorical (bool), [(value, proba)])
        for each random var in domain_df.
        """
        raise NotImplementedError
