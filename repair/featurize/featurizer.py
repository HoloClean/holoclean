from abc import ABCMeta, abstractmethod
from multiprocessing import Pool


class Featurizer:
    __metaclass__ = ABCMeta

    def __init__(self, learnable=True, init_weight=1.0):
        self.name = None
        self.setup_done = False
        self.learnable = learnable
        self.init_weight = init_weight

    def setup_featurizer(self, dataset, total_vars, classes, processes=20, batch_size=32):
        self.ds = dataset
        self.total_vars = total_vars
        self.classes = classes
        # only create a pool if processes > 1
        self._pool = Pool(processes) if processes > 1 else None
        self._batch_size = batch_size
        self.setup_done = True
        self.specific_setup()

    @abstractmethod
    def specific_setup(self):
        raise NotImplementedError

    # @abstractmethod
    # def create_tensor(self):
    #     """
    #      This method creates a tensor which has shape
    #      [rv_index, (a torch tensor of dimensions classes x features)]
    #     :return PyTorch Tensor
    #     """
    #     raise NotImplementedError

    @abstractmethod
    def feature_names(self):
        """
        Returns list of human-readable description/names for each feature
        this featurizer produces.
        """
        raise NotImplementedError
    def num_features(self):
        return len(self.feature_names())

    def _apply_func(self, func, collection):
        if self._pool is None:
            return list(map(func, collection))
        return self._pool.map(func, collection, min(self._batch_size, len(collection)))
