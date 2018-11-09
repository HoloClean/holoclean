from abc import ABCMeta, abstractmethod
from torch.multiprocessing import Pool

class Featurizer:
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name
        self.setup_done = False

    def setup_featurizer(self, dataset, total_vars, classes, processes=20):
        self.ds = dataset
        self.total_vars = total_vars
        self.classes = classes
        self.pool = Pool(processes)
        self.setup_done = True
        self.specific_setup()

    @abstractmethod
    def specific_setup(self):
        raise NotImplementedError

    @abstractmethod
    def create_tensor(self):
        """
         This method creates a tensor which has shape
         [rv_index, (a torch tensor of dimensions classes x features)]
        :return PyTorch Tensor
        """
        raise NotImplementedError
