from abc import ABCMeta, abstractmethod


class Detector:
    """
    This class is an abstract class for general error detection,
     it requires for every sub-class to implement the
    setup and detect_noisy_cells method
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):
        """
        Construct error detection object
        
        :param name: The name of the error detector
        """
        self.name = name
        self.ds = None

    @abstractmethod
    def setup(self, dataset, env):
        raise NotImplementedError

    @abstractmethod
    def detect_noisy_cells(self):
        """
        This method creates a dataframe which has the information
        (tuple index,attribute) for the dk_cells

        :return dataframe  for the dk_cell
        """
        raise NotImplementedError