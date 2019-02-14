from functools import partial

import torch

from dataset import AuxTables
from .featurizer import Featurizer


def gen_feat_tensor(input, classes, total_attrs):
    vid = int(input[0])
    attr_idx = input[1]
    init_idx = int(input[2])
    tensor = -1 * torch.ones(1, classes, total_attrs)
    tensor[0][init_idx][attr_idx] = 1.0
    return tensor


class InitAttrFeaturizer(Featurizer):

    def __init__(self, init_weight=1.0):
        """
        InitAttrFeaturizer cannot be learnable.

        :param init_weight: (float or list of floats) a fixed weight for all attributes
                            or a list of floats that represent the weights of attributes
                            in the same order in the dataset.
        """
        if isinstance(init_weight, list):
            # If init_weight is a list, we convert to a tensor to be correctly
            # initialized in the TiedLinear model initialization, where init_weight
            # is multiplied by a tensor of ones for initialization.
            init_weight = torch.FloatTensor(init_weight)

        Featurizer.__init__(self, learnable=False, init_weight=init_weight)

    def specific_setup(self):
        self.name = 'InitAttrFeaturizer'
        self.all_attrs = self.ds.get_attributes()
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)
        # Make sure that the size of 'init_weight' equals to the number of attributes
        # in the dataset.
        if isinstance(self.init_weight, torch.FloatTensor):
            if self.init_weight.shape[0] != len(self.all_attrs):
                raise ValueError("The size of init_weight for InitAttrFeaturizer %d does not match the number of attributes %d." %  (self.init_weight.shape[0], len(self.all_attrs)))

    def create_tensor(self):
        query = 'SELECT _vid_, attribute, init_index FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
        results = self.ds.engine.execute_query(query)
        map_input = []
        for res in results:
            map_input.append((res[0], self.attr_to_idx[res[1]], res[2]))
        tensors = self._apply_func(partial(gen_feat_tensor, classes=self.classes, total_attrs=self.total_attrs), map_input)
        combined = torch.cat(tensors)
        return combined

    def feature_names(self):
        return self.all_attrs
