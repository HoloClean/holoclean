from functools import partial

import torch

from dataset import AuxTables
from .featurizer import Featurizer


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
        # List[tuple(vid, attribute, init_index)] sorted by vid.
        self.featurization_query_results = self._get_featurization_query_results()

        # Make sure that the size of 'init_weight' equals to the number of attributes
        # in the dataset.
        if isinstance(self.init_weight, torch.FloatTensor):
            if self.init_weight.shape[0] != len(self.all_attrs):
                raise ValueError("The size of init_weight for InitAttrFeaturizer %d does not match the number of attributes %d." %  (self.init_weight.shape[0], len(self.all_attrs)))

    def _get_featurization_query_results(self):
        query = 'SELECT _vid_, attribute, init_index FROM %s ORDER BY _vid_'%AuxTables.cell_domain.name
        return self.ds.engine.execute_query(query)

    def gen_feat_tensor(self, vid):
        assert(self.featurization_query_results[vid][0] == vid)
        input = self.featurization_query_results[vid]
        vid = int(input[0])
        attr_idx = self.attr_to_idx[input[1]]
        init_idx = int(input[2])
        tensor = -1*torch.ones(self.classes, self.total_attrs)
        tensor[init_idx][attr_idx] = 1.0
        return tensor

    def feature_names(self):
        return self.all_attrs
