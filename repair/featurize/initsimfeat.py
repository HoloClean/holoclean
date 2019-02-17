from functools import partial

import torch
import Levenshtein

from dataset import AuxTables
from .featurizer import Featurizer


class InitSimFeaturizer(Featurizer):

    def __init__(self, init_weight=1.0):
        """
        InitSimFeaturizer cannot be learnable.

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
        self.name = 'InitSimFeaturizer'
        self.all_attrs = self.ds.get_attributes()
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)
        # List[tuple(vid, attribute, init_value, domain)].
        self.featurization_query_results = self._get_featurization_query_results()

        # Make sure that the size of 'init_weight' equals to the number of attributes
        # in the dataset.
        if isinstance(self.init_weight, torch.FloatTensor):
            if self.init_weight.shape[0] != len(self.all_attrs):
                raise ValueError("The size of init_weight for InitSimFeaturizer %d does not match the number of attributes %d." % (self.init_weight.shape[0], len(self.all_attrs)))

    def gen_feat_tensor(self, vid):
        assert(self.featurization_query_results[vid][0] == vid)
        input = self.featurization_query_results[vid]
        vid = int(input[0])
        attr_idx = self.attr_to_idx[input[1]]
        init_value = input[2]
        # TODO: To add more similarity metrics increase the last dimension of tensor.
        tensor = torch.zeros(self.classes, self.total_attrs)
        domain = input[3].split('|||')
        for idx, val in enumerate(domain):
            if val == init_value:
                sim = -1.0
            else:
                sim = 2*Levenshtein.ratio(val, init_value) - 1
            tensor[idx][attr_idx] = sim
        return tensor

    def _get_featurization_query_results(self):
        query = 'SELECT _vid_, attribute, init_value, domain FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
        return self.ds.engine.execute_query(query)


    def feature_names(self):
        return self.all_attrs
