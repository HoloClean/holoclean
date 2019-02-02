from functools import partial

import torch
import Levenshtein

from dataset import AuxTables
from .featurizer import Featurizer


class InitSimFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'InitSimFeaturizer'
        self.all_attrs = self.ds.get_attributes()
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)
        self.featurization_query_results = self._get_featurization_query_results()

    # def create_tensor(self):
    #
    #     map_input = []
    #     for res in results:
    #         map_input.append((res[0], self.attr_to_idx[res[1]],res[2]))
    #     tensors = self._apply_func(partial(gen_feat_tensor, classes=self.classes, total_attrs=self.total_attrs), map_input)
    #     combined = torch.cat(tensors)
    #     return combined

    def gen_feat_tensor(self, vid):
        assert(self.featurization_query_results[vid][0] == vid)
        vid = int(input[0])
        attr_idx = self.attr_to_idx[input[1]]
        init_value = input[2]
        # TODO: To add more similarity metrics increase the last dimension of tensor.
        tensor = torch.zeros(self.classes, self.total_attrs)
        # Changed
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
