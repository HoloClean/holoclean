from functools import partial

import torch

from dataset import AuxTables
from .featurizer import Featurizer


class InitAttrFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'InitAttrFeaturizer'
        self.all_attrs = self.ds.get_attributes()
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)
        # The query results used to featurize the dataset
        self.featurization_query_results = self._get_featurization_query_results()

    # def create_tensor(self):
    #     map_input = []
    #     for res in results:
    #         map_input.append((res[0], self.attr_to_idx[res[1]], res[2]))
    #     tensors = self._apply_func(partial(gen_feat_tensor, classes=self.classes, total_attrs=self.total_attrs), map_input)
    #     combined = torch.cat(tensors)
    #     return combined

    def _get_featurization_query_results(self):
        query = 'SELECT _vid_, attribute, init_index FROM %s ORDER BY _vid_'%AuxTables.cell_domain.name
        return self.ds.engine.execute_query(query)

    def gen_feat_tensor(self, input, vid):
        assert(self.featurization_query_results[vid][0] == vid)
        input = self.featurization_query_results[vid]
        vid = int(input[0])
        attr_idx = self.attr_to_idex[input[1]]
        init_idx = int(input[2])
        tensor = torch.zeros(self.classes, self.total_attrs)
        tensor[init_idx][attr_idx] = 1.0
        return tensor

    def feature_names(self):
        return self.all_attrs
