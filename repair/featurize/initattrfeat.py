from functools import partial

import torch

from dataset import AuxTables
from .featurizer import Featurizer


def gen_feat_tensor(input, classes, total_attrs):
    vid = int(input[0])
    attr_idx = input[1]
    init_idx = int(input[2])
    tensor = -1 * torch.ones(1,classes,total_attrs)
    tensor[0][init_idx][attr_idx] = 1.0
    return tensor


class InitAttrFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'InitAttrFeaturizer'
        self.all_attrs = self.ds.get_attributes()
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)

    def create_tensor(self):
        query = 'SELECT _vid_, attribute, init_index FROM %s ORDER BY _vid_'%AuxTables.cell_domain.name
        results = self.ds.engine.execute_query(query)
        map_input = []
        for res in results:
            map_input.append((res[0], self.attr_to_idx[res[1]], res[2]))
        tensors = self._apply_func(partial(gen_feat_tensor, classes=self.classes, total_attrs=self.total_attrs), map_input)
        combined = torch.cat(tensors)
        return combined

    def feature_names(self):
        return self.all_attrs
