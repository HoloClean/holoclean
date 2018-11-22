from dataset import AuxTables
from .featurizer import Featurizer
from functools import partial
import torch


def gen_feat_tensor(input, classes, total_attrs):
    vid = int(input[0])
    attr_idx = input[1]
    init_idx = int(input[2])
    tensor = -1.0*torch.ones(1,classes,total_attrs)
    tensor[0][init_idx][attr_idx] = 1.0
    return tensor

class InitAttFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'InitAttFeaturizer'
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)

    def create_tensor(self):
        query = 'SELECT _vid_, attribute, init_index FROM %s ORDER BY _vid_'%AuxTables.cell_domain.name
        results = self.ds.engine.execute_query(query)
        map_input = []
        for res in results:
            map_input.append((res[0], self.attr_to_idx[res[1]], res[2]))
        tensors = self.pool.map(partial(gen_feat_tensor, classes=self.classes, total_attrs=self.total_attrs), map_input)
        combined = torch.cat(tensors)
        return combined
