from functools import partial
import torch
import Levenshtein

from ...dataset import AuxTables
from .featurizer import Featurizer

def gen_feat_tensor(input, classes, total_attrs):
    vid = int(input[0])
    attr_idx = input[1]
    init_value = input[2]
    # TODO: To add more similarity metrics increase the last dimension of tensor.
    tensor = torch.zeros(1, classes, total_attrs)
    domain = input[2].split('|||')
    for idx, val in enumerate(domain):
        if val == init_value:
            sim = -1.0
        else:
            sim = 2*Levenshtein.ratio(val, init_value) - 1
        tensor[0][idx][attr_idx] = sim
    return tensor

class InitSimFeaturizer(Featurizer):
    def __init__(self, name='InitSimFeaturizer'):
        super(InitSimFeaturizer, self).__init__(name)

    def specific_setup(self):
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)

    def create_tensor(self):
        query = 'SELECT _vid_, attribute, init_value, domain FROM %s ORDER BY _vid_'%AuxTables.cell_domain.name
        results = self.ds.engine.execute_query(query)
        map_input = []
        for res in results:
            map_input.append((res[0],self.attr_to_idx[res[1]],res[2]))
        tensors = self.pool.map(partial(gen_feat_tensor, classes=self.classes, total_attrs=self.total_attrs), map_input)
        combined = torch.cat(tensors)
        return combined
