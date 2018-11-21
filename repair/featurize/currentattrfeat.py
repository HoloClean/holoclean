from dataset import AuxTables
from .featurizer import Featurizer
from functools import partial
import torch


def gen_feat_tensor(input, classes, total_attrs):
    vid = int(input[0])
    attr_idx = input[1]
    current_idx = int(input[2])
    tensor = -1.0*torch.ones(1,classes,total_attrs)
    tensor[0][current_idx][attr_idx] = 1.0
    return tensor

class CurrentAttrFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'CurrentAttrFeaturizer'
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)

    def create_tensor(self):
        query = """
        SELECT
            _vid_,
            attribute_idx,
            current_value_idx
        FROM {cell_domain}
        ORDER BY _vid_
        """.format(cell_domain=AuxTables.cell_domain.name)
        results = self.ds.engine.execute_query(query)
        tensors = self.pool.map(partial(gen_feat_tensor, classes=self.classes, total_attrs=self.total_attrs), results)
        combined = torch.cat(tensors)
        return combined
