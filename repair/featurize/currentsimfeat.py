from functools import partial
import torch
import Levenshtein

from dataset import AuxTables
from .featurizer import Featurizer


def gen_feat_tensor(input, classes, total_attrs):
    vid = int(input[0])
    attr_idx = input[1]
    current_value = input[2]
    domain = input[3].split('|||')
    # TODO: To add more similarity metrics increase the last dimension of tensor.
    tensor = torch.zeros(1, classes, total_attrs)
    for idx, val in enumerate(domain):
        if val == current_value:
            sim = -1.0
        else:
            sim = 2*Levenshtein.ratio(val, current_value) - 1
        tensor[0][idx][attr_idx] = sim
    return tensor


class CurrentSimFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'CurrentSimFeaturizer'
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)

    def create_tensor(self):
        query = """
        SELECT
            _vid_,
            attribute_idx,
            current_value,
            domain
        FROM {cell_domain}
        ORDER BY _vid_
        """.format(cell_domain=AuxTables.cell_domain.name)
        results = self.ds.engine.execute_query(query)
        # Map attribute to their attribute indexes
        tensors = self.pool.map(partial(gen_feat_tensor, classes=self.classes, total_attrs=self.total_attrs), results)
        combined = torch.cat(tensors)
        return combined
