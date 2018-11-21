from functools import partial
import torch

from dataset import AuxTables
from .featurizer import Featurizer


def gen_feat_tensor(input, classes):
    vid = int(input[0])
    current_idx = int(input[1])
    tensor = -1.0*torch.ones(1,classes,1)
    tensor[0][current_idx][0] = 1.0
    return tensor


class CurrentFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'CurrentFeaturizer'

    def create_tensor(self):
        query = """
        SELECT
            _vid_,
            current_value_idx
        FROM {cell_domain}
        ORDER BY _vid_
        """.format(cell_domain=AuxTables.cell_domain.name)
        results = self.ds.engine.execute_query(query)
        tensors = self.pool.map(partial(gen_feat_tensor, classes=self.classes), results)
        combined = torch.cat(tensors)
        return combined
