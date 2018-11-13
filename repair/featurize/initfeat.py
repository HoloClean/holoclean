from functools import partial
import torch

from dataset import AuxTables
from .featurizer import Featurizer


def gen_feat_tensor(input, classes):
    vid = int(input[0])
    init_idx = int(input[1])
    tensor = -1.0*torch.ones(1,classes,1)
    tensor[0][init_idx][0] = 1.0
    return tensor


class InitFeaturizer(Featurizer):
    def __init__(self, name='InitFeaturizer'):
        super(InitFeaturizer, self).__init__(name)

    def specific_setup(self):
        pass

    def create_tensor(self):
        query = 'SELECT _vid_, init_index FROM %s ORDER BY _vid_'%AuxTables.cell_domain.name
        results = self.ds.engine.execute_query(query)
        tensors = self.pool.map(partial(gen_feat_tensor, classes=self.classes), results)
        combined = torch.cat(tensors)
        return combined
