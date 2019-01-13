import torch

from dataset import AuxTables
from .featurizer import Featurizer


class FreqFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'FreqFeaturizer'
        self.all_attrs = self.ds.get_attributes()
        self.attrs_number = len(self.ds.attr_to_idx)
        total, single_stats, pair_stats = self.ds.get_statistics()
        self.total = total
        self.single_stats = single_stats

    def gen_feat_tensor(self, input, classes):
        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        attr_idx = self.ds.attr_to_idx[attribute]
        tensor = torch.zeros(1, classes, self.attrs_number)
        for idx, val in enumerate(domain):
            prob = float(self.single_stats[attribute][val])/float(self.total)
            tensor[0][idx][attr_idx] = prob
        return tensor

    def create_tensor(self):
        query = 'SELECT _vid_, attribute, domain FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
        results = self.ds.engine.execute_query(query)
        tensors = [self.gen_feat_tensor(res, self.classes) for res in results]
        combined = torch.cat(tensors)
        return combined

    def feature_names(self):
        return self.all_attrs
