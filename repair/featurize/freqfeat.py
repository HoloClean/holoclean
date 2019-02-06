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
        # List[tuple(vid, attribute, '|||' separated domain)] sorted by vid.
        self.featurization_query_results = self._get_featurization_query_results()

    def gen_feat_tensor(self, vid):
        assert(self.featurization_query_results[vid][0] == vid)
        input = self.featurization_query_results[vid]
        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        attr_idx = self.ds.attr_to_idx[attribute]
        tensor = torch.zeros(self.classes, self.attrs_number)
        for idx, val in enumerate(domain):
            prob = float(self.single_stats[attribute][val])/float(self.total)
            tensor[idx][attr_idx] = prob
        return tensor

    def _get_featurization_query_results(self):
        query = 'SELECT _vid_, attribute, domain FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
        return self.ds.engine.execute_query(query)

    def feature_names(self):
        return self.all_attrs
