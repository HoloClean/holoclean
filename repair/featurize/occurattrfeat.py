import torch
from tqdm import tqdm

from .featurizer import Featurizer
from dataset import AuxTables
from utils import NULL_REPR, NA_COOCCUR_FV


class OccurAttrFeaturizer(Featurizer):
    """
    Generates co-occurrence features for
        RV ATTR | COND ATTR
    where RV ATTR are all attributes that are "active" i.e. those that
    we want to learn for (in domain) and COND ATTR are all attributes in the
    dataset.
    """
    def specific_setup(self):
        self.name = 'OccurAttrFeaturizer'
        if not self.setup_done:
            raise Exception('Featurizer {} is not properly setup.'.format(self.name))
        self.active_attrs = self.ds.get_active_attributes()
        self.active_attr_to_idx = {attr: idx for idx, attr in enumerate(self.active_attrs)}
        self.all_attrs = self.ds.get_attributes()
        self.raw_data_dict = {}
        self.total = None
        self.single_stats = None
        self.pair_stats = None
        self.setup_stats()

    def setup_stats(self):
        raw_df = self.ds.get_quantized_data() if self.ds.do_quantization else self.ds.get_raw_data()
        self.raw_data_dict = raw_df.set_index('_tid_').to_dict('index')
        total, single_stats, pair_stats = self.ds.get_statistics()
        self.total = float(total)
        self.single_stats = single_stats
        self.pair_stats = pair_stats

    def create_tensor(self):
        # Iterate over tuples in domain
        tensors = []
        # Set tuple_id index on raw_data
        t = self.ds.aux_table[AuxTables.cell_domain]
        sorted_domain = t.df.reset_index().sort_values(by=['_vid_'])[['_tid_', 'attribute', '_vid_', 'domain']]
        records = sorted_domain.to_records()
        for row in tqdm(list(records)):
            # Get tuple from raw_dataset.
            tid = row['_tid_']
            tuple = self.raw_data_dict[tid]
            feat_tensor = self.gen_feat_tensor(row, tuple)
            tensors.append(feat_tensor)
        combined = torch.cat(tensors)
        return combined

    def gen_feat_tensor(self, row, tuple):
        # We only generate co-occurrence features for cells that are 'active' or
        # in our domain. Each active attribute will have co-occurrences with
        # all other attributes, hence the following product.
        tensor = torch.zeros(1, self.classes, len(self.active_attrs) * len(self.all_attrs))
        rv_attr = row['attribute']
        domain = row['domain'].split('|||')
        rv_domain_idx = {val: idx for idx, val in enumerate(domain)}

        # We should not have any NULLs in our domain.
        assert NULL_REPR not in rv_domain_idx
        rv_attr_idx = self.active_attr_to_idx[rv_attr]
        # Iterate through all /given/ or /conditional/ attributes
        for attr in self.all_attrs:
            val = tuple[attr]

            # Ignore co-occurrences of same attribute or with null values.
            # It's possible a value is not in pair_stats if it only co-occurred
            # with NULL values.
            if attr == rv_attr \
                    or val == NULL_REPR \
                    or val not in self.pair_stats[attr][rv_attr]:
                continue
            attr_idx = self.ds.attr_to_idx[attr]
            count1 = float(self.single_stats[attr][val])
            all_vals = self.pair_stats[attr][rv_attr][val]
            for rv_val in domain:
                count2 = float(all_vals.get(rv_val, 0.0))
                prob = count2 / count1
                if rv_val in rv_domain_idx:
                    index = rv_attr_idx * len(self.active_attrs) + attr_idx
                    tensor[0][rv_domain_idx[rv_val]][index] = prob
        return tensor

    def feature_names(self):
        return ["{} | {}".format(attr1, attr2) for attr1 in self.active_attrs for attr2 in self.all_attrs]
