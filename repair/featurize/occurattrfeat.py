import logging

import pandas as pd
import torch
from tqdm import tqdm

from .featurizer import Featurizer
from dataset import AuxTables


class OccurAttrFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'OccurAttrFeaturizer'
        if not self.setup_done:
            raise Exception('Featurizer %s is not properly setup.'%self.name)
        self.all_attrs = self.ds.get_attributes()
        self.attrs_number = len(self.ds.attr_to_idx)
        self.raw_data_dict = {}
        self.total = None
        self.single_stats = None
        self.pair_stats = None
        self.setup_stats()

    def setup_stats(self):
        self.raw_data_dict = self.ds.raw_data.df.set_index('_tid_').to_dict('index')
        total, single_stats, pair_stats = self.ds.get_statistics()
        self.total = float(total)
        self.single_stats = single_stats
        self.pair_stats = pair_stats

    def create_tensor(self):
        # Iterate over tuples in domain
        tensors = []
        # Set tuple_id index on raw_data
        t = self.ds.aux_table[AuxTables.cell_domain]
        sorted_domain = t.df.reset_index().sort_values(by=['_vid_'])[['_tid_','attribute','_vid_','domain']]
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
        tensor = torch.zeros(1, self.classes, self.attrs_number*self.attrs_number)
        rv_attr = row['attribute']
        domain = row['domain'].split('|||')
        rv_domain_idx = {val: idx for idx, val in enumerate(domain)}
        rv_attr_idx = self.ds.attr_to_idx[rv_attr]
        for attr in self.all_attrs:
            if attr != rv_attr and (not pd.isnull(tuple[attr])):
                attr_idx = self.ds.attr_to_idx[attr]
                val = tuple[attr]
                count1 = float(self.single_stats[attr][val])
                # Get topK values
                if val not in self.pair_stats[attr][rv_attr]:
                    if not pd.isnull(tuple[rv_attr]):
                        logging.error('Cannot find attribute: %s with value %s in pair-wise statistics' % (attr, val))
                        raise Exception('Something is wrong with the pairwise statistics. <Val> should be present in dictionary.')
                else:
                    all_vals = self.pair_stats[attr][rv_attr][val]
                    if len(all_vals) <= len(rv_domain_idx):
                        candidates = all_vals
                    else:
                        candidates = domain
                    for rv_val in candidates:
                        count2 = float(all_vals.get(rv_val,0.0))
                        prob = count2/count1
                        if rv_val in rv_domain_idx:
                            index = rv_attr_idx * self.attrs_number + attr_idx
                            tensor[0][rv_domain_idx[rv_val]][index] = prob
        return tensor

    def feature_names(self):
        return ["{} X {}".format(attr1, attr2) for attr1 in self.all_attrs for attr2 in self.all_attrs]
