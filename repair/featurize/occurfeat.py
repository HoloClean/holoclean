import pandas as pd
import torch
from tqdm import tqdm

from .featurizer import Featurizer
from dataset import AuxTables


class OccurFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'OccurFeaturizer'
        if not self.setup_done:
            raise Exception('Featurizer %s is not properly setup.'%self.name)
        self.all_attrs = self.ds.get_attributes()
        self.attrs_number = len(self.ds.attr_to_idx)
        self.current_values_dict = {}
        self.total = None
        self.single_stats = None
        self.pair_stats = None
        self.setup_stats()

    def setup_stats(self):
        """
        Memoize single (frequency of attribute-value) and
        pairwise stats (frequency of attr1-value1-attr2-value2)
        for the current values from loaded dataset.

        self.single_stats is a dict { attribute -> { value -> count } }.
        self.pair_stats is a dict { attr1 -> { attr2 -> { val1 -> {val2 -> co-occur frequency } } } }.
        """
        # current_values_dict is a Dictionary mapping TID -> { attribute -> current value }
        self.current_values_dict = {}

        for (tid, attr, cur_val) in self.ds.get_aux_table(AuxTables.cell_domain).df[['_tid_', 'attribute', 'current_value']].to_records(index=False):
            self.current_values_dict[tid] = self.current_values_dict.get(tid, {})
            self.current_values_dict[tid][attr] = cur_val

        # frequency and co-occurrence frequencies
        total, single_stats, pair_stats = self.ds.get_statistics()
        self.total = float(total)
        self.single_stats = single_stats
        self.pair_stats = pair_stats

    def create_tensor(self):
        """
        For each unique VID (cell) returns the co-occurrence probability between
        each possible domain value for this VID and the current value for the
        corresponding entity/tuple of this cell.

        :return: Torch.Tensor of shape (# of VIDs) X (max domain) X (# of attributes)
            where tensor[i][j][k] contains the co-occur probability between the j-th domain value
            of the i-th random variable (VID) and the current value of the k-th
            attribute for the corresponding entity.
        """
        # Iterate over tuples in domain
        tensors = []
        # Set tuple_id index on raw_data
        t = self.ds.get_aux_table(AuxTables.cell_domain)
        sorted_domain = t.df.reset_index().sort_values(by=['_vid_'])[['_tid_','attribute','_vid_','domain']]
        records = sorted_domain.to_records()
        for row in tqdm(list(records)):
            #Get current values for this TID
            tid = row['_tid_']
            current_tuple = self.current_values_dict[tid]
            feat_tensor = self.gen_feat_tensor(row, current_tuple)
            tensors.append(feat_tensor)
        combined = torch.cat(tensors)
        return combined

    def gen_feat_tensor(self, row, current_tuple):
        """
        For a given cell, we calculate the co-occurence probability of all domain values
        for row['attribute'] with the row's current value in every co-attributes.

        That is for a domain value 'd' and current co-attribute value 'c' we have
            P(d | c)   P(d, c) / P(c)

        where P(d,c) is the empirical co-occurrence frequency of 'd' and 'c' and P(c)
        is the frequency of 'c'.
        """
        # tensor is a (1 X domain size X # of attributes) pytorch.Tensor
        # tensor[0][domain_idx][rv_idx] contains the co-occurrence probability
        # between the current attribute (row['attribute']) and the domain values
        # a possible domain value for this entity
        tensor = torch.zeros(1, self.classes, self.attrs_number)
        rv_attr = row['attribute']
        # Domain value --> index mapping
        domain = row['domain'].split('|||')
        rv_domain_idx = {val: idx for idx, val in enumerate(domain)}

        # Iterate through every attribute (and current value for that
        # attribute) and set the co-occurrence probability for every
        # domain value for our current row['attribute'].
        for attr in self.all_attrs:
            # Skip pairwise with current attribute or NULL value
            # 'attr' may not be in 'current_tuple' since we do not store
            # attributes with all NULL values in our current values
            if attr == rv_attr or pd.isnull(current_tuple.get(attr, None)):
                continue

            attr_idx = self.ds.attr_to_idx[attr]
            val = current_tuple[attr]
            attr_freq = float(self.single_stats[attr][val])
            # Get topK values
            if val not in self.pair_stats[attr][rv_attr]:
                if not pd.isnull(current_tuple[rv_attr]):
                    raise Exception('Something is wrong with the pairwise statistics. <{val}> should be present in dictionary.'.format(val))
            else:
                # dict of { val -> co-occur count }
                all_vals = self.pair_stats[attr][rv_attr][val]
                if len(all_vals) <= len(rv_domain_idx):
                    candidates = list(all_vals.keys())
                else:
                    candidates = domain

                # iterate through all possible domain values of row['attribute']
                for rv_val in candidates:
                    cooccur_freq = float(all_vals.get(rv_val,0.0))
                    prob = cooccur_freq/attr_freq
                    if rv_val in rv_domain_idx:
                        tensor[0][rv_domain_idx[rv_val]][attr_idx] = prob
        return tensor
