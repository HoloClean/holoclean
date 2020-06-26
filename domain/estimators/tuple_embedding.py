import logging
import math
import os
import pickle
import sys

import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import CrossEntropyLoss, Softmax, MSELoss, ReLU
import torch.nn.functional as F
from tqdm import tqdm

from dataset import AuxTables
from ..estimator import Estimator
from utils import NULL_REPR

NONNUMERICS = "[^0-9+-.e]"

def verify_numerical_attr_groups(dataset, numerical_attr_groups):
    """
    Verify numerical attribute groups are disjoint and exist

    Returns a list of the individual attributes.
    """
    numerical_attrs = None
    # Check if numerical attributes exist and are disjoint
    if numerical_attr_groups is not None:
        numerical_attrs = [attr for group in numerical_attr_groups for attr in group]

        if not all(attr in dataset.get_attributes() for attr in numerical_attrs):
            logging.error('all numerical attributes specified %s must exist in dataset: %s',
                    numerical_attr_groups,
                    dataset.get_attributes())
            raise Exception()

        if len(set(numerical_attrs)) < len(numerical_attrs):
            logging.error('all attribute groups specified %s must be disjoint in dataset',
                    numerical_attr_groups)
            raise Exception()

    return numerical_attrs


class LookupDataset(Dataset):
    # Memoizes vectors (e.g. init vector, domain vector, negative indexes)
    # for every sample indexed by idx.
    class MemoizeVec:
        # If other_dims is none, we have a variable size indexing structure
        def __init__(self, n_samples, dtype, *other_dims):
            self._variable = len(other_dims) == 0 or other_dims == (None,)
            if self._variable:
                self._vec = [None for _ in range(n_samples)]
            else:
                self._vec = torch.zeros(n_samples, *other_dims, dtype=dtype)
            self._isset = torch.zeros(n_samples, dtype=torch.uint8)

        def __contains__(self, idx):
            return self._isset[idx] != 0

        # idx can be an integer, slice or a tuple or either
        def __getitem__(self, idx):
            if idx not in self:
                raise IndexError("tried to access un-set value at index %d" % idx)
            return self._vec[idx]

        # idx can be an integer, slice or a tuple or either
        def __setitem__(self, idx, vec):
            self._vec[idx] = vec
            if isinstance(idx, tuple):
                self._isset[idx[0]] = 1
            else:
                self._isset[idx] = 1

    def __init__(self, env, dataset, domain_df,
            numerical_attr_groups, neg_sample, memoize):
        """
        :param dataset: (Dataset) original dataset
        :param domain_df: (DataFrame) dataframe containing VIDs and their
            domain values we want to train on. VIDs not included in this
            dataframe will not be trained on e.g. if you only want to
            sub-select VIDs of certain attributes.
        :param numerical_attr_groups: (list[list[str]]) groups of attributes
            to be treated as d-dimensional numerical values.
        :param neg_sample: (bool) add negative examples for clean cells during training
        :param memoize: (bool) memoize re-lookups on subsequent epochs.
        """
        self.env = env
        self.ds = dataset
        self.memoize = memoize
        self.inference_mode = False

        self._neg_sample = neg_sample

        # Init attrs/vals herein refers to attributes and values embedded to use
        # as context during training.
        # Train attrs/vals herein refers to the attributes/columns and possible values
        # embedded to use as the targets during training.
        # _cat_ refers to categorical attributes and _num_ refers to numerical
        # attributes.
        self._all_attrs = sorted(self.ds.get_attributes())
        self._numerical_attrs = verify_numerical_attr_groups(self.ds,
                numerical_attr_groups) or []
        self._numerical_attr_groups = numerical_attr_groups

        # Attributes to derive context from
        self._init_cat_attrs, self._init_num_attrs = self._split_cat_num_attrs(self._all_attrs)
        self._n_init_cat_attrs, self._n_init_num_attrs = len(self._init_cat_attrs), len(self._init_num_attrs)
        self._n_init_attrs = len(self._all_attrs)
        logging.debug('%s: init categorical attributes: %s',
                type(self).__name__,
                self._init_cat_attrs)
        logging.debug('%s: init numerical attributes: %s',
                type(self).__name__,
                self._init_num_attrs)

        # Attributes to train on (i.e. target columns).
        self._train_attrs = sorted(domain_df['attribute'].unique())
        assert all(attr in self._all_attrs for attr in self._train_attrs)
        self._train_cat_attrs, self._train_num_attrs = self._split_cat_num_attrs(self._train_attrs)
        self._n_train_cat_attrs, self._n_train_num_attrs = len(self._train_cat_attrs), len(self._train_num_attrs)
        logging.debug('%s: train categorical attributes: %s',
                type(self).__name__,
                self._train_cat_attrs)
        logging.debug('%s: train numerical attributes: %s',
                type(self).__name__,
                self._train_num_attrs)

        # Make copy of raw data
        # Quantized data is used for co-occurrence statistics in the last layer
        # for categorical targets.
        self._raw_data = self.ds.get_raw_data().copy()
        # self._qtized_raw_data = self.ds.get_quantized_data() if self.ds.do_quantization else self._raw_data
        # self._qtized_raw_data_dict = self._qtized_raw_data.set_index('_tid_').to_dict('index')

        # Statistics for cooccurrences.
        # _, self._single_stats, self._pair_stats = self.ds.get_statistics()

        self.load_raw_data(self.ds.get_raw_data())

        # Indexes assigned to attributes: FIRST categorical THEN numerical.
        # (this order is important since we shift the numerical idxs).
        self._init_attr_idxs = {attr: idx for idx, attr in enumerate(self._init_cat_attrs + self._init_num_attrs)}
        self._train_attr_idxs = {attr: idx for idx, attr in enumerate(self._train_cat_attrs + self._train_num_attrs)}

        # Assign indices for every categorical value for both init values
        # and training values.
        # Assign index for every unique value-attr (init values, input)
        self._init_val_idxs = {attr: {} for attr in self._init_cat_attrs}
        # Assign index for every unique value-attr (train/possible values, target)
        self._train_val_idxs = {attr: {} for attr in self._train_cat_attrs}

        # Initial categorical values we've seen during training. Otherwise
        # we need to zero out the associated embedding since un-seen initial
        # values will have garbage embeddings.
        self._seen_init_cat_vals = {attr: set() for attr in self._init_cat_attrs}

        # Reserve the 0th index as placeholder for padding in domain_idx and
        # for NULL values.
        cur_init_idx = 1
        for row in self._raw_data.to_records():
            for attr in self._init_cat_attrs:
                if row[attr] in self._init_val_idxs[attr]:
                    continue

                # Use special index 0 for NULL values
                if row[attr] == NULL_REPR:
                    self._init_val_idxs[attr][row[attr]] = 0
                    continue

                # Assign index for init values
                self._init_val_idxs[attr][row[attr]] = cur_init_idx
                cur_init_idx += 1

        # Do the same for the train/target categorical values.
        cur_train_idx = 1
        for row in domain_df.to_records():
            val = row['init_value']
            attr = row['attribute']

            # We handle numerical attrs differently
            if attr in self._train_num_attrs:
                continue

            # Assign index for train/possible values
            if val in self._train_val_idxs[attr]:
                continue

            # Use special index 0 for NULL values
            if val == NULL_REPR:
                self._train_val_idxs[attr][val] = 0
                continue

            # Assign index for train/domain values
            self._train_val_idxs[attr][val] = cur_train_idx
            cur_train_idx += 1

        # Unique train values (their indexes) by attr
        self._train_val_idxs_by_attr = {attr: torch.LongTensor([v for v in self._train_val_idxs[attr].values() if v != 0])
                for attr in self._train_cat_attrs}


        # Maps each numerical attribute to a copy of its group (of other
        # numerical attributes).
        self._train_num_attrs_group = {attr: group.copy() for group in
                self._numerical_attr_groups for attr in group
                if attr in self._train_num_attrs}

        # Number of unique INIT attr-values
        self.n_init_vals = cur_init_idx
        self.n_train_vals = cur_train_idx

        self.load_domain_df(domain_df)

        # Maximum dimension across all numerical attributes.
        self._max_num_dim = max(list(map(len, self._numerical_attr_groups)) or [0])

        self._init_dummies()
        self._init_memoize_vecs()

    def load_raw_data(self, df_raw):
        self._raw_data = df_raw.copy()

        # Keep track of mean + std to un-normalize during prediction
        self._num_attrs_mean = {}
        self._num_attrs_std = {}
        # Mean-0 variance 1 normalize all numerical attributes in the raw data
        for num_attr in self._init_num_attrs:
            temp = self._raw_data[num_attr].copy()
            fil_notnull = temp != NULL_REPR
            self._num_attrs_mean[num_attr] = temp[fil_notnull].astype(np.float).mean(axis=0)
            self._num_attrs_std[num_attr] = temp[fil_notnull].astype(np.float).std(axis=0)
            temp[fil_notnull] = ((temp[fil_notnull].astype(np.float) \
                    - self._num_attrs_mean[num_attr]) \
                    / (self._num_attrs_std[num_attr] or 1.)).astype(str)
            self._raw_data[num_attr] = temp
        # This MUST go after the mean-0 variance 1 normalization above since
        # this is looked up subsequently during training.
        self._raw_data_dict = self._raw_data.set_index('_tid_').to_dict('index')

    def load_domain_df(self, domain_df):
        """
        Loads a new domain_df such that __getitem__ and other methods work with
        the new cells.
        """
        if not np.isin(domain_df['attribute'].unique(), self._train_cat_attrs).all():
            logging.error("not all cat. target attributes of new domain DF has "
                          "been used to train model before."
                          "\nDomain target attrs: %s\nTrained target attrs: %s",
                    sorted(domain_df['attribute'].unique()),
                    self._train_cat_attrs)
            raise Exception()

        self._vid_to_idx = {vid: idx for idx, vid in enumerate(domain_df['_vid_'].values)}
        # LookupDataset always gets its information on the domain_df from this
        # list of records.
        # To modify what stuff the dataset looks up, one needs to modify this
        # along with the set of VIDs passed to DataLoader.
        train_record_cols = ['_vid_', '_tid_', 'attribute',
                                   'weak_label', 'weak_label_idx', 'fixed',
                                   'domain', 'domain_size', 'is_clean']
        self._train_records = domain_df[train_record_cols].to_records()

        # Maximum domain size: we don't use the domain of numerical attributes
        # so we can discard them.
        self.max_cat_domain = domain_df.loc[domain_df['attribute'].isin(self._train_cat_attrs), 'domain_size'].max()
        if pd.isna(self.max_cat_domain):
            self.max_cat_domain = 0

    def _init_dummies(self):
        # Dummy vectors
        self._dummy_domain_mask = torch.zeros(self.max_cat_domain,
                                              dtype=torch.float)
        self._dummy_domain_idxs = torch.zeros(self.max_cat_domain,
                                              dtype=torch.long)
        self._dummy_domain_cooccur = torch.zeros(self.max_cat_domain, self._n_init_attrs,
                                              dtype=torch.float)
        self._dummy_target_numvals = torch.zeros(self._max_num_dim,
                                                 dtype=torch.float)
        self._dummy_cat_target = torch.LongTensor([-1])

    def _init_memoize_vecs(self):
        # Memoize certain lookups.
        if not self.memoize:
            return
        self._domain_idxs = self.MemoizeVec(len(self), torch.int64, self.max_cat_domain)
        self._init_cat_idxs = self.MemoizeVec(len(self), torch.int64, self._n_init_cat_attrs)
        self._neg_idxs = self.MemoizeVec(len(self), None, None)

        if self._max_num_dim > 0:
            self._target_numvals = self.MemoizeVec(len(self), torch.float32, self._max_num_dim)
            self._init_numvals = self.MemoizeVec(len(self), torch.float32, self._n_init_num_attrs)
            self._init_nummask = self.MemoizeVec(len(self), torch.float32, self._n_init_num_attrs)

    def _split_cat_num_attrs(self, attrs):
        """
        Splits a list of attributes into their categorical and numerical groupings.
        """
        cat_attrs = [attr for attr in attrs if attr not in self._numerical_attrs]
        num_attrs = [attr for attr in attrs if attr in self._numerical_attrs]
        return cat_attrs, num_attrs

    def __len__(self):
        return len(self._train_records)

    def _get_neg_dom_idxs(self, idx):
        if not self.memoize or idx not in self._neg_idxs:
            cur = self._train_records[idx]
            assert cur['attribute'] in self._train_cat_attrs

            # Value indices that are not in the domain
            neg_idxs = torch.LongTensor(np.setdiff1d(self._train_val_idxs_by_attr[cur['attribute']],
                    self._get_domain_idxs(idx),
                    assume_unique=True))
            if not self.memoize:
                return neg_idxs
            self._neg_idxs[idx] = neg_idxs
        return self._neg_idxs[idx]

    def _get_domain_idxs(self, idx):
        """
        Get domain indexes for categorical cells.
        """
        if not self.memoize or idx not in self._domain_idxs:
            cur = self._train_records[idx]
            assert cur['attribute'] in self._train_cat_attrs

            # Domain values and their indexes (softmax indexes)
            domain_idxs = torch.zeros(self.max_cat_domain, dtype=torch.long)

            domain_idxs[:cur['domain_size']] = torch.LongTensor([self._train_val_idxs[cur['attribute']][val]
                    for val in cur['domain']])

            if not self.memoize:
                return domain_idxs
            self._domain_idxs[idx,0:len(domain_idxs)] = domain_idxs

        return self._domain_idxs[idx]

    # def _get_domain_cooccur_probs(self, idx):
    #     """
    #     Returns co-occurrence probability for every domain value with every
    #     initial context value (categorical and numerical (quantized)).

    #     Returns (max_cat_domain, # of init attrs) tensor.
    #     """
    #     cur = self._train_records[idx]

    #     cooccur_probs = torch.zeros(self.max_cat_domain,
    #             self._n_init_attrs,
    #             dtype=torch.float)

    #     # Compute co-occurrence statistics.
    #     for attr_idx, attr in enumerate(self._all_attrs):
    #         ctx_val = self._qtized_raw_data_dict[cur['_tid_']][attr]
    #         if attr == cur['attribute'] or ctx_val == NULL_REPR or \
    #                 ctx_val not in self._pair_stats[attr][cur['attribute']]:
    #             continue

    #         denom = self._single_stats[attr][ctx_val]
    #         for dom_idx, dom_val in enumerate(cur['domain']):
    #             numer = self._pair_stats[attr][cur['attribute']][ctx_val].get(dom_val, 0.)
    #             cooccur_probs[dom_idx,attr_idx] = numer / denom

    #     return cooccur_probs

    def _get_target_numvals(self, idx):
        if not self.memoize or idx not in self._target_numvals:
            cur = self._train_records[idx]
            assert cur['attribute'] in self._train_num_attrs

            target_numvals = torch.zeros(self._max_num_dim, dtype=torch.float32)

            # Get the target values for this numerical group.
            attr_group = self._train_num_attrs_group[cur['attribute']]
            target_val_strs = [self._raw_data_dict[cur['_tid_']][attr]
                    for attr in attr_group]

            # We can skip this if we are in inference mode and any of the
            # target/current values in the numerical group are NULL.
            if not (self.inference_mode and any(val == NULL_REPR for val in target_val_strs)):
                target_numvals[:len(attr_group)] = torch.FloatTensor(np.array(target_val_strs, dtype=np.float32))

            if not self.memoize:
                return target_numvals
            self._target_numvals[idx,0:len(target_numvals)] = target_numvals

        return self._target_numvals[idx]


    def _get_init_cat_idxs(self, idx):
        """
        Note: implicitly assigns the 0th index/vector for init value of the current
        attribute (i.e. target).

        This should not affect anything since we sum up the vectors whereby
        the 0th vector is always the 0 vector.

        Returns init_cat_idxs: (n_init_cat_attrs)
        """
        if self._n_init_cat_attrs == 0:
            return torch.empty(0)

        if not self.memoize or idx not in self._init_cat_idxs:
            cur = self._train_records[idx]

            init_cat_idxs = []
            for attr in self._init_cat_attrs:
                ctx_val = self._raw_data_dict[cur['_tid_']][attr]
                # If the context attribute is the current target attribute
                # we use the 0-vector.
                # If we are in inference mode, we need to ensure we've seen
                # the context value before, otherwise we assign the 0-vector.
                if attr == cur['attribute'] or \
                        (self.inference_mode and \
                        ctx_val not in self._seen_init_cat_vals[attr]):
                    init_cat_idxs.append(0)
                    continue
                self._seen_init_cat_vals[attr].add(ctx_val)
                init_cat_idxs.append(self._init_val_idxs[attr][ctx_val])
            init_cat_idxs = torch.LongTensor(init_cat_idxs)

            if not self.memoize:
                return init_cat_idxs
            self._init_cat_idxs[idx] = init_cat_idxs

        return self._init_cat_idxs[idx]

    def _get_init_numvals(self, idx):
        """
        Note: value AND mask are set to 0 if the attribute is the current
        attribute (i.e.  target) OR if the value is _nan_.

        Returns (
            init_numvals: (n_init_num_attrs),
            init_nummmask: (n_init_num_attrs),
            ).
        """
        if self._n_init_num_attrs == 0:
            return torch.empty(0, 0), torch.empty(0)

        if not self.memoize or idx not in self._init_numvals:
            cur = self._train_records[idx]

            init_numvals = torch.zeros(self._n_init_num_attrs, dtype=torch.float32)
            init_nummask = torch.ones(self._n_init_num_attrs)
            for attr_idx, attr in enumerate(self._init_num_attrs):
                val_str = self._raw_data_dict[cur['_tid_']][attr]
                if attr == cur['attribute'] or val_str == NULL_REPR:
                    init_nummask[attr_idx] = 0.
                    continue

                init_numvals[attr_idx] = float(val_str)

            if not self.memoize:
                return init_numvals, init_nummask
            self._init_numvals[idx] = init_numvals
            self._init_nummask[idx] = init_nummask

        return self._init_numvals[idx], self._init_nummask[idx]

    def set_mode(self, inference_mode):
        """
        inference_mode = True means to start inference (i.e. use KNN
        for domain instead of random vectors, allows _nan_ target num vals).
        """
        self.inference_mode = inference_mode

    def _get_cat_domain_target(self, idx):
        """
        Returns (domain_idxs, domain_mask, target (scalar)) torch tensors for
        categorical VIDs.
        """
        cur = self._train_records[idx]
        assert cur['attribute'] in self._train_cat_attrs

        domain_idxs = self._get_domain_idxs(idx)
        dom_size = cur['domain_size']
        # During training, add negative samples to a most likely correct (clean) cell
        if not self.inference_mode and self._neg_sample \
                and dom_size < self.max_cat_domain and cur['is_clean']:
            # It is faster not to memoize these.
            neg_idxs = self._get_neg_dom_idxs(idx)
            neg_sample = torch.LongTensor(np.random.choice(neg_idxs,
                    size=min(len(neg_idxs), self.max_cat_domain - dom_size),
                    replace=False))

            domain_idxs[dom_size:dom_size+len(neg_sample)] = neg_sample
            dom_size += len(neg_sample)

        # Position of init in domain values (target)
        target = cur['weak_label_idx']

        # Mask out non-relevant values from padding (see below)
        domain_mask = torch.zeros(self.max_cat_domain, dtype=torch.float)
        domain_mask[dom_size:] = -1 * 1e9

        return domain_idxs, domain_mask, torch.LongTensor([target])

    def __getitem__(self, vid):
        """
        :param:`vid` is the desired VID.

        In parenthese is the size of the tensor. torch.DataLoader will stack these
        into (batch size, *tensor size).

        Returns (vid: (1),
            is_categorical: (1),
            attr_idx: (1),
            init_cat_idxs: (n_init_cat_attrs),
            init_numvals: (n_init_num_attrs),
            init_nummask: (n_init_num_attrs),
            domain_idxs (if categorical): (max domain),
            domain_mask (if categorical): (max domain),
            target_numvals (if numerical): (max num dim),
            cat_target (if categorical): (1),
            )

        where if VID is not categorical/numerical, then the corresponding
        vector are replaced with dummy vectors.

        target_numvals is 0-padded up to max_num_dim for concating.
        """
        idx = self._vid_to_idx[vid]
        cur = self._train_records[idx]
        assert cur['_vid_'] == vid

        is_categorical = torch.ByteTensor([int(cur['attribute'] in self._train_cat_attrs)])
        attr_idx  = torch.LongTensor([self._train_attr_idxs[cur['attribute']]])
        init_cat_idxs = self._get_init_cat_idxs(idx)
        init_numvals, init_nummask = self._get_init_numvals(idx)

        vid = torch.LongTensor([vid])

        # NB: we always return tensors so we can stack them easily even if
        # we index into dataset one-by-one.

        # Categorical VID
        if cur['attribute'] in self._train_cat_attrs:
            domain_idxs, domain_mask, target = self._get_cat_domain_target(idx)
            # TODO(richardwu): decide if we care about co-occurrence probabilities or not.
            # domain_cooccur = self._get_domain_cooccur_probs(idx)
            return vid, \
                is_categorical, \
                attr_idx, \
                init_cat_idxs, \
                init_numvals, \
                init_nummask, \
                domain_idxs, \
                domain_mask, \
                self._dummy_target_numvals, \
                target

        # Numerical VID
        target_numvals = self._get_target_numvals(idx)
        return vid, \
            is_categorical, \
            attr_idx, \
            init_cat_idxs, \
            init_numvals, \
            init_nummask, \
            self._dummy_domain_idxs, \
            self._dummy_domain_mask, \
            target_numvals, \
            self._dummy_cat_target

    def domain_values(self, vid):
        idx = self._vid_to_idx[vid]
        cur = self._train_records[idx]
        assert cur['attribute'] in self._train_cat_attrs
        return cur['domain']

    def _state_attrs(self):
        """
        Attributes/local vars to dump as state. Basically everything
        used when __getitem__ is invoked.
        """
        return ['_vid_to_idx',
                '_train_records',
                '_raw_data_dict',
                # '_qtized_raw_data_dict',
                # '_single_stats',
                # '_pair_stats',
                'max_cat_domain',
                '_max_num_dim',
                '_init_val_idxs',
                '_train_val_idxs',
                '_init_attr_idxs',
                '_train_attr_idxs',
                '_init_cat_attrs',
                '_init_num_attrs',
                '_train_cat_attrs',
                '_train_num_attrs',
                '_train_num_attrs_group',
                '_numerical_attr_groups',
                '_num_attrs_mean',
                '_num_attrs_std',
                '_n_init_cat_attrs',
                '_n_init_num_attrs',
                '_train_val_idxs_by_attr',
                ]

    def get_state(self):
        return {attr: getattr(self, attr) for attr in self._state_attrs()}

    def load_state(self, state):
        for attr, val in state.items():
            setattr(self, attr, val)
        self._init_dummies()
        self._init_memoize_vecs()

class IterSampler(Sampler):
    def __init__(self, iter):
        self.iter = iter

    def __iter__(self):
        return iter(self.iter)

    def __len__(self):
        return len(self.iter)

class VidSampler(Sampler):
    def __init__(self, domain_df, raw_df, num_attrs, numerical_attr_groups,
            shuffle=True, train_only_clean=False):
        # No NULL categorical targets
        domain_df = domain_df[domain_df['attribute'].isin(num_attrs) | (domain_df['weak_label'] != NULL_REPR)]

        # No NULL values in each cell's numerical group (all must be non-null
        # since target_numvals requires all numerical values.
        if numerical_attr_groups:
            raw_data_dict = raw_df.set_index('_tid_').to_dict('index')
            attr_to_group = {attr: group for group in numerical_attr_groups
                    for attr in group}
            def group_notnull(row):
                tid = row['_tid_']
                cur_attr = row['attribute']
                # Non-numerical cell: return true
                if cur_attr not in attr_to_group:
                    return True
                return all(raw_data_dict[tid][attr] != NULL_REPR
                        for attr in attr_to_group[cur_attr])
            fil_notnull = domain_df.apply(group_notnull, axis=1)
            if domain_df.shape[0] and sum(fil_notnull) < domain_df.shape[0]:
                logging.warning('dropping %d targets where target\'s numerical group contain NULLs',
                        domain_df.shape[0] - sum(fil_notnull))
                domain_df = domain_df[fil_notnull]

        # Train on only clean cells
        if train_only_clean:
            self._vids = domain_df.loc[(domain_df['is_clean'] | domain_df['fixed'] >= 1), '_vid_']
        else:
            self._vids = domain_df['_vid_'].values

        if shuffle:
            self._vids = np.random.permutation(self._vids)

    def __iter__(self):
        return iter(self._vids.tolist())

    def __len__(self):
        return len(self._vids)


class TupleEmbedding(Estimator, torch.nn.Module):
    WEIGHT_DECAY = 0.

    # TODO: replace numerical_attrs references with self.ds.numerical_attrs
    def __init__(self, env, dataset, domain_df,
            numerical_attr_groups=None,
            memoize=False,
            neg_sample=True,
            dropout_pct=0.,
            learning_rate=0.05,
            validate_fpath=None, validate_tid_col=None, validate_attr_col=None,
            validate_val_col=None, validate_epoch=None):
        """
        :param dataset: (Dataset) original dataset
        :param domain_df: (DataFrame) dataframe containing domain values
        :param numerical_attr_groups: (list[list[str]]) attributes/columns to treat as numerical.
            A list of groups of column names. Each group consists of d attributes
            to be treated as d-dimensional numerical attribute.
            For example one can pass in [['lat', 'lon'],...] to treat both columns as
            a 2-d numerical attribute.

            The groups must be disjoint.

            Everything else will be treated as categorical.

            If None, treats everything as categorical.
        :param neg_sample: (bool) add negative examples for clean cells during training
        :param validate_fpath: (string) filepath to validation CSV
        :param validate_tid_col: (string) column containing TID
        :param validate_attr_col: (string) column containing attribute
        :param validate_val_col: (string) column containing correct value
        """
        torch.nn.Module.__init__(self)
        Estimator.__init__(self, env, dataset, domain_df)

        self.inference_mode = False

        assert dropout_pct < 1 and dropout_pct >= 0
        self.dropout_pct = dropout_pct

        self._embed_size = self.env['estimator_embedding_size']
        train_attrs = self.env['train_attrs']
        numerical_attr_groups = numerical_attr_groups or []

        # Check if train attributes exist
        if train_attrs is not None:
            if not all(attr in self.ds.get_attributes() for attr in train_attrs):
                logging.error('%s: all attributes specified to use for training %s must exist in dataset: %s',
                        type(self).__name__,
                        train_attrs,
                        self.ds.get_attributes())
                raise Exception()

        ### Numerical attribute groups validation checks

        self._numerical_attr_groups = numerical_attr_groups.copy()
        self._numerical_attrs = verify_numerical_attr_groups(self.ds, self._numerical_attr_groups)
        # Verify numerical dimensions are not bigger than the embedding size
        if  max(list(map(len, numerical_attr_groups)) or [0]) > self._embed_size:
            logging.error("%s: maximum numeric value dimension %d must be <= embedding size %d",
                    type(self).__name__,
                    max(list(map(len, numerical_attr_groups)) or [0]),
                    self._embed_size)
            raise Exception()

        self.load_domain_df(self.domain_df, load_into_ds=False)

        # Dataset
        self._dataset = LookupDataset(env, dataset, self.domain_df,
                numerical_attr_groups, neg_sample, memoize)

        self.max_cat_domain = self._dataset.max_cat_domain
        logging.debug('%s: max domain size: (categorical) %d, (numerical) %d',
                type(self).__name__,
                self.max_cat_domain,
                self.max_domain)

        self._train_cat_attrs = self._dataset._train_cat_attrs
        self._train_num_attrs = self._dataset._train_num_attrs

        # word2vec-like model.

        self._n_init_vals = self._dataset.n_init_vals
        self._n_train_vals = self._dataset.n_train_vals

        self._n_init_cat_attrs = self._dataset._n_init_cat_attrs
        self._n_init_num_attrs = self._dataset._n_init_num_attrs
        self._n_init_attrs = self._dataset._n_init_attrs

        self._n_train_cat_attrs = self._dataset._n_train_cat_attrs
        self._n_train_num_attrs = self._dataset._n_train_num_attrs
        self._n_train_attrs = self._n_train_cat_attrs + self._n_train_num_attrs

        self.max_cat_domain = self._dataset.max_cat_domain
        self._max_num_dim = self._dataset._max_num_dim

        self.in_W = torch.nn.Parameter(torch.zeros(self._n_init_vals, self._embed_size))
        self.out_W = torch.nn.Parameter(torch.zeros(self._n_train_vals, self._embed_size))
        self.out_B = torch.nn.Parameter(torch.zeros(self._n_train_vals, 1))

        ### Bases vectors for numerical attributes and their spans.

        # Mask to combine numerical bases to form a numerical group
        self._n_num_attr_groups = len(self._numerical_attr_groups)
        self.num_attr_groups_mask = torch.zeros(self._n_num_attr_groups,
                self._n_init_num_attrs, dtype=torch.float32)
        for group_idx, group in enumerate(self._numerical_attr_groups):
            for attr in group:
                attr_idx = self._dataset._init_attr_idxs[attr] - self._n_init_cat_attrs
                self.num_attr_groups_mask[group_idx,attr_idx] = 1.
        # For each numerical attribute we have a basis vector.
        # For each numerical group we find the linear combination from the
        # individual vectors.
        # We also have a learnable zero vector for every numerical group.
        self.in_num_bases = torch.nn.Parameter(torch.zeros(self._n_init_num_attrs,
            self._embed_size))
        self.in_num_zero_vecs = torch.nn.Parameter(torch.zeros(self._n_num_attr_groups,
            self._embed_size))

        # Non-linearity for numerical init attrs
        self.in_num_w1 = torch.nn.Parameter(torch.zeros(self._n_num_attr_groups, self._embed_size, self._embed_size))
        self.in_num_bias1 = torch.nn.Parameter(torch.zeros(self._n_num_attr_groups, self._embed_size))

        self.out_num_bases = torch.nn.Parameter(torch.zeros(self._n_train_num_attrs, self._embed_size, self._max_num_dim))
        # Non-linearity for combined_init for each numerical attr
        self.out_num_w1 = torch.nn.Parameter(torch.zeros(self._n_train_num_attrs, self._embed_size, self._embed_size))
        self.out_num_bias1 = torch.nn.Parameter(torch.zeros(self._n_train_num_attrs, self._embed_size))


        # Mask for _num_forward to restrict which dimensions are active for each attribute.
        # Hadamard/elementwise multiply this mask.
        self.out_num_masks = torch.zeros(self._n_train_num_attrs,
                self._max_num_dim, dtype=torch.float32)
        # Mask to select out the relevant 1-d value for an attribute from
        # its attr group.
        self._num_attr_group_mask = torch.zeros(self._n_train_num_attrs,
                self._max_num_dim, dtype=torch.float32)
        for idx, attr in enumerate(self._dataset._train_num_attrs):
            dim = len(self._dataset._train_num_attrs_group[attr])
            attr_idx = self._dataset._train_num_attrs_group[attr].index(attr)
            self.out_num_masks[idx,:dim] = 1.
            self._num_attr_group_mask[idx, attr_idx] = 1.

        # logits fed into softmax used in weighted sum to combine
        # dot products of in_W and out_W per attribute.
        # Equivalent to choosing which input vectors to "focus" on.
        # Each row corresponds to the logits per each attr/column we want
        # to predict for and there are init_attr weights since we have
        # init attrs to combine.
        self.attr_W = torch.nn.Parameter(torch.zeros(self._n_train_attrs,
            self._n_init_cat_attrs + self._n_num_attr_groups))

        # Weights for 1) embedding score and 2) co-occurrence probabilities
        # for categorical domain values.
        self.cat_feat_W = torch.nn.Parameter(torch.zeros(self._n_train_attrs,
            1 + self._n_init_attrs, 1))

        # Initialize all but the first 0th vector embedding (reserved).
        torch.nn.init.xavier_uniform_(self.in_W[1:])
        torch.nn.init.xavier_uniform_(self.out_W[1:])
        torch.nn.init.xavier_uniform_(self.out_B[1:])

        if self._n_init_num_attrs > 0:
            torch.nn.init.xavier_uniform_(self.in_num_zero_vecs)
            torch.nn.init.xavier_uniform_(self.in_num_bases)
            torch.nn.init.xavier_uniform_(self.in_num_w1)
            torch.nn.init.xavier_uniform_(self.in_num_bias1)

        if self._n_train_num_attrs > 0:
            torch.nn.init.xavier_uniform_(self.out_num_bases)
            torch.nn.init.xavier_uniform_(self.out_num_w1)
            torch.nn.init.xavier_uniform_(self.out_num_bias1)

        torch.nn.init.xavier_uniform_(self.attr_W)
        torch.nn.init.xavier_uniform_(self.cat_feat_W)

        self._cat_loss = CrossEntropyLoss()
        # TODO: we use MSE loss for all numerical attributes for now.
        # Allow user to pass in their desired loss.
        self._num_loss = MSELoss(reduction='mean')
        self._optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=self.WEIGHT_DECAY)

        # Validation stuff
        self._do_validation = False
        if validate_fpath is not None \
            and validate_tid_col is not None \
            and validate_attr_col is not None \
            and validate_val_col is not None:
            self._validate_df = pd.read_csv(validate_fpath, dtype=str)
            self._validate_df.rename({validate_tid_col: '_tid_',
                validate_attr_col: '_attribute_',
                validate_val_col: '_value_',
                }, axis=1, inplace=True)
            self._validate_df['_tid_'] = self._validate_df['_tid_'].astype(int)
            self._validate_df['_value_'] = self._validate_df['_value_'].str.strip().str.lower()
            # Merge left so we can still get # of repairs for cells without
            # ground truth.
            self._validate_df = self.domain_df.merge(self._validate_df, how='left',
                    left_on=['_tid_', 'attribute'], right_on=['_tid_', '_attribute_'])
            self._validate_df['_value_'].fillna(NULL_REPR, inplace=True)
            # | separated correct values
            self._validate_df['_value_'] = self._validate_df['_value_'].str.split('\|')

            fil_notnull = self._validate_df['_value_'].apply(lambda arr: arr != [NULL_REPR])

            # Raise error if validation set has non-numerical values for numerical attrs
            if self._numerical_attrs is not None:
                fil_attr = self._validate_df['attribute'].isin(self._numerical_attrs)
                fil_notnumeric = self._validate_df['_value_'].apply(lambda arr: arr[0]).str.contains(NONNUMERICS)
                bad_numerics = fil_attr & fil_notnull & fil_notnumeric
                if bad_numerics.sum():
                    logging.error('%s: validation dataframe contains %d non-numerical values in numerical attrs %s',
                        type(self).__name__,
                        bad_numerics.sum(),
                        self._numerical_attrs)
                    raise Exception()

            # Log how many cells are actually repairable based on domain generated.
            # Cells without ground truth are "not repairable".
            fil_repairable = self._validate_df[fil_notnull].apply(lambda row: any(v in row['domain'] for v in row['_value_']), axis=1)
            logging.debug("%s: max repairs possible (# cells ground truth in domain): (DK) %d, (all): %d",
                        type(self).__name__,
                        (fil_repairable & ~self._validate_df['is_clean']).sum(),
                        fil_repairable.sum())

            self._validate_df = self._validate_df[['_vid_', 'attribute', 'init_value', '_value_', 'is_clean']]
            self._validate_epoch = validate_epoch or 1
            self._do_validation = True

    def load_raw_data(self, df_raw):
        self._dataset.load_raw_data(df_raw)

    def load_domain_df(self, domain_df, load_into_ds=True):
        """
        load_domain_df loads a new domain DataFrame to use to output predictions.

        Note an exception will be thrown if there are out-of-vocab words.

        load_into_ds will also load the domain_df into the underlying LookupDataset.
        One may want this to be false during initialization.
        """
        self.domain_df = domain_df
        # Remove domain for numerical attributes.
        fil_numattr = self.domain_df['attribute'].isin(self._numerical_attrs)

        # Memoize max domain size for numerical attribue for padding later.
        self.max_domain = int(self.domain_df['domain_size'].max())
        self.domain_df.loc[fil_numattr, 'domain'] = ''
        self.domain_df.loc[fil_numattr, 'domain_size'] = 0
        # Remove categorical domain/training cells without a domain
        filter_empty_domain = (self.domain_df['domain_size'] == 0) & ~fil_numattr
        if filter_empty_domain.sum():
            logging.warning('%s: removing %d categorical cells with empty domains',
                type(self).__name__,
                filter_empty_domain.sum())
            self.domain_df = self.domain_df[~filter_empty_domain]
        # Pre-split domain.
        self.domain_df['domain'] = self.domain_df['domain'].str.split('\|\|\|')

        # Add DK information to domain dataframe
        if self.ds.aux_table[AuxTables.dk_cells] is not None:
            df_dk = self.ds.aux_table[AuxTables.dk_cells].df
            self.domain_df = self.domain_df.merge(df_dk,
                    on=['_tid_', 'attribute'], how='left', suffixes=('', '_dk'))
            self.domain_df['is_clean'] = self.domain_df['_cid__dk'].isnull()
        else:
            self.domain_df['is_clean'] = True
            self.domain_df.loc[self.domain_df['weak_label'] == NULL_REPR, 'is_clean'] = False
        self.domain_df = self.domain_df[self.domain_df['attribute'].isin(self.env['train_attrs'])]

        self.domain_recs = self.domain_df.to_records()
        if load_into_ds:
            self._dataset.load_domain_df(domain_df)
        import pdb; pdb.set_trace()


    def _get_combined_init_vec(self, init_cat_idxs, init_numvals, init_nummasks, attr_idxs):
        """
        Constructs the "context vector" by combining the init embedding vectors.

        init_cat_idxs: (batch, n_init_cat_attrs)
        init_numvals: (batch, n_init_num_attrs)
        init_nummasks: (batch, n_init_num_attrs)
        attr_idxs: (batch, 1)

        out: (batch, embed_size, 1)
        """
        init_cat_vecs = torch.zeros(init_cat_idxs.shape[0], 0, self._embed_size)
        if self._n_init_cat_attrs:
            # (batch, n_init_cat_attrs, embed size)
            init_cat_vecs = self.in_W.index_select(0, init_cat_idxs.view(-1)).view(*init_cat_idxs.shape, self._embed_size)

        init_num_vecs = torch.zeros(init_numvals.shape[0], 0, self._embed_size)
        if self._n_init_num_attrs:
            # (batch, n_init_num_attrs, 1)
            init_numvals = init_numvals.unsqueeze(-1)
            # self.in_num_bases is shape (n_init_num_attrs, embed_size)
            # (batch, n_init_num_attrs, embed_size)
            in_num_bases = self.in_num_bases.expand(init_numvals.shape[0], -1, -1)
            # (batch, n_init_num_attrs, embed_size)
            init_num_vecs = in_num_bases.mul(init_numvals)


            # self.num_attr_groups_mask is shape (n_num_attr_groups, n_init_num_attrs)
            # (batch, n_num_attr_groups, n_init_num_attrs)
            groups_mask = self.num_attr_groups_mask.expand(init_numvals.shape[0],
                    -1, -1)

            # (batch, n_num_attr_groups, n_init_num_attrs, embed_size)
            init_num_vecs = groups_mask.unsqueeze(-1) \
                    * init_num_vecs.unsqueeze(1).expand(-1, self._n_num_attr_groups, -1, -1)
            # (batch, n_num_attr_groups, embed_size)
            init_num_vecs = init_num_vecs.sum(dim=2) + self.in_num_zero_vecs.unsqueeze(0)


            #### Add non-linearity to numerical attributes
            # (batch, n_num_attr_groups, embed_size)
            ReLU(inplace=True)(init_num_vecs)
            # (batch, n_num_attr_groups, embed_size, embed_size)
            in_num_w1 = self.in_num_w1.expand(init_numvals.shape[0], -1, -1, -1)
            # (batch, n_num_attr_groups, embed_size)
            init_num_vecs = init_num_vecs.unsqueeze(-2).matmul(in_num_w1).squeeze(-2) \
                    + self.in_num_bias1.unsqueeze(0)


            # (batch, n_num_attr_groups, 1)
            # If any of the init values are NULL in a group, zero it out.
            # We do this by multiplying through the groups mask with each
            # individual numeric attribute's mask and comparing
            # how many numerical attributes got dropped per group.
            init_group_nummasks = (groups_mask.sum(dim=-1, keepdim=True) \
                    == (groups_mask * init_nummasks.unsqueeze(1)).sum(dim=-1, keepdim=True)).type(torch.FloatTensor)

            # (batch, n_num_attr_groups, embed_size)
            init_num_vecs.mul_(init_group_nummasks)

        # (batch, n_init_cat_attrs + n_num_attr_groups, embed size)
        init_vecs = torch.cat([init_cat_vecs, init_num_vecs], dim=1)
        # Scale vectors to unit norm ALONG the embedding dimension.
        # (batch, n_init_cat_attrs + n_num_attr_groups, embed size)
        init_vecs = F.normalize(init_vecs, p=2, dim=2)

        # (batch, 1, n_init_cat_attrs + n_num_attr_groups)
        attr_logits = self.attr_W.index_select(0, attr_idxs.view(-1)).unsqueeze(1)
        # (batch, 1, n_init_cat_attrs + n_num_attr_groups)
        attr_weights = Softmax(dim=2)(attr_logits)

        # Apply dropout to individual attributes of context
        if self.dropout_pct > 0 and not self.inference_mode:
            dropout_mask = (torch.rand_like(attr_weights) > self.dropout_pct).type(torch.FloatTensor)
            attr_weights = attr_weights.mul(dropout_mask / (1. - self.dropout_pct))

        # (batch, 1, embed size)
        combined_init = attr_weights.matmul(init_vecs)
        # (batch, embed size, 1)
        combined_init = combined_init.view(combined_init.shape[0], combined_init.shape[2], 1)

        # (batch, embed size, 1)
        return combined_init

    def _cat_forward(self, combined_init, domain_idxs, domain_masks):
        """
        combined_init: (batch, embed size, 1)
        cat_attr_idxs: (batch, 1)
        domain_idxs: (batch, max domain)
        domain_masks: (batch, max domain)
        domain_cooccur: (batch, max domain, # of init attrs)

        Returns logits: (batch, max domain)
        """
        # (batch, max domain, embed size)
        domain_vecs = self.out_W.index_select(0, domain_idxs.view(-1)).view(*domain_idxs.shape, self._embed_size)
        # (batch, max domain, 1)
        embed_prods = domain_vecs.matmul(combined_init)
        # (batch, max domain, 1)
        domain_biases = self.out_B.index_select(0, domain_idxs.view(-1)).view(*domain_idxs.shape, 1)
        # (batch, max domain, 1)
        embed_prods.add_(domain_biases)

        logits = embed_prods.squeeze(-1)

        # # (batch, max domain, 1 + # of init attrs)
        # domain_feats = torch.cat([embed_prods, domain_cooccur], dim=-1)

        # # (batch, 1 + # of init attrs, 1)
        # cat_feat_W = self.cat_feat_W.index_select(0, cat_attr_idxs.view(-1)).view(domain_feats.shape[0],
        #         *self.cat_feat_W.shape[1:])
        # # (batch, max domain)
        # logits = domain_feats.matmul(cat_feat_W).squeeze(-1)

        # Add mask to void out-of-domain indexes
        # (batch, max domain)
        logits.add_(domain_masks)

        return logits

    def _num_forward(self, combined_init, num_attr_idxs):
        """
        batch is actually "# of num cells".

        combined_init: (batch, embed size, 1)
        num_attr_idxs: (batch, 1)

        Returns pred_numvals: (batch, max_num_dim)
        """
        # (batch, embed size, max num dim)
        out_num_bases = self.out_num_bases.index_select(0, num_attr_idxs.view(-1))
        # (batch, max num dim)
        out_num_masks = self.out_num_masks.index_select(0, num_attr_idxs.view(-1))

        """
        Use pseudo-inverse for regression
        """
        # (batch, embed size, max num dim)
        normed_out_num_bases = F.normalize(out_num_bases, p=2, dim=1)

        # mask out useless vectors
        # (batch, embed size, max num dim)
        normed_out_num_bases.mul_(out_num_masks.unsqueeze(1))

        # (batch, embed size, embed size)
        out_num_w1 = self.out_num_w1.index_select(0, num_attr_idxs.view(-1))
        # (batch, 1, embed size)
        out_num_bias1 = self.out_num_bias1.index_select(0, num_attr_idxs.view(-1)).unsqueeze(1)


        # Apply non-linearity
        ReLU(inplace=True)(combined_init)
        # (batch, 1, embed size)
        combined_init2 = combined_init.view(-1, 1, self._embed_size).matmul(out_num_w1) + out_num_bias1
        # (batch, embed size, 1)
        combined_init2 = combined_init2.view(-1, self._embed_size, 1)

        ### Project non-linear context onto basis vectors.
        # (batch, max num dim)
        pred_numvals = (combined_init2 * normed_out_num_bases).sum(dim=1)
        pred_numvals.mul_(out_num_masks)

        return pred_numvals

    def set_mode(self, inference_mode):
        self.inference_mode = inference_mode
        self._dataset.set_mode(inference_mode)

    def forward(self, is_categorical, attr_idxs,
                init_cat_idxs, init_numvals, init_nummasks,
                domain_idxs, domain_masks):
        """
        Performs one forward pass.
        """
        # (batch, embed size, 1)
        combined_init = self._get_combined_init_vec(init_cat_idxs, init_numvals,
                init_nummasks, attr_idxs)

        # (# of cat VIDs), (# of num VIDs)
        cat_mask, num_mask = self._cat_num_masks(is_categorical)

        cat_logits = torch.empty(0, self.max_cat_domain)
        if len(cat_mask):
            cat_combined_init, domain_idxs, domain_masks = \
                    combined_init[cat_mask], \
                    domain_idxs[cat_mask], \
                    domain_masks[cat_mask]
            # (# of cat VIDs, max_cat_domain)
            cat_logits = self._cat_forward(cat_combined_init, domain_idxs,
                    domain_masks)

        pred_numvals = torch.empty(0, self._max_num_dim)
        if len(num_mask):
            num_combined_init, num_attr_idxs = combined_init[num_mask], \
                    self._num_attr_idxs(is_categorical, attr_idxs)
            # (# of num VIDs, max_num_dim)
            pred_numvals = self._num_forward(num_combined_init, num_attr_idxs)

        return cat_logits, pred_numvals

    def _cat_num_masks(self, is_categorical):
        """
        is_categorical: (batch, 1)
        """
        # TODO: is_catrgorical is already ByteTensor: use  torch.mask_tensor
        cat_mask, num_mask = is_categorical.view(-1).nonzero().view(-1),\
            (is_categorical.view(-1) == 0).nonzero().view(-1)
        return cat_mask, num_mask

    def _num_attr_idxs(self, is_categorical, attr_idxs):
        """
        Returns the 0-indexed numerical attribute indexes from a batch of
        attribute indexes.

        is_categorical: (batch, 1)
        attr_idxs: (batch, 1)

        Returns tensor of shape (# of numerical examples, 1)
        """
        _, num_mask = self._cat_num_masks(is_categorical)
        # shift attribute indexes back to 0
        num_attr_idxs = attr_idxs[num_mask] - self._n_train_cat_attrs

        # (# of numerical examples, 1)
        return num_attr_idxs

    def train(self, num_epochs=10, batch_size=32, weight_entropy_lambda=0.,
            shuffle=True, train_only_clean=False):
        """
        :param num_epochs: (int) number of epochs to train for
        :param batch_size: (int) size of batches
        :param weight_entropy_lambda: (float) penalization strength for
            weights assigned to other attributes for a given attribute.
            A higher penalization strength means the model will depend
            on more attributes instead of putting all weight on a few
            attributes. Recommended values between 0 to 0.5.
        :param shuffle: (bool) shuffle the dataset while training
        :param train_only_clean: (bool) train only on clean cells not marked by
            error detection. Recommend False if error detector is very liberal.
        """

        # Returns VIDs to train on.
        sampler = VidSampler(self.domain_df, self.ds.get_raw_data(),
                self._numerical_attrs, self._numerical_attr_groups,
                shuffle=shuffle, train_only_clean=train_only_clean)

        logging.debug("%s: training (lambda = %f) on %d cells (%d cells in total) in:\n1) %d categorical columns: %s\n2) %d numerical columns: %s",
                      type(self).__name__,
                      weight_entropy_lambda,
                      len(sampler),
                      self.domain_df.shape[0],
                      self._n_train_cat_attrs,
                      self._train_cat_attrs,
                      self._n_train_num_attrs,
                      self._train_num_attrs)

        num_batches = len(DataLoader(self._dataset, batch_size=batch_size, sampler=sampler))
        num_steps = num_epochs * num_batches
        # scheduler = CosineAnnealingLR(self._optimizer, num_steps)
        # logging.debug('%s: using cosine LR scheduler with %d steps', type(self).__name__, num_steps)

        batch_losses = []
        # Main training loop.
        for epoch_idx in range(1, num_epochs+1):
            logging.debug('%s: epoch %d of %d', type(self).__name__, epoch_idx, num_epochs)
            batch_cnt = 0
            scheduler = CosineAnnealingLR(self._optimizer, num_batches)
            logging.debug('%s: using cosine LR scheduler with %d steps', type(self).__name__, num_batches)

            for vids, is_categorical, attr_idxs, \
                init_cat_idxs, init_numvals, init_nummasks, \
                domain_idxs, domain_masks, \
                target_numvals, cat_targets \
                in tqdm(DataLoader(self._dataset, batch_size=batch_size, sampler=sampler)):

                cat_preds, numval_preds = self.forward(is_categorical, attr_idxs,
                        init_cat_idxs, init_numvals, init_nummasks,
                        domain_idxs, domain_masks)

                # Select out the appropriate targets
                cat_mask, num_mask = self._cat_num_masks(is_categorical)
                cat_targets = cat_targets.view(-1)[cat_mask]
                target_numvals = target_numvals[num_mask]

                assert cat_preds.shape[0] == cat_targets.shape[0]
                assert numval_preds.shape == target_numvals.shape

                batch_loss = 0.
                if cat_targets.shape[0] > 0:
                    batch_loss += self._cat_loss(cat_preds, cat_targets)
                if target_numvals.shape[0] > 0:
                    # Note both numval_preds and target_numvals have 0-ed out
                    # values if the sample's dimension is < max dim.
                    # TODO: downweight samples that are part of a group of n attributes
                    # by 1/n.
                    batch_loss += self._num_loss(numval_preds, target_numvals)

                # Add the negative entropy of the attr_W to the cost: that is
                # we maximize entropy of the logits of attr_W to encourage
                # non-sparsity of logits.
                if weight_entropy_lambda != 0.:
                    attr_weights = Softmax(dim=1)(self.attr_W).view(-1)
                    neg_attr_W_entropy = attr_weights.dot(attr_weights.log()) / self.attr_W.shape[0]
                    batch_loss.add_(weight_entropy_lambda * neg_attr_W_entropy)

                batch_losses.append(float(batch_loss))
                self.zero_grad()
                batch_loss.backward()

                # Do not update weights for 0th reserved vectors.
                if self.in_W._grad is not None:
                    self.in_W._grad[0].zero_()
                if self.out_W._grad is not None:
                    self.out_W._grad[0].zero_()
                if self.out_B._grad is not None:
                    self.out_B._grad[0].zero_()

                self._optimizer.step()
                scheduler.step()
                batch_cnt += 1

            logging.debug('%s: average batch loss: %f',
                    type(self).__name__,
                    sum(batch_losses[-1 * batch_cnt:]) / batch_cnt)

            if self._do_validation and epoch_idx % self._validate_epoch == 0:
                res = self.validate()

        return batch_losses

    def get_features(self, vids):
        """
        Returns three tensors:
            cat_probas: (# of vids, max domain)
            num_predvals: (# of vids, 1)
            is_categorical: (# of vids, 1)
        """
        # No gradients required.
        with torch.no_grad():
            ret_cat_probas = torch.zeros(len(vids), self.max_cat_domain)
            ret_num_predvals = torch.zeros(len(vids), 1)
            ret_is_categorical = torch.zeros(len(vids), 1, dtype=torch.uint8)

            batch_sz = int(1e5 / self._embed_size)
            num_batches = math.ceil(len(vids) / batch_sz)
            logging.debug('%s: getting features in batches (# batches = %d, size = %d) ...',
                    type(self).__name__, num_batches, batch_sz)

            mask_offset = 0

            self.set_mode(inference_mode=True)
            for vids, is_categorical, attr_idxs, \
                init_cat_idxs, init_numvals, init_nummasks, \
                domain_idxs, domain_masks, \
                target_numvals, cat_targets in tqdm(DataLoader(self._dataset, batch_size=batch_sz, sampler=IterSampler(vids))):

                # (# of cats, max cat domain), (# of num, max_num_dim)
                cat_logits, num_predvals = self.forward(is_categorical,
                        attr_idxs,
                        init_cat_idxs,
                        init_numvals,
                        init_nummasks,
                        domain_idxs,
                        domain_masks)

                if cat_logits.nelement():
                    cat_probas = Softmax(dim=1)(cat_logits)
                else:
                    cat_probas = cat_logits

                # (# of cats), (# of num)
                cat_masks, num_masks = self._cat_num_masks(is_categorical)
                cat_masks.add_(mask_offset)
                num_masks.add_(mask_offset)
                mask_offset += is_categorical.shape[0]
                # (# of num VIDs, 1)
                num_attr_idxs = self._num_attr_idxs(is_categorical, attr_idxs)
                num_attr_group_mask = self._num_attr_group_mask.index_select(0, num_attr_idxs.view(-1))
                # (# of num VIDS, 1)
                num_predvals_masked = (num_attr_group_mask * num_predvals).sum(dim=1, keepdim=True)

                # write values to return tensor
                ret_cat_probas.scatter_(0, cat_masks.unsqueeze(-1).expand(-1, self.max_cat_domain), cat_probas.data)
                ret_num_predvals.scatter_(0, num_masks.unsqueeze(-1), num_predvals_masked.data)
                ret_is_categorical[cat_masks] = 1

                del cat_probas, num_predvals_masked

            self.set_mode(inference_mode=False)

            return ret_cat_probas.detach(), ret_num_predvals.detach(), ret_is_categorical.detach()


    def _model_fpaths(self, prefix):
        return '%s_sdict.pkl' % prefix, '%s_ds_state.pkl' % prefix

    def dump_model(self, prefix):
        """
        Dump this model's parameters and other metadata (e.g. attr-val to
        corresponding index in embedding matrix) with the given
        :param:`prefix`.

        When loading the model one must use the same domain DF.
        """
        sdict_fpath, ds_fpath = self._model_fpaths(prefix)
        logging.debug('%s: saving model to %s and %s',
                      type(self).__name__, sdict_fpath, ds_fpath)

        torch.save(self.state_dict(), sdict_fpath)
        with open(ds_fpath, 'wb') as f:
            pickle.dump(self._dataset.get_state(), f)

    def load_model(self, prefix):
        """
        Tries to load the parameters and state from the given dump prefix.
        Note this EmbeddingModel must be initialized with the same domain DF
        (otherwise it does not make sense to load the same parameters).

        Returns whether the model could be loaded.
        """
        sdict_fpath, ds_fpath = self._model_fpaths(prefix)

        if not os.path.exists(sdict_fpath) or not os.path.exists(ds_fpath):
            logging.warning('%s: cannot load model from prefix %s',
                    type(self).__name__,
                    prefix)
            return False

        logging.debug('%s: loading saved model from %s and %s',
                      type(self).__name__, sdict_fpath, ds_fpath)

        # strict=False to allows backwards compat
        self.load_state_dict(torch.load(sdict_fpath), strict=False)
        with open(ds_fpath, 'rb') as f:
            self._dataset.load_state(pickle.load(f))
        return True

    def dump_predictions(self, prefix, include_all=False):
        """
        Dump inference results to ":param:`prefix`_predictions.pkl" (if not None).
        Returns the dataframe of results.

        include_all = True will include all domain values and their prediction
        probabilities for categorical attributes.
        """
        preds = self.predict_pp_batch()
        import pdb; pdb.set_trace()

        logging.debug('%s: constructing and dumping predictions...',
                      type(self).__name__)
        results = []
        for ((vid, is_cat, pred), row) in zip(preds, self.domain_recs):
            assert vid == row['_vid_']
            if is_cat:
                # Include every domain value and their predicted probabilities
                if include_all:
                    for val, proba in pred:
                        results.append({'tid': row['_tid_'],
                            'vid': vid,
                            'attribute': row['attribute'],
                            'inferred_val': val,
                            'proba': proba})
                else:
                    max_val, max_proba = max(pred, key=lambda t: t[1])
                    results.append({'tid': row['_tid_'],
                        'vid': vid,
                        'attribute': row['attribute'],
                        'inferred_val': max_val,
                        'proba': max_proba})
            else:
                results.append({'tid': row['_tid_'],
                    'vid': vid,
                    'attribute': row['attribute'],
                    'inferred_val': pred,
                    'proba': -1})

        results = pd.DataFrame(results)

        if prefix is not None:
            fpath = '{}_predictions.pkl'.format(prefix)
            logging.debug('%s: dumping predictions to %s', type(self).__name__, fpath)
            results.to_pickle(fpath)
        return results

    def validate(self):
        logging.debug('%s: running validation set...', type(self).__name__)

        # Construct DataFrame with inferred values
        validation_preds = list(self.predict_pp_batch(self._validate_df))
        df_pred = []
        for vid, is_cat, preds in tqdm(validation_preds):
            if is_cat:
                inf_val, inf_prob = max(preds, key=lambda t: t[1])
            else:
                # preds is just a float
                inf_val, inf_prob = preds, -1

            df_pred.append({'_vid_': vid,
                'is_cat': is_cat,
                'inferred_val': inf_val,
                'proba': inf_prob})
        df_pred = pd.DataFrame(df_pred)
        df_res = self._validate_df.merge(df_pred, on=['_vid_'])


        # General filters and metrics
        fil_dk = ~df_res['is_clean']
        fil_cat = df_res['is_cat']
        fil_grdth = df_res['_value_'].apply(lambda arr: arr != [NULL_REPR])

        if (~fil_grdth).sum():
            logging.debug('%s: there are %d cells with no validation ground truth',
                    type(self).__name__,
                    (~fil_grdth).sum())

        n_cat = fil_cat.sum()
        n_num = (~fil_cat).sum()

        n_cat_dk = (fil_dk & fil_cat).sum()
        n_num_dk = (fil_dk & ~fil_cat).sum()

        # Categorical filters and metrics
        fil_err = df_res.apply(lambda row: row['init_value'] not in row['_value_'],
                axis=1) & fil_cat & fil_grdth
        fil_noterr = ~fil_err & fil_cat & fil_grdth
        fil_cor = df_res.apply(lambda row: row['inferred_val'] in row['_value_'],
                axis=1) & fil_cat & fil_grdth
        fil_repair = (df_res['init_value'] != df_res['inferred_val']) & fil_cat

        total_err = fil_err.sum()
        detected_err = (fil_dk & fil_err).sum()

        n_repair = fil_repair.sum()
        n_repair_dk = (fil_dk & fil_repair).sum()
        n_cor_repair = (fil_cor & fil_repair).sum()
        n_cor_repair_dk = (fil_dk & fil_cor & fil_repair).sum()

        if total_err == 0:
            logging.warning('%s: total errors in validation set is 0', type(self).__name__)
        if detected_err == 0:
            logging.warning('%s: total detected errors in validation set is 0', type(self).__name__)

        # In-sample accuracy (predict init value that is already correcT)
        sample_acc = (fil_noterr & fil_cor).sum() / (fil_noterr).sum()

        precision = n_cor_repair / max(n_repair, 1)
        recall = n_cor_repair / max(total_err, 1)

        precision_dk = n_cor_repair_dk / max(n_repair_dk, 1)
        repair_recall = n_cor_repair_dk / max(detected_err, 1)


        def calc_rmse(df_filter):
            if df_filter.sum() == 0:
                return 0
            X_cor = df_res.loc[df_filter, '_value_'].apply(lambda arr: arr[0] if arr[0] != '_nan_' else 0.).values.astype(np.float)
            X_inferred = df_res.loc[df_filter, 'inferred_val'].values.astype(np.float)
            assert X_cor.shape == X_inferred.shape
            return np.sqrt(np.mean((X_cor - X_inferred) ** 2))

        # Numerical metrics (RMSE)
        rmse = 0
        rmse_dk = 0
        rmse_by_attr = {}
        rmse_dk_by_attr = {}
        if n_num:
            rmse = calc_rmse(~fil_cat)
            rmse_dk = calc_rmse(~fil_cat & fil_dk)
            for attr in self._numerical_attrs:
                fil_attr = df_res['attribute'] == attr
                rmse_by_attr[attr] = calc_rmse(fil_attr)
                rmse_dk_by_attr[attr] = calc_rmse(fil_attr & fil_dk)

        # Compile results
        val_res = {'n_cat': n_cat,
            'n_num': n_num,
            'n_cat_dk': n_cat_dk,
            'n_num_dk': n_num_dk,
            'total_err': total_err,
            'detected_err': detected_err,
            'n_repair': n_repair,
            'n_repair_dk': n_repair_dk,
            'sample_acc': sample_acc,
            'precision': precision,
            'recall': recall,
            'precision_dk': precision_dk,
            'repair_recall': repair_recall,
            'rmse': rmse,
            'rmse_dk': rmse_dk,
            'rmse_by_attr': rmse_by_attr,
            'rmse_dk_by_attr': rmse_dk_by_attr,
            }

        logging.debug("%s: # categoricals: (all) %d, (DK) %d",
                type(self).__name__, val_res['n_cat'], val_res['n_cat_dk'])
        logging.debug("%s: # numericals: (all) %d, (DK) %d",
                type(self).__name__, val_res['n_num'], val_res['n_num_dk'])

        logging.debug("%s: # of errors: %d, # of detected errors: %d",
                type(self).__name__, val_res['total_err'], val_res['detected_err'])

        logging.debug("%s: In-sample accuracy: %.3f",
                type(self).__name__, val_res['sample_acc'])

        logging.debug("%s: # repairs: (all) %d, (DK) %d",
                type(self).__name__, val_res['n_repair'], val_res['n_repair_dk'])

        logging.debug("%s: (Infer on all) Precision: %.3f, Recall: %.3f",
                type(self).__name__, val_res['precision'], val_res['recall'])
        logging.debug("%s: (Infer on DK) Precision: %.3f, Repair Recall: %.3f",
                type(self).__name__, val_res['precision_dk'], val_res['repair_recall'])

        if val_res['n_num']:
            logging.debug("%s: RMSE: (all) %f, (DK) %f", type(self).__name__,
                    val_res['rmse'], val_res['rmse_dk'])
            logging.debug("%s: RMSE per attr:", type(self).__name__)
            for attr in self._numerical_attrs:
                logging.debug("\t'%s': (all) %f, (DK) %f", attr,
                        val_res['rmse_by_attr'].get(attr, np.nan),
                        val_res['rmse_dk_by_attr'].get(attr, np.nan))

        return val_res

    def predict_pp_batch(self, df=None):
        """
        Performs batch prediction.

        df must have column '_vid_'.
        One should only pass in VIDs that have been trained on (see
        :param:`train_attrs`).

        Returns (vid, is_categorical, list[(value, proba)] OR real value (np.array))
            where if is_categorical = True then list[(value, proba)]  is returned.
        """
        if df is None:
            df = self.domain_df

        train_idx_to_attr = {idx: attr for attr, idx in self._dataset._train_attr_idxs.items()}
        n_cats, n_nums = 0, 0

        # Limit max batch size to prevent memory explosion.
        batch_sz = int(1e5 / self._embed_size)
        num_batches = math.ceil(df.shape[0] / batch_sz)
        logging.debug('%s: starting batched (# batches = %d, size = %d) prediction...',
                type(self).__name__, num_batches, batch_sz)
        self.set_mode(inference_mode=True)

        # No gradients required.
        with torch.no_grad():
            for vids, is_categorical, attr_idxs, \
                init_cat_idxs, init_numvals, init_nummasks, \
                domain_idxs, domain_masks, \
                target_numvals, cat_targets in tqdm(DataLoader(self._dataset, batch_size=batch_sz, sampler=IterSampler(df['_vid_'].values))):
                pred_cats, pred_nums = self.forward(is_categorical,
                        attr_idxs,
                        init_cat_idxs,
                        init_numvals,
                        init_nummasks,
                        domain_idxs,
                        domain_masks)

                pred_cat_idx = 0
                pred_num_idx = 0

                for idx, is_cat in enumerate(is_categorical.view(-1)):
                    vid = int(vids[idx, 0])
                    if is_cat:
                        logits = pred_cats[pred_cat_idx]
                        pred_cat_idx += 1
                        n_cats += 1
                        yield vid, bool(is_cat), zip(self._dataset.domain_values(vid), map(float, Softmax(dim=0)(logits)))
                        continue

                    # Real valued prediction

                    # Find the z-score and map it back to its actual value
                    attr = train_idx_to_attr[int(attr_idxs[idx,0])]
                    group_idx = self._dataset._train_num_attrs_group[attr].index(attr)
                    mean = self._dataset._num_attrs_mean[attr]
                    std = self._dataset._num_attrs_std[attr]
                    pred_num = float(pred_nums[pred_num_idx,group_idx]) * std + mean
                    pred_num_idx += 1
                    n_nums += 1
                    yield vid, False, pred_num

        self.set_mode(inference_mode=False)
        logging.debug('%s: done batch prediction on %d categorical and %d numerical VIDs.',
                type(self).__name__, n_cats, n_nums)
