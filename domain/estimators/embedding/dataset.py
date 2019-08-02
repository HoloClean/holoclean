import torch
import logging
import numpy as np
import pandas as pd
from utils import NULL_REPR
from torch.utils.data import Dataset

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
                raise IndexError(
                    "tried to access un-set value at index %d" % idx)
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
        self._numerical_attrs = self._verify_numerical_attr_groups(self.ds, numerical_attr_groups)
        self._numerical_attr_groups = numerical_attr_groups or []

        # Attributes to derive context from
        self._init_cat_attrs, self._init_num_attrs = self._split_cat_num_attrs(
            self._all_attrs)
        self._n_init_cat_attrs, self._n_init_num_attrs = len(
            self._init_cat_attrs), len(self._init_num_attrs)
        logging.debug('%s: init categorical attributes: %s',
                      type(self).__name__,
                      self._init_cat_attrs)
        logging.debug('%s: init numerical attributes: %s',
                      type(self).__name__,
                      self._init_num_attrs)

        # Attributes to train on (i.e. target columns).
        self._train_attrs = sorted(domain_df['attribute'].unique())
        assert all(attr in self._all_attrs for attr in self._train_attrs)
        self._train_cat_attrs, self._train_num_attrs = self._split_cat_num_attrs(
            self._train_attrs)
        self._n_train_cat_attrs, self._n_train_num_attrs = len(
            self._train_cat_attrs), len(self._train_num_attrs)
        logging.debug('%s: train categorical attributes: %s',
                      type(self).__name__,
                      self._train_cat_attrs)
        logging.debug('%s: train numerical attributes: %s',
                      type(self).__name__,
                      self._train_num_attrs)

        # Make copy of raw data
        self._raw_data = self.ds.get_raw_data().copy()
        # Keep track of mean + std to un-normalize during prediction
        self._num_attrs_mean = {}
        self._num_attrs_std = {}
        # Mean-0 variance 1 normalize all numerical attributes in the raw data
        for num_attr in self._init_num_attrs:
            temp = self._raw_data[num_attr].copy()
            fil_notnull = temp != NULL_REPR
            self._num_attrs_mean[num_attr] = temp[fil_notnull].astype(
                np.float).mean(axis=0)
            self._num_attrs_std[num_attr] = temp[fil_notnull].astype(
                np.float).std(axis=0)
            temp[fil_notnull] = ((temp[fil_notnull].astype(np.float)
                                  - self._num_attrs_mean[num_attr])
                                 / (self._num_attrs_std[num_attr] or 1.)).astype(str)
            self._raw_data[num_attr] = temp

        # Indexes assigned to attributes: first categorical then numerical.
        self._init_attr_idxs = {attr: idx for idx, attr in enumerate(
            self._init_cat_attrs + self._init_num_attrs)}
        self._train_attr_idxs = {attr: idx for idx, attr in enumerate(
            self._train_cat_attrs + self._train_num_attrs)}

        # Assign indices for every categorical value for both init values
        # and training values.
        # Assign index for every unique value-attr (init values, input)
        self._init_val_idxs = {attr: {} for attr in self._init_cat_attrs}
        # Assign index for every unique value-attr (train/possible values, target)
        self._train_val_idxs = {attr: {} for attr in self._train_cat_attrs}

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

        self._raw_data_dict = self._raw_data.set_index(
            '_tid_').to_dict('index')

        self._vid_to_idx = {vid: idx for idx,
                            vid in enumerate(domain_df['_vid_'].values)}
        self._train_records = domain_df[['_vid_', '_tid_', 'attribute', 'init_value',
                                         'init_index',
                                         'weak_label',
                                         'weak_label_idx', 'fixed',
                                         'domain', 'domain_size', 'is_clean']].to_records()

        # max_domain_size
        self.max_domain = domain_df['domain_size'].max()

        # Maximum domain size: we don't use the domain of numerical attributes
        # so we can discard them.
        self.max_cat_domain = domain_df.loc[domain_df['attribute'].isin(
            self._train_cat_attrs), 'domain_size'].max()
        if pd.isna(self.max_cat_domain):
            self.max_cat_domain = 0
        # Maximum dimension across all numerical attributes.
        self._max_num_dim = max(
            list(map(len, self._numerical_attr_groups)) or [0])

        self._init_dummies()
        self._init_memoize_vecs()

    def _init_dummies(self):
        # Dummy vectors
        self._dummy_domain_mask = torch.zeros(self.max_cat_domain,
                                              dtype=torch.float)
        self._dummy_domain_idxs = torch.zeros(self.max_cat_domain,
                                              dtype=torch.long)
        self._dummy_target_numvals = torch.zeros(self._max_num_dim,
                                                 dtype=torch.float)
        self._dummy_cat_target = torch.LongTensor([-1])

    def _init_memoize_vecs(self):
        # Memoize certain lookups.
        if not self.memoize:
            return
        self._domain_idxs = self.MemoizeVec(
            len(self), torch.int64, self.max_cat_domain)
        self._init_cat_idxs = self.MemoizeVec(
            len(self), torch.int64, self._n_init_cat_attrs)
        self._neg_idxs = self.MemoizeVec(len(self), None, None)

        if self._max_num_dim > 0:
            self._target_numvals = self.MemoizeVec(
                len(self), torch.float32, self._max_num_dim)
            self._init_numvals = self.MemoizeVec(
                len(self), torch.float32, self._n_init_num_attrs)
            self._init_nummask = self.MemoizeVec(
                len(self), torch.float32, self._n_init_num_attrs)

    def _split_cat_num_attrs(self, attrs):
        """
        Splits a list of attributes into their categorical and numerical groupings.
        """
        cat_attrs = [
            attr for attr in attrs if attr not in self._numerical_attrs]
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
                                                     self._get_domain_idxs(
                                                         idx),
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
            self._domain_idxs[idx, 0:len(domain_idxs)] = domain_idxs

        return self._domain_idxs[idx]

    def _get_target_numvals(self, idx):
        if not self.memoize or idx not in self._target_numvals:
            cur = self._train_records[idx]
            assert cur['attribute'] in self._train_num_attrs

            target_numvals = torch.zeros(
                self._max_num_dim, dtype=torch.float32)

            # Get the target values for this numerical group.
            attr_group = self._train_num_attrs_group[cur['attribute']]
            target_val_strs = [self._raw_data_dict[cur['_tid_']][attr]
                               for attr in attr_group]

            # We can skip this if we are in inference mode and any of the
            # target/current values in the numerical group are NULL.
            if not (self.inference_mode and any(val == NULL_REPR for val in target_val_strs)):
                target_numvals[:len(attr_group)] = torch.FloatTensor(
                    np.array(target_val_strs, dtype=np.float32))

            if not self.memoize:
                return target_numvals
            self._target_numvals[idx, 0:len(target_numvals)] = target_numvals

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

            init_cat_idxs = torch.LongTensor([self._init_val_idxs[attr][self._raw_data_dict[cur['_tid_']][attr]]
                                              if attr != cur['attribute'] else 0
                                              for attr in self._init_cat_attrs])

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

            init_numvals = torch.zeros(
                self._n_init_num_attrs, dtype=torch.float32)
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
                                                           size=min(
                                                               len(neg_idxs), self.max_cat_domain - dom_size),
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
            cat_target (if categorical): (1))

        where if VID is not categorical/numerical, then the corresponding
        vector are replaced with dummy vectors.

        target_numvals is 0-padded up to max_num_dim for concating.
        """
        idx = self._vid_to_idx[vid]
        cur = self._train_records[idx]
        assert cur['_vid_'] == vid

        is_categorical = torch.ByteTensor(
            [int(cur['attribute'] in self._train_cat_attrs)])
        attr_idx = torch.LongTensor([self._train_attr_idxs[cur['attribute']]])
        init_cat_idxs = self._get_init_cat_idxs(idx)
        init_numvals, init_nummask = self._get_init_numvals(idx)

        vid = torch.LongTensor([vid])

        # NB: we always return tensors so we can stack them easily even if
        # we index into dataset one-by-one.

        # Categorical VID
        if cur['attribute'] in self._train_cat_attrs:
            domain_idxs, domain_mask, target = self._get_cat_domain_target(idx)
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

    def get_num_attr(self):
        return self._numerical_attrs

    def get_num_attr_groups(self):
        return self._numerical_attr_groups

    def load_state(self, state):
        for attr, val in state.items():
            setattr(self, attr, val)
        self._init_dummies()
        self._init_memoize_vecs()

    def _verify_numerical_attr_groups(self, dataset, numerical_attr_groups):
        """
        Verify numerical attribute groups are disjoint and exist
        Returns a list of the individual attributes.
        """
        numerical_attrs = []
        # Check if numerical attributes exist and are disjoint
        if numerical_attr_groups is not None:
            numerical_attrs = [
                attr for group in numerical_attr_groups for attr in group]

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
    
