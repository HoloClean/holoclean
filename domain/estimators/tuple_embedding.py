import logging
import pickle
import sys

import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import CrossEntropyLoss, Softmax, MSELoss
import torch.nn.functional as F
from tqdm import tqdm

from dataset import AuxTables
from evaluate import EvalEngine
from ..estimator import Estimator
from utils import NULL_REPR

NUMERICAL_SEP = "|"

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
            numerical_attrs, neg_sample, memoize):
        """
        :param dataset: (Dataset) original dataset
        :param domain_df: (DataFrame) dataframe containing VIDs and their
            domain values we want to train on. VIDs not included in this
            dataframe will not be trained on e.g. if you only want to
            sub-select VIDs of certain attributes.
        :param numerical_attrs: (list[str]) attributes to treat as numerical.
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
        self._numerical_attrs = numerical_attrs or []
        if not all(attr in self._all_attrs for attr in self._numerical_attrs):
            logging.error('%s: numerical attributes %s must exist as a column in the data: %s',
                    type(self).__name__,
                    self._numerical_attrs,
                    self._all_attrs)
            sys.exit(1)
        # Attributes to derive context from
        self._init_cat_attrs, self._init_num_attrs = self._split_cat_num_attrs(self._all_attrs)
        self._n_init_cat_attrs, self._n_init_num_attrs = len(self._init_cat_attrs), len(self._init_num_attrs)
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

        self._raw_data = self.ds.get_raw_data().copy()

        # Indexes assigned to attributes: first categorical then numerical.
        self._init_attr_idxs = {attr: idx for idx, attr in enumerate(self._init_cat_attrs + self._init_num_attrs)}
        self._train_attr_idxs = {attr: idx for idx, attr in enumerate(self._train_cat_attrs + self._train_num_attrs)}

        # Assign indices for every categorical value for both init values
        # and training values.
        # Assign index for every unique value-attr (init values, input)
        self._init_val_idxs = {attr: {} for attr in self._init_cat_attrs}
        # Assign index for every unique value-attr (train/possible values, target)
        self._train_val_idxs = {attr: {} for attr in self._train_cat_attrs}

        self._train_idx_to_val = {0: NULL_REPR}
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

        # Number of dimensions for each numerical attribute
        self._num_attr_dim = {}
        for attr in self._init_num_attrs:
            attr_col = self._raw_data[attr]
            fil_notnull = attr_col != NULL_REPR
            fil_notnumeric = attr_col.str.contains('[^0-9.,]')

            notnull_attr_col = attr_col[fil_notnull]
            attr_dims = notnull_attr_col.str.split(NUMERICAL_SEP).apply(len)

            attr_dim = attr_dims.values[0]
            if not (attr_dims == attr_dim).all():
                logging.error('%s: values in attribute "%s" must be comma seprated with the same # of values (dimension) %d',
                        type(self).__name__,
                        attr,
                        attr_dim)
                sys.exit(1)

            bad_numerics = fil_notnull & fil_notnumeric
            if bad_numerics.sum():
                # Replace any values with non-numerical values with NULL
                self._raw_data.loc[fil_notnull & fil_notnumeric, attr] = NULL_REPR
                logging.warning('%s: replaced %d non-numerical values in attribute "%s" in RAW DATA as "%s" (NULL)',
                        type(self).__name__,
                        bad_numerics.sum(),
                        attr,
                        NULL_REPR)

            self._num_attr_dim[attr] = attr_dim

        cur_train_idx = 1
        for row in domain_df.to_records():
            val = row['init_value']
            attr = row['attribute']

            # Handle numerical attributes differently
            if attr in self._train_num_attrs:
                # Check init value in domain dataframe has the right
                # dimensions as the one detected in raw data
                if len(val.split(NUMERICAL_SEP)) != self._num_attr_dim[attr]:
                    logging.error('%s: value "%s" in domain for vid %d for attribute "%s" must have %d ,-separate values',
                            type(self).__name__,
                            val,
                            row['_vid_'],
                            attr,
                            self._num_attr_dim[attr])
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
            self._train_idx_to_val[cur_train_idx] = val
            cur_train_idx += 1

        # Unique train values (their indexes) by attr
        self._train_val_idxs_by_attr = {attr: torch.LongTensor([v for v in self._train_val_idxs[attr].values() if v != 0])
                for attr in self._train_cat_attrs}

        # Number of unique INIT attr-values
        self.n_init_vals = cur_init_idx
        self.n_train_vals = cur_train_idx

        self._raw_data_dict = self._raw_data.set_index('_tid_').to_dict('index')

        self._vid_to_idx = {vid: idx for idx, vid in enumerate(domain_df['_vid_'].values)}
        self._train_records = domain_df[['_tid_', 'attribute', 'init_value',
                                         'init_index', 'domain', 'domain_size', 'is_clean']].to_records()

        # Maximum domain size
        self._max_domain = int(domain_df['domain_size'].max())
        # Maximum dimension across all numerical attributes.
        self._max_num_dim = 0
        self._total_num_dim = 0
        if len(self._num_attr_dim):
            self._max_num_dim = max(self._num_attr_dim.values())
            self._total_num_dim = sum(self._num_attr_dim.values())


        # Dummy vectors
        self._dummy_domain_mask = torch.zeros(self._max_domain, dtype=torch.float)
        self._dummy_domain_idxs = torch.zeros(self._max_domain, dtype=torch.long)
        self._dummy_target_numvals = torch.zeros(self._max_num_dim, dtype=torch.float)
        self._dummy_cat_target = torch.LongTensor([-1])

        # Memoize certain lookups.
        if memoize:
            self._domain_idxs = self.MemoizeVec(len(self), torch.int64, self._max_domain)
            self._init_cat_idxs = self.MemoizeVec(len(self), torch.int64, self._n_init_cat_attrs)
            if self._total_num_dim > 0:
                self._target_numvals = self.MemoizeVec(len(self), torch.float32, self._max_num_dim)
                self._init_numvals = self.MemoizeVec(len(self), torch.float32, self._n_init_num_attrs, self._max_num_dim)
                self._init_nummask = self.MemoizeVec(len(self), torch.float32, self._n_init_num_attrs)
            self._neg_idxs = self.MemoizeVec(len(self), None, None)

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
        if not self.memoize or idx not in self._domain_idxs:
            cur = self._train_records[idx]
            assert cur['attribute'] in self._train_cat_attrs

            # Domain values and their indexes (softmax indexes)
            domain_idxs = torch.zeros(self._max_domain, dtype=torch.long)

            domain_idxs[:cur['domain_size']] = torch.LongTensor([self._train_val_idxs[cur['attribute']][val]
                    for val in cur['domain'].split('|||')])

            if not self.memoize:
                return domain_idxs
            self._domain_idxs[idx,0:len(domain_idxs)] = domain_idxs

        return self._domain_idxs[idx]

    def _get_target_numvals(self, idx):
        if not self.memoize or idx not in self._target_numvals:
            cur = self._train_records[idx]
            assert cur['attribute'] in self._train_num_attrs

            target_numvals = torch.zeros(self._max_num_dim, dtype=torch.float32)

            # We can skip this if we are in inference mode and the current
            # value is a nan value.
            if not (self.inference_mode and cur['init_value'] == NULL_REPR):
                target_numvals[:self._num_attr_dim[cur['attribute']]] = torch.FloatTensor(
                        np.array(cur['init_value'].split(NUMERICAL_SEP), dtype=np.float32))

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
            init_numvals: (n_init_num_attrs, max_num_dim),
            init_nummmask: (n_init_num_attrs),
            ).
        """
        if self._n_init_num_attrs == 0:
            return torch.empty(0, 0), torch.empty(0)

        if not self.memoize or idx not in self._init_numvals:
            cur = self._train_records[idx]

            init_numvals = torch.zeros(self._n_init_num_attrs, self._max_num_dim)
            init_nummask = torch.ones(self._n_init_num_attrs)
            for idx, attr in enumerate(self._init_num_attrs):
                val_str = self._raw_data_dict[cur['_tid_']][attr]
                if attr == cur['attribute'] or val_str == NULL_REPR:
                    init_nummask[idx] = 0.
                    continue

                attr_dim = self._num_attr_dim[attr]
                init_numvals[idx,:attr_dim] = torch.FloatTensor(np.float32(val_str.split(NUMERICAL_SEP)))

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
        # Add negative samples to a most likely correct (clean) cell
        if self._neg_sample and dom_size < self._max_domain and cur['is_clean']:
            # It is faster not to memoize these.
            neg_idxs = self._get_neg_dom_idxs(idx)
            neg_sample = torch.LongTensor(np.random.choice(neg_idxs,
                    size=min(len(neg_idxs), self._max_domain - dom_size),
                    replace=False))

            domain_idxs[dom_size:dom_size+len(neg_sample)] = neg_sample
            dom_size += len(neg_sample)

        # Position of init in domain values (target)
        target = cur['init_index']

        # Mask out non-relevant values from padding (see below)
        domain_mask = torch.zeros(self._max_domain, dtype=torch.float)
        domain_mask[dom_size:] = -1 * 1e9

        return domain_idxs, domain_mask, torch.LongTensor([target])

    def __getitem__(self, vid):
        """
        :param:`vid` is the desired VID.

        Returns (vid,
            is_categorical,
            attr_idx,
            init_cat_idxs,
            init_numvals,
            init_nummask,
            domain_idxs (if categorical),
            domain_mask (if categorical),
            target_numvals (if numerical),
            cat_target (if categorical),
            )

        where if VID is not categorical/numerical, then the corresponding
        vector are replaced with dummy vectors.

        target_numvals is 0-padded up to max_num_dim for concating.
        """
        idx = self._vid_to_idx[vid]
        cur = self._train_records[idx]

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
        return cur['domain'].split('|||')

    def dump_state(self):
        return {'init_val_idxs': self._init_val_idxs,
            'train_val_idxs': self._train_val_idxs,
            'init_attr_idxs': self._init_attr_idxs,
            'train_attr_idxs': self._train_attr_idxs,
            'init_cat_attrs': self._init_cat_attrs,
            'init_num_attrs': self._init_num_attrs,
            'train_cat_attrs': self._train_cat_attrs,
            'train_num_attrs': self._train_num_attrs,
            'num_attr_dim': self._num_attr_dim,
            }

class VidSampler(Sampler):
    def __init__(self, domain_df, shuffle=True, train_only_clean=True):
        # No NULLs and non-zero domain
        domain_df = domain_df[domain_df['init_value'] != NULL_REPR]

        # Train on only clean cells
        if train_only_clean:
            self._vids = domain_df.loc[domain_df['is_clean'], '_vid_']
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
    INIT_BIAS = 0.

    def __init__(self, env, dataset, domain_df,
            numerical_attrs=None,
            memoize=False,
            embed_size=10,
            neg_sample=True,
            validate_fpath=None, validate_tid_col=None, validate_attr_col=None,
            validate_val_col=None):
        """
        :param dataset: (Dataset) original dataset
        :param domain_df: (DataFrame) dataframe containing domain values
        :param numerical_attrs: (list[list[str]]) attributes/columns to treat as numerical.
            A list of column names. Each column must consist of d-separated
            values (for d-dimensional columns). For example the column
            2D column 'lat,long' must consist of all "123,456" values.

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

        train_attrs = self.env['train_attrs']

        # Check is train attributes exist
        if train_attrs is not None:
            if not all(attr in self.ds.get_attributes() for attr in train_attrs):
                logging.error('%s: all attributes specified to use for training %s must exist in dataset: %s',
                        type(self).__name__,
                        train_attrs,
                        self.ds.get_attributes())
                sys.exit(1)

        # Check is numerical attributes exist
        if numerical_attrs is not None:
            if not all(attr in self.ds.get_attributes() for attr in numerical_attrs):
                logging.error('%s: all numerical attributes specified %s must exist in dataset: %s',
                        type(self).__name__,
                        numerical_attrs,
                        self.ds.get_attributes())
                sys.exit(1)

        # Remove domain/training cells without a domain
        # TODO: relax for numerical
        filter_empty_domain = self.domain_df['domain_size'] == 0
        if filter_empty_domain.sum():
            logging.warning('%s: removing %d cells with empty domains',
                type(self).__name__,
                filter_empty_domain.sum())
            self.domain_df = self.domain_df[~filter_empty_domain]

        # Convert non numerical init values in numerical attributes with _nan_.
        if numerical_attrs is not None:
            fil_attr = self.domain_df['attribute'].isin(numerical_attrs)
            fil_notnull = self.domain_df['init_value'] != NULL_REPR
            fil_notnumeric = self.domain_df['init_value'].str.contains('[^0-9.,]')
            bad_numerics = fil_attr & fil_notnull & fil_notnumeric
            if bad_numerics.sum():
                self.domain_df.loc[bad_numerics, 'init_value'] = NULL_REPR
                logging.warning('%s: replaced %d non-numerical values in DOMAIN as "%s" (NULL)',
                        type(self).__name__,
                        bad_numerics.sum(),
                        NULL_REPR)

        # Add DK information to domain dataframe
        if self.ds.aux_table[AuxTables.dk_cells] is not None:
            df_dk = self.ds.aux_table[AuxTables.dk_cells].df
            self.domain_df = self.domain_df.merge(df_dk,
                    on=['_tid_', 'attribute'], how='left', suffixes=('', '_dk'))
            self.domain_df['is_clean'] = self.domain_df['_cid__dk'].isnull()
        else:
            self.domain_df['is_clean'] = True

            self.domain_df = self.domain_df[self.domain_df['attribute'].isin(train_attrs)]
        self.domain_recs = self.domain_df.to_records()

        # Dataset
        self._dataset = LookupDataset(env, dataset, self.domain_df,
                numerical_attrs, neg_sample, memoize)

        if len(self._dataset._num_attr_dim) and max(self._dataset._num_attr_dim.values()) > embed_size:
            logging.error("%s: maximum numeric value dimension %d must be <= embedding size %d",
                    type(self).__name__,
                    max(self._dataset._num_attr_dim.values()),
                    embed_size)
            sys.exit(1)

        self._train_cat_attrs = self._dataset._train_cat_attrs
        self._train_num_attrs = self._dataset._train_num_attrs
        self._train_idx_to_val = self._dataset._train_idx_to_val

        # word2vec-like model.

        self._n_init_vals = self._dataset.n_init_vals
        self._n_train_vals = self._dataset.n_train_vals

        self._n_init_cat_attrs = self._dataset._n_init_cat_attrs
        self._n_init_num_attrs = self._dataset._n_init_num_attrs
        self._n_init_attrs = self._n_init_cat_attrs + self._n_init_num_attrs

        self._n_train_cat_attrs = self._dataset._n_train_cat_attrs
        self._n_train_num_attrs = self._dataset._n_train_num_attrs
        self._n_train_attrs = self._n_train_cat_attrs + self._n_train_num_attrs

        self._max_domain = self._dataset._max_domain
        self._max_num_dim = self._dataset._max_num_dim
        self._embed_size = embed_size

        self.in_W = torch.nn.Parameter(torch.zeros(self._n_init_vals, self._embed_size))
        self.out_W = torch.nn.Parameter(torch.zeros(self._n_train_vals, self._embed_size))
        self.out_B = torch.nn.Parameter(torch.zeros(self._n_train_vals, 1))

        # Bases vectors for numerical attributes and their spans.

        # We have a learnable 0 zero vector for every numerical attribute.
        # We then have for each attribute of d dimension, d basis vectors where
        # d <= embed_size.
        #
        # Note X_num_bases is a (# of attrs, max dimension, embed size) tensor:
        #   if an attr has dimension < max dimension then the padded vectors
        #   will not be trained nor used.
        self.in_num_zero_vecs = torch.nn.Parameter(torch.zeros(self._n_init_num_attrs, self._embed_size))
        self.in_num_bases = torch.nn.Parameter(torch.zeros(self._n_init_num_attrs, self._embed_size, self._max_num_dim))
        self.out_num_zero_vecs = torch.nn.Parameter(torch.zeros(self._n_train_num_attrs, self._embed_size))
        self.out_num_bases = torch.nn.Parameter(torch.zeros(self._n_train_num_attrs, self._embed_size, self._max_num_dim))

        # Mask for _num_forward to restrict which dimensions are active for each attribute.
        self.out_num_masks = torch.zeros(self._n_train_num_attrs, self._max_num_dim, dtype=torch.float32)
        for idx, attr in enumerate(self._dataset._train_num_attrs):
            dim = self._dataset._num_attr_dim[attr]
            self.out_num_masks[idx,:dim] = 1.

        # logits fed into softmax used in weighted sum to combine
        # dot products of in_W and out_W per attribute.
        # Equivalent to choosing which input vectors to "focus" on.
        # Each row corresponds to the logits per each attr/column we want
        # to predict for and there are init_attr weights since we have
        # init attrs to combine.
        self.attr_W = torch.nn.Parameter(torch.zeros(self._n_train_attrs,
            self._n_init_attrs))

        # Initialize all but the first 0th vector embedding (reserved).
        torch.nn.init.xavier_uniform_(self.in_W[1:])
        torch.nn.init.xavier_uniform_(self.out_W[1:])
        torch.nn.init.xavier_uniform_(self.out_B[1:])

        if self._n_init_num_attrs > 0:
            torch.nn.init.xavier_uniform_(self.in_num_zero_vecs)
            torch.nn.init.xavier_uniform_(self.in_num_bases)
        if self._n_train_num_attrs > 0:
            torch.nn.init.xavier_uniform_(self.out_num_zero_vecs)
            torch.nn.init.xavier_uniform_(self.out_num_bases)

        torch.nn.init.xavier_uniform_(self.attr_W)

        self._cat_loss = CrossEntropyLoss()
        # TODO: we use MSE loss for all numerical attributes for now.
        # Allow user to pass in their desired loss.
        self._num_loss = MSELoss(reduction='mean')
        self._optimizer = Adam(self.parameters(), lr=self.env['learning_rate'], weight_decay=self.WEIGHT_DECAY)


        # Validation stuff
        self._do_validation = False
        if validate_fpath is not None \
            and validate_tid_col is not None \
            and validate_attr_col is not None \
            and validate_val_col is not None:
            eengine = EvalEngine(self.env, self.ds)
            eengine.load_data(self.ds.raw_data.name + '_tuple_embedding_validate', validate_fpath,
                tid_col=validate_tid_col,
                attr_col=validate_attr_col,
                val_col=validate_val_col)
            self._validate_df = self.domain_df.merge(eengine.clean_data.df,
                    left_on=['_tid_', 'attribute'], right_on=['_tid_', '_attribute_'])

            # Raise error if validation set has non-numerical values for numerical attrs
            if numerical_attrs is not None:
                fil_attr = self._validate_df['attribute'].isin(numerical_attrs)
                fil_notnull = self._validate_df['_value_'] != NULL_REPR
                fil_notnumeric = self._validate_df['_value_'].str.contains('[^0-9.,]')
                bad_numerics = fil_attr & fil_notnull & fil_notnumeric
                if bad_numerics.sum():
                    logging.error('%s: validation dataframe contains %d non-numerical values in numerical attrs %s',
                        type(self).__name__,
                        bad_numerics.sum(),
                        numerical_attrs)
                    sys.exit(1)

            self._validate_recs = self._validate_df[['_vid_', 'init_value', '_value_', 'is_clean']] \
                    .set_index('_vid_').to_dict('index')
            self._validate_total_errs = (self._validate_df['init_value'] != self._validate_df['_value_']).sum()
            self._validate_detected_errs = ((self._validate_df['init_value'] != self._validate_df['_value_']) & ~self._validate_df['is_clean']).sum()
            self._do_validation = True

    def _get_combined_init_vec(self, init_cat_idxs, init_numvals, init_nummasks, attr_idxs):
        """
        Constructs the "context vector" by combining the init embedding vectors.

        init_cat_idxs: (batch, n_init_cat_attrs)
        init_numvals: (batch, n_init_num_attrs, max_num_dim)
        init_nummasks: (batch, n_init_num_attrs)
        attr_idxs: (batch)
        """
        init_cat_vecs = torch.zeros(init_cat_idxs.shape[0], 0, self._embed_size)
        if self._n_init_cat_attrs:
            # (batch, n_init_cat_attrs, embed size)
            init_cat_vecs = self.in_W.index_select(0, init_cat_idxs.view(-1)).view(*init_cat_idxs.shape, self._embed_size)

        init_num_vecs = torch.zeros(init_numvals.shape[0], 0, self._embed_size)
        if self._n_init_num_attrs:
            # (batch, n_init_num_attrs, max_num_dim, 1)
            init_numvals = init_numvals.unsqueeze(-1)
            # in_num_bases is shape (n_init_num_attrs, embed_size, max_num_dim)
            # (batch, n_init_num_attrs, embed_size, max_num_dim)
            in_num_bases = self.in_num_bases.expand(init_numvals.shape[0], -1, -1, -1)

            # in_num_zero_vecs is shape (n_init_num_attrs, embed_size)
            # (batch, n_init_num_attrs, embed_size)
            init_num_vecs = in_num_bases.matmul(init_numvals).squeeze(-1) + self.in_num_zero_vecs.unsqueeze(0)
            # (batch, n_init_num_attrs, embed_size)
            init_num_vecs.mul_(init_nummasks.unsqueeze(-1))

        # (batch, n_init_attrs, embed size)
        init_vecs = torch.cat([init_cat_vecs, init_num_vecs], dim=1)
        # Scale vectors to unit norm ALONG the embedding dimension.
        # (batch, n_init_attrs, embed size)
        init_vecs = F.normalize(init_vecs, p=2, dim=2)

        # (batch, 1, n_init_attrs)
        attr_logits = self.attr_W.index_select(0, attr_idxs).unsqueeze(1)
        # (batch, 1, n_init_attrs)
        attr_weights = Softmax(dim=2)(attr_logits)

        # (batch, 1, embed size)
        combined_init = attr_weights.matmul(init_vecs)
        # (batch, embed size, 1)
        combined_init = combined_init.view(combined_init.shape[0], combined_init.shape[2], 1)

        # (batch, embed size, 1)
        return combined_init

    def _cat_forward(self, combined_init, domain_idxs, domain_masks, cat_targets):
        """
        combined_init: (batch, embed size, 1)
        domain_idxs: (batch, max domain)
        domain_masks: (batch, max domain)
        cat_targets: (batch)

        Returns logits: (batch, max domain)
        """
        # (batch, max domain, embed size)
        domain_vecs = self.out_W.index_select(0, domain_idxs.view(-1)).view(*domain_idxs.shape, self._embed_size)

        # (batch, max domain, 1)
        logits = domain_vecs.matmul(combined_init)

        # (batch, max domain, 1)
        domain_biases = self.out_B.index_select(0, domain_idxs.view(-1)).view(*domain_idxs.shape, 1)

        # (batch, max domain, 1)
        logits.add_(domain_biases)
        # (batch, max domain)
        logits = logits.squeeze(-1)


        # Init bias
        if self.INIT_BIAS != 0.:
            logits.scatter_add_(1, cat_targets.unsqueeze(-1),
                    self.INIT_BIAS * torch.ones_like(cat_targets.unsqueeze(-1), dtype=torch.float32))

        # Add mask to void out-of-domain indexes
        # (batch, max domain)
        logits.add_(domain_masks)

        return logits

    def _num_forward(self, combined_init, num_attr_idxs):
        """
        combined_init: (batch, embed size, 1)
        num_attr_idxs: (batch)

        Returns pred_numvals: (batch, max_num_dim)
        """
        # (batch, embed size, max num dim)
        out_num_bases = self.out_num_bases.index_select(0, num_attr_idxs)
        # (batch, max num dim)
        out_num_masks = self.out_num_masks.index_select(0, num_attr_idxs)

        """
        Use pseudo-inverse for regression
        """
        # (batch, embed size, max num dim)
        normed_out_num_bases = F.normalize(out_num_bases, p=2, dim=1)

        # mask out useless vectors
        # (batch, embed size, max num dim)
        normed_out_num_bases.mul_(out_num_masks.unsqueeze(1))

        # Project combined_init onto basis vectors:
        #   That is given normalized basis vectors v_1,...v_n the projection
        #   of vector c is
        #       c'  = \sum_{i=1}^n (c \cdot v_i) v_i
        # (batch, embed size, max num dim)
        # We perform the dot product by multiplying element-wise then
        # summing ALONG THE EMBEDDING DIMENSION.
        # We then project it back onto the normed basis vectors
        projected_inits = (combined_init * normed_out_num_bases).sum(dim=1, keepdim=True) * normed_out_num_bases
        # (batch, embed size, 1)
        projected_inits = projected_inits.sum(dim=2, keepdim=True)
        # Calculate pseudo-inverse of bases for all attributes in the current batch
        unique_num_attr_idxs = num_attr_idxs.unique(sorted=True)
        # (unique attrs, max_num_dim, embed size)
        pinverse_num_bases = torch.stack([self.out_num_bases[idx].pinverse() for idx in unique_num_attr_idxs])
        # We need to find which index each attr_idx corresponds to in
        # dim = 0 of inverse_num_bases.
        # e.g. given
        #     unique_num_attr_idxs = [2,5,1]
        #     num_attr_idxs = [1,5,2,5,1]
        # then temp = [[ 1, -3,  0, -3,  1],
        #         [ 4,  0,  3,  0,  4],
        #         [ 0, -4, -1, -4,  0]]
        temp = unique_num_attr_idxs.unsqueeze(1) - num_attr_idxs.unsqueeze(0)
        # e.g. temp = [[0, 0, 1],
        #         [0, 1, 0],
        #         [1, 0, 0],
        #         [0, 1, 0],
        #         [0, 0, 1]]
        temp = (temp == 0).t()
        assert temp.any(dim=1).all()
        # e.g. num_attr_idxs_pinverses = [2, 1, 0, 1, 2]
        # (batch)
        num_attr_idxs_pinverses = temp.nonzero()[:,1]
        # Apply (pseudo)-inverse to find real value again.
        # That is for an attribute with d dimension and embedding size k, d <= k
        # we have:
        #     V: k X d basis
        #     z: k X 1 zero vector
        #     r: d X 1 d-dimensional real value
        # We map r into a k-dimensional vector c by
        #     Vr + z = c
        # Now that we have c our k-dimensional context vector projected onto the span
        # of V, we recover r by:
        #     r = V^{-1}(c - z)
        # where V^{-1} is the pseudo-inverse.
        # (batch, max_num_dim, embed size)
        batch_pinverses = pinverse_num_bases.index_select(0, num_attr_idxs_pinverses)

        # (batch, embed size, 1)
        out_num_zero_vecs = self.out_num_zero_vecs.index_select(0, num_attr_idxs).unsqueeze(-1)

        # (batch, max_num_dim)
        pred_numvals = batch_pinverses.matmul(projected_inits - out_num_zero_vecs).squeeze(-1)

        # 0 out extraneous dimensions since extraneous targets are 0
        # (batch, max_num_dim)
        pred_numvals.mul_(out_num_masks)

        return pred_numvals

    def forward(self, is_categorical, attr_idxs,
                init_cat_idxs, init_numvals, init_nummasks,
                domain_idxs, domain_masks, cat_targets):
        """
        Performs one forward pass.
        """
        # (batch, embed size, 1)
        combined_init = self._get_combined_init_vec(init_cat_idxs, init_numvals, init_nummasks, attr_idxs)

        # (# of cat VIDs), (# of num VIDs)
        # TODO: convert is_categorical to ByteTensor and use torch.mask_tensor
        cat_mask, num_mask = is_categorical.nonzero().view(-1), (is_categorical == 0).nonzero().view(-1)

        cat_logits = torch.empty(0, self._max_domain)
        if len(cat_mask):
            cat_combined_init, domain_idxs, domain_masks, cat_targets = combined_init[cat_mask], \
                domain_idxs[cat_mask], \
                domain_masks[cat_mask], \
                cat_targets[cat_mask]
            # (# of cat VIDs, max_domain)
            cat_logits = self._cat_forward(cat_combined_init, domain_idxs, domain_masks, cat_targets)

        pred_numvals = torch.empty(0, self._max_num_dim)
        if len(num_mask):
            num_combined_init, num_attr_idxs = combined_init[num_mask], \
                attr_idxs[num_mask] - self._n_train_cat_attrs   # shift attribute indexes back to 0
            # (# of num VIDs, max_num_dim)
            pred_numvals = self._num_forward(num_combined_init, num_attr_idxs)

        return cat_logits, pred_numvals

    def train(self, num_epochs=10, batch_size=32, weight_entropy_lambda=0.,
            shuffle=True, train_only_clean=True):
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
            error detection
        """

        # Returns VIDs to train on.
        sampler = VidSampler(self.domain_df, shuffle=shuffle, train_only_clean=train_only_clean)

        logging.debug("%s: training (lambda = %f) on %d cells (%d cells in total) in:\n1) %d categorical columns: %s\n2) %d numerical columns: %s",
                      type(self).__name__,
                      weight_entropy_lambda,
                      len(sampler),
                      self.domain_df.shape[0],
                      self._n_train_cat_attrs,
                      self._train_cat_attrs,
                      self._n_train_num_attrs,
                      self._train_num_attrs)

        batch_losses = []
        # Main training loop.
        for epoch_idx in range(1, num_epochs+1):
            logging.debug('%s: epoch %d of %d', type(self).__name__, epoch_idx, num_epochs)
            batch_cnt = 0
            for vids, is_categorical, attr_idxs, \
                init_cat_idxs, init_numvals, init_nummasks, \
                domain_idxs, domain_masks, \
                target_numvals, cat_targets \
                in tqdm(DataLoader(self._dataset, batch_size=batch_size, sampler=sampler)):
                is_categorical = is_categorical.view(-1)
                attr_idxs = attr_idxs.view(-1)
                cat_targets = cat_targets.view(-1)

                cat_preds, numval_preds = self.forward(is_categorical, attr_idxs,
                        init_cat_idxs, init_numvals, init_nummasks,
                        domain_idxs, domain_masks, cat_targets)

                # Select out the appropriate targets
                cat_targets = cat_targets[is_categorical.nonzero().view(-1)]
                target_numvals = target_numvals[(is_categorical == 0).nonzero().view(-1)]

                assert cat_preds.shape[0] == cat_targets.shape[0]
                assert numval_preds.shape == target_numvals.shape

                batch_loss = 0.
                if cat_targets.shape[0] > 0:
                    batch_loss += self._cat_loss(cat_preds, cat_targets)
                if target_numvals.shape[0] > 0:
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
                batch_cnt += 1

            logging.debug('%s: average batch loss: %f',
                    type(self).__name__,
                    sum(batch_losses[-1 * batch_cnt:]) / batch_cnt)

        return batch_losses

    def dump_model(self, prefix):
        """
        Dump this model's parameters and other metadata (e.g. attr-val to corresponding
        index in embedding matrix) with the given :param:`prefix`.
        """
        torch.save(self.state_dict(), '%s_sdict.pkl' % prefix)
        pickle.dump(self._dataset.dump_state(), open('%s_ds_state.pkl' % prefix, 'wb'))

    def dump_predictions(self, prefix):
        """
        Dump inference results to ":param:`prefix`_predictions.pkl".
        """
        preds = self.predict_pp_batch()

        logging.debug('%s: constructing and dumping predictions...'. type(self).__name__)
        results = []
        for ((vid, is_cat, pred), row) in zip(preds, self.domain_recs):
           assert vid == row['_vid_']
           if is_cat:
               max_val, max_proba = max(pred, key=lambda t: t[1])
               results.append({'tid': row['_tid_'],
                   'vid': vid,
                   'attribute': row['attribute'],
                   'inferred_val': max_val,
                   'proba': max_proba})
           else:
               num_pred = NUMERICAL_SEP.join(map(str, pred))
               results.append({'tid': row['_tid_'],
                   'vid': vid,
                   'attribute': row['attribute'],
                   'inferred_val': num_pred,
                   'proba': -1})

        results = pd.DataFrame(results)

        fpath = '{}_predictions.pkl'.format(prefix)
        logging.debug('%s: dumping predictions to %s', type(self).__name__, fpath)
        results.to_pickle(fpath)

    def validate(self):
        ### Categorical
        n_cat = 0
        n_cat_repair = 0
        # repairs on clean + DK cells
        cor_repair = 0
        incor_repair = 0
        # repairs only on DK cells
        cor_repair_dk = 0
        incor_repair_dk = 0


        ### Numerical
        n_num = 0
        total_se = 0
        squared_resids = []

        logging.debug('%s: running validation set...', type(self).__name__)

        validation_preds = self.predict_pp_batch(self._validate_df)

        for vid, is_cat, preds in tqdm(list(validation_preds)):
            row = self._validate_recs[vid]

            if is_cat:
                n_cat += 1

                inf_val, inf_prob = max(preds, key=lambda t: t[1])

                if row['init_value'] != inf_val:
                    n_cat_repair += 1
                    # Correct val == inf val
                    if row['_value_'] == inf_val:
                        cor_repair += 1
                        if not row['is_clean']:
                            cor_repair_dk += 1
                    # Correct val != inf val
                    else:
                        incor_repair += 1
                        if not row['is_clean']:
                            incor_repair_dk += 1
                continue

            # Numerical
            n_num += 1
            cor_val = np.array(row['_value_'].split(NUMERICAL_SEP), dtype=np.float32)
            assert cor_val.shape == preds.shape
            sq_res = np.sum((cor_val - preds) ** 2)
            squared_resids.append(sq_res)
            total_se += sq_res

        if self._validate_total_errs == 0:
            logging.warning('%s: total errors in validation set is 0', type(self).__name__)
        if self._validate_detected_errs == 0:
            logging.warning('%s: total detected errors in validation set is 0', type(self).__name__)

        val_res = {'precision': cor_repair / max(cor_repair + incor_repair, 1),
            'recall': cor_repair / max(self._validate_total_errs, 1),
            'dk_precision': cor_repair_dk / max(cor_repair_dk + incor_repair_dk, 1),
            'repair_recall': cor_repair_dk / max(self._validate_detected_errs, 1),
            # 'dk_recall': cor_repair_dk / max(self._validate_total_errs, 1),
            'n_cat': n_cat,
            'n_num': n_num,
            'n_cat_repair': n_cat_repair,
            'total_se': total_se,
            'squared_resids': pd.Series(squared_resids),
            }

        logging.debug("%s: # categorical: %d (repairs: %d), # numerical: %d",
                type(self).__name__, val_res['n_cat'], val_res['n_cat_repair'],
                val_res['n_num'])
        logging.debug("%s: (Infer on all) Precision: %.2f, Recall: %.2f",
                type(self).__name__, val_res['precision'], val_res['recall'])
        logging.debug("%s: (Infer only on DK) Precision: %.2f, Repair Recall: %.2f",
                type(self).__name__, val_res['dk_precision'], val_res['repair_recall'])
        if val_res['n_num']:
            logging.debug("%s: MSE: %.2f, RMSE: %2.f", type(self).__name__,
                    val_res['total_se'] / val_res['n_num'],
                    (val_res['total_se'] / val_res['n_num']) ** 0.5)
            logging.debug("%s: Squared resids: %s", type(self).__name__,
                    val_res['squared_resids'].describe())

        return val_res

    def predict_pp(self, row, attr=None, values=None):
        raise NotImplementedError

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

        logging.debug('%s: getting dataset indices...', type(self).__name__)
        self._dataset.set_mode(inference_mode=True)
        ds_tuples = [self._dataset[vid] for vid in df['_vid_'].values]
        self._dataset.set_mode(inference_mode=False)
        vids, is_categorical, attr_idxs, \
            init_cat_idxs, init_numvals, init_nummasks, \
            domain_idxs, domain_masks, \
            target_numvals, cat_targets = map(torch.stack, list(zip(*ds_tuples)))
        is_categorical = is_categorical.view(-1)
        attr_idxs = attr_idxs.view(-1)
        cat_targets = cat_targets.view(-1)
        logging.debug('%s: done getting dataset indices.', type(self).__name__)

        logging.debug('%s: starting batch prediction...', type(self).__name__)
        pred_cats, pred_nums = self.forward(is_categorical, attr_idxs,
                init_cat_idxs, init_numvals, init_nummasks,
                domain_idxs, domain_masks, cat_targets)
        logging.debug('%s: done batch prediction on %d categorical and %d numerical VIDs.',
                type(self).__name__, pred_cats.shape[0], pred_nums.shape[0])

        pred_cat_idx = 0
        pred_num_idx = 0
        for idx, is_cat in enumerate(is_categorical):
            vid = int(vids[idx, 0])
            if is_cat:
                logits = pred_cats[pred_cat_idx]
                pred_cat_idx += 1
                yield vid, bool(is_cat), zip(self._dataset.domain_values(vid), map(float, Softmax(dim=0)(logits)))
                continue

            # Real valued prediction
            dim = self.out_num_masks[attr_idxs[idx] - self._n_train_cat_attrs].nonzero().max()
            pred_num = pred_nums[pred_num_idx][:dim+1]
            pred_num_idx += 1
            yield vid, False, pred_num.detach().numpy()
