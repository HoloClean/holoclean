import logging
import math
import os
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import Softmax, ReLU
from tqdm import tqdm
from dataset import AuxTables

class EmbeddingModel(torch.nn.Module):

    def __init__(self, attrs, embed_size, embedding_dataset, dropout_pct=0.):
        """
        :param attrs (list) a list of the trainable/learnable attributes
        :param embedding_dataset: (Dataset) embedding dataset
        :param embedding_dataset (Dataset) lookup dataset for the embedding model
        """

        torch.nn.Module.__init__(self)
        self.inference_mode = False
        self.attrs = attrs

        self._embed_size = embed_size

        # embedding dataset
        self._dataset = embedding_dataset

        assert dropout_pct < 1 and dropout_pct >= 0
        self.dropout_pct = dropout_pct

        ### Numerical attribute groups validation checks
        self._numerical_attr_groups = self._dataset.get_num_attr_groups()
        self._numerical_attrs = self._dataset.get_num_attr()

        # Memoize max domain size for numerical attribue for padding later.
        self.max_domain = self._dataset.max_domain
        self.max_cat_domain = self._dataset.max_cat_domain

        logging.debug('%s: max domain size: (categorical) %d, (numerical) %d',
                type(self).__name__,
                self.max_cat_domain,
                self.max_domain)

        self._train_cat_attrs = self._dataset._train_cat_attrs
        self._train_num_attrs = self._dataset._train_num_attrs

        # word2vec-like model.
        self._init_word2vec_model()
        
    def _init_word2vec_model(self):
        self._n_init_vals = self._dataset.n_init_vals
        self._n_train_vals = self._dataset.n_train_vals

        self._n_init_cat_attrs = self._dataset._n_init_cat_attrs
        self._n_init_num_attrs = self._dataset._n_init_num_attrs
        self._n_init_attrs = self._n_init_cat_attrs + self._n_init_num_attrs

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
        domain_idxs: (batch, max domain)
        domain_masks: (batch, max domain)

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

        is_categorical: (batch, 1)
        attr_idxs: (batch, 1)
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
            cat_logits = self._cat_forward(cat_combined_init, domain_idxs, domain_masks)

        pred_numvals = torch.empty(0, self._max_num_dim)
        if len(num_mask):
            num_combined_init, num_attr_idxs = combined_init[num_mask], \
                    self._num_attr_idxs(is_categorical, attr_idxs)
            # (# of num VIDs, max_num_dim)
            pred_numvals = self._num_forward(num_combined_init, num_attr_idxs)

        return cat_logits, pred_numvals, cat_mask, num_mask

    def _cat_num_masks(self, is_categorical):
        """
        is_categorical: (batch, 1)
        """
        # TODO: is_catrgorical is already ByteTensor: use torch.mask_tensor
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

    def _model_fpaths(self, prefix):
        return '%s_sdict.pkl' % prefix, '%s_ds_state.pkl' % prefix

    def get_embed_size(self):
        return self._embed_size

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
