import logging
import os
import numpy as np
import pandas as pd
import sys

import torch
from torch.nn import Softmax
from torch.nn import functional as F

from .featurizer import Featurizer
from dataset import AuxTables
from domain.estimators import TupleEmbedding


class EmbeddingFeaturizer(Featurizer):
    """
    Featurizer that wraps the learned vectors from a TupleEmbedding model
    cached in Dataset.

    Takes in two optional parameters (if estimator_type != TupleEmbedding i.e.
    if estimator is not the TupleEmbedding estimator):
        epochs (int): # of epochs to run
        batch_size (int): batch size
    """
    DEFAULT_EPOCHS = 10
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_DROPOUT = 0.
    DEFAULT_LR = 0.05
    DEFAULT_WEIGHT_LAMBDA = 0.

    def specific_setup(self):
        self.name = 'EmbeddingFeaturizer'

        if self.env['estimator_type'] != 'TupleEmbedding':
            logging.debug('%s: training TupleEmbedding model since it was not used as an estimator',
                    self.name)

            epochs = self.addn_kwargs.get('epochs', self.DEFAULT_EPOCHS)
            batch_size = self.addn_kwargs.get('batch_size', self.DEFAULT_BATCH_SIZE)
            dropout_pct = self.addn_kwargs.get('dropout_pct', self.DEFAULT_DROPOUT)
            weight_lambda = self.addn_kwargs.get('weight_lambda', self.DEFAULT_WEIGHT_LAMBDA)
            learning_rate = self.addn_kwargs.get('learning_rate', self.DEFAULT_LR)
            numerical_attr_groups = self.addn_kwargs.get('numerical_attr_groups', None)
            validate_fpath = self.addn_kwargs.get('validate_fpath', None)
            validate_tid_col = self.addn_kwargs.get('validate_tid_col', 'tid')
            validate_attr_col = self.addn_kwargs.get('validate_attr_col', 'attribute')
            validate_val_col = self.addn_kwargs.get('validate_val_col', 'correct_val')
            validate_epoch = self.addn_kwargs.get('validate_epoch', 1)

            logging.debug('%s: training with %d epochs and %d batch size',
                          self.name,
                          epochs,
                          batch_size)

            domain_df = self.ds.aux_table[AuxTables.cell_domain].df.sort_values('_vid_')
            self.embedding_model = TupleEmbedding(self.env, self.ds, domain_df,
                    numerical_attr_groups=numerical_attr_groups,
                    dropout_pct=dropout_pct,
                    learning_rate=learning_rate,
                    validate_fpath=validate_fpath,
                    validate_tid_col=validate_tid_col,
                    validate_attr_col=validate_attr_col,
                    validate_val_col=validate_val_col,
                    validate_epoch=validate_epoch)

            dump_prefix = self.addn_kwargs.get('dump_prefix', None)
            if dump_prefix is None \
                    or not self.embedding_model.load_model(dump_prefix):
                self.embedding_model.train(epochs, batch_size, weight_entropy_lambda=weight_lambda)

                if dump_prefix is not None:
                    self.embedding_model.dump_model(dump_prefix)

            logging.debug('%s: done training TupleEmbedding model.', self.name)
        else:
            logging.debug('%s: retrieving embedding vectors learned from TupleEmbedding estimator',
                    self.name)
            self.embedding_model = self.ds.get_embedding_model()

    def create_tensor(self):
        """
        For a batch of vids, returns a batch of softmax probabilities for each
        domain value learned by an embedding model.

        That is returns a (batch, max domain, 1) tensor.
        """
        domain_df = self.ds.aux_table[AuxTables.cell_domain].df.sort_values('_vid_').reset_index(drop=True)

        vids = domain_df['_vid_']

        # (# of vids, max_cat_domain), (# of vids, 1), (# of vids, 1)
        cat_probas, num_predvals, is_categorical = self.embedding_model.get_features(vids)

        # (# of vids, max domain)
        pad_len = self.embedding_model.max_domain - self.embedding_model.max_cat_domain
        if pad_len:
            # Pad last dimension on the right side with pad_len
            cat_probas = F.pad(cat_probas, pad=(0,pad_len), mode='constant', value=0.)

        # Create tensor for z-scored domain values
        num_attrs_idx = {attr: idx for idx, attr in enumerate(self.embedding_model._train_num_attrs)}
        domain_numvals = torch.zeros(len(vids), self.embedding_model.max_domain, len(num_attrs_idx))
        # Mask to mask out RMSE computed on padding outside of cell's domain.
        domain_mask = torch.zeros(len(vids), self.embedding_model.max_domain)
        for idx, (_, attr, domain, domain_sz) in enumerate(domain_df[['attribute', 'domain', 'domain_size']].to_records()):
            if domain_sz == 0:
                continue
            if is_categorical[idx,0] == 1.:
                continue
            dom_arr = np.array(domain.split('|||'), dtype=np.float32)
            mean = self.embedding_model._dataset._num_attrs_mean[attr]
            std = self.embedding_model._dataset._num_attrs_std[attr]
            dom_arr = (dom_arr - mean) / std
            domain_numvals[idx,:domain_sz, num_attrs_idx[attr]] = torch.FloatTensor(dom_arr)
            domain_mask[idx,:domain_sz] = 1.

        # (# of vids, max domain, # of num attrs)
        # This RMSE is between z-scored values. This is equivalent to dividing
        # the RMSE by std^2.
        num_rmse = torch.abs(num_predvals.unsqueeze(-1).expand(-1, self.embedding_model.max_domain, len(num_attrs_idx)) - domain_numvals)
        num_rmse.mul_(domain_mask.unsqueeze(-1).expand(-1, -1, len(num_attrs_idx)))

        # (# of vids, max domain, 1 + # num attrs)
        return torch.cat([cat_probas.unsqueeze(-1), num_rmse], dim=-1)



    def feature_names(self):
        return ["Embedding Cat Proba"]  + ["Embedding Num RMSE (%s)" % attr for attr in self.embedding_model._train_num_attrs]
