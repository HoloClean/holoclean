import logging
import os
import pandas as pd
import sys

import torch
from torch.nn import Softmax

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
    DEFAULT_LR = 0.05

    def specific_setup(self):
        self.name = 'EmbeddingFeaturizer'

        if self.env['estimator_type'] != 'TupleEmbedding':
            logging.debug('%s: training TupleEmbedding model since it was not used as an estimator',
                    self.name)

            epochs = self.addn_kwargs.get('epochs', self.DEFAULT_EPOCHS)
            batch_size = self.addn_kwargs.get('epochs', self.DEFAULT_BATCH_SIZE)
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
                    learning_rate=learning_rate,
                    validate_fpath=validate_fpath,
                    validate_tid_col=validate_tid_col,
                    validate_attr_col=validate_attr_col,
                    validate_val_col=validate_val_col,
                    validate_epoch=validate_epoch)

            dump_prefix = self.addn_kwargs.get('dump_prefix', None)
            if dump_prefix is None \
                    or not self.embedding_model.load_model(dump_prefix):
                self.embedding_model.train(epochs, batch_size)

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
        vids = self.ds.aux_table[AuxTables.cell_domain].df['_vid_'].sort_values()

        # (# of vids, embed_size)
        context_vecs = self.embedding_model.get_context_vecs(vids)
        # (# of vids, max domain, embed_size + 1)
        target_vecs = self.embedding_model.get_target_vecs(vids)
        # (# of vids, max domain, embed_size)
        context_vecs = context_vecs.unsqueeze(1).expand(-1, target_vecs.shape[1], -1)

        # Verify the non-zero target_vectors correspond to the # of domain
        # values actually in each VID.
        assert ((target_vecs != 0).all(dim=2).sum(dim=1).detach().numpy()
                == self.ds.aux_table[AuxTables.cell_domain].df.sort_values('_vid_')['domain_size'].values).all()

        # (# of vids, max domain, 1)
        logits = (target_vecs[:,:,:-1] * context_vecs).sum(dim=-1, keepdim=True) + target_vecs[:,:,-1:]
        # Logits without an actual domain value needs to be negative large number
        logits[(target_vecs[:,:,:-1] == 0.).all(dim=-1, keepdim=True)] = -1e9

        # (# of vids, max domain, 1)
        probs = Softmax(dim=1)(logits)
        return probs



    def feature_names(self):
        return ["Embedding_Proba"]
