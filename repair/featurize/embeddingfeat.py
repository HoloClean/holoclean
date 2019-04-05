import logging
import sys

import torch

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

    def specific_setup(self):
        self.name = 'EmbeddingFeaturizer'

        if self.env['estimator_type'] != 'TupleEmbedding':
            logging.debug('%s: training TupleEmbedding model since it was not used as an estimator',
                    self.name)

            epochs = self.addn_kwargs.get('epochs', self.DEFAULT_EPOCHS)
            batch_size = self.addn_kwargs.get('epochs', self.DEFAULT_BATCH_SIZE)

            domain_df = self.ds.aux_table[AuxTables.cell_domain].df.sort_values('_vid_')
            self.embedding_model = TupleEmbedding(self.env, self.ds, domain_df)
            self.embedding_model.train(epochs, batch_size)

            logging.debug('%s: done training TupleEmbedding model.')
        else:
            logging.debug('%s: retrieving embedding vectors learned from TupleEmbedding estimator',
                    self.name)
            self.embedding_model = self.ds.get_embedding_model()

    def create_tensor(self):
        """
        For a batch of vids, returns a batch of learned embeddings for each of its
        domain values.

        That is returns a (batch, max domain, 2 * embed_size) tensor where
        the first embed_size features are the context vectors (same for a given
        batch) and the second embed_size features are the "target" vectors (unique
        for each domain value).
        """
        vids = self.ds.aux_table[AuxTables.cell_domain].df['_vid_'].sort_values()

        # (# of vids, embed_size)
        context_vecs = self.embedding_model.get_context_vecs(vids)
        # (# of vids, max domain, embed_size)
        target_vecs = self.embedding_model.get_target_vecs(vids)
        # (# of vids, max domain, embed_size)
        context_vecs = context_vecs.unsqueeze(1).expand(*target_vecs.shape)

        # Verify the non-zero target_vectors correspond to the # of domain
        # values actually in each VID.
        assert ((target_vecs != 0).all(dim=2).sum(dim=1).detach().numpy() \
                == self.ds.aux_table[AuxTables.cell_domain].df.sort_values('_vid_')['domain_size'].values).all()

        # (# of vids, max domain, 2 * embed_size)
        combined_vecs = torch.cat([context_vecs, target_vecs], dim=2)

        return combined_vecs

    def feature_names(self):
        return ["Context_%d" % idx for idx in range(self.env['estimator_embedding_size'])] \
            + ["Target_%d" % idx for idx in range(self.env['estimator_embedding_size'])]
