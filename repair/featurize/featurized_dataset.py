from collections import namedtuple
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import AuxTables, CellStatus

FeatInfo = namedtuple('FeatInfo', ['name', 'size', 'learnable', 'init_weight', 'feature_names'])
BatchedFeaturizedDataset = namedtuple('BatchedFeaturizedDataset', ['X', 'Y', 'var_mask'])


class FeaturizedDataset:
    def __init__(self, dataset, env, featurizers):
        self.ds = dataset
        self.env = env
        self.total_vars, self.classes = self.ds.get_domain_info()
        self.processes = self.env['threads']
        self.featurizers = featurizers
        for f in self.featurizers:
            f.setup_featurizer(self.ds, self.total_vars, self.classes, self.processes, self.env['batch_size'])
        # logging.debug('featurizing training data...')
        # tensors = [f.create_tensor() for f in featurizers]
        self.featurizer_info = [FeatInfo(featurizer.name,
                                         featurizer.num_features(),
                                         featurizer.learnable,
                                         featurizer.init_weight,
                                         featurizer.feature_names())
                                for featurizer in self.featurizers]
        # tensor = torch.cat(tensors, 2)
        # self.tensor = tensor
        #
        # logging.debug('DONE featurization.')

        # Is this still necessary since we have no initial weights for all examples?
        # if self.env['debug_mode']:
        #     weights_df = pd.DataFrame(self.tensor.reshape(-1, self.tensor.shape[-1]).numpy())
        #     weights_df.columns = ["{}::{}".format(f.name, featname) for f in featurizers for featname in f.feature_names()]
        #     weights_df.insert(0, 'vid', np.floor_divide(np.arange(weights_df.shape[0]), self.tensor.shape[1]) + 1)
        #     weights_df.insert(1, 'val_idx', np.tile(np.arange(self.tensor.shape[1]), self.tensor.shape[0]))
        #     weights_df.to_pickle('debug/{}_train_features.pkl'.format(self.ds.id))

        # TODO: remove after we validate it is not needed.
        self.in_features = sum(featurizer.size for featurizer in self.featurizer_info)
        logging.debug("generating weak labels...")
        self.weak_labels, self.is_clean = self.generate_weak_labels()
        logging.debug("DONE generating weak labels.")
        logging.debug("generating mask...")
        self.var_class_mask, self.var_to_domsize = self.generate_var_mask()
        logging.debug("DONE generating mask.")

    def generate_weak_labels(self):
        """
        generate_weak_labels returns a tensor where for each VID we have the
        domain index of the initial value.

        :return: Torch.Tensor of size (# of variables) X 1 where tensor[i][0]
            contains the domain index of the initial value for the i-th
            variable/VID.
        """
        # Trains with clean cells AND cells that have been weak labelled.
        query = 'SELECT _vid_, weak_label_idx, fixed, (t2._cid_ IS NULL) AS clean ' \
                'FROM {} AS t1 LEFT JOIN {} AS t2 ON t1._cid_ = t2._cid_ ' \
                'WHERE t2._cid_ is NULL ' \
                '   OR t1.fixed != {};'.format(AuxTables.cell_domain.name,
                                               AuxTables.dk_cells.name,
                                               CellStatus.NOT_SET.value)
        res = self.ds.engine.execute_query(query)
        if len(res) == 0:
            raise Exception("No weak labels available. Reduce pruning threshold.")
        labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        is_clean = torch.zeros(self.total_vars, 1).type(torch.LongTensor)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            label = int(tuple[1])
            fixed = int(tuple[2])
            clean = int(tuple[3])
            labels[vid] = label
            is_clean[vid] = clean
        return labels, is_clean

    def generate_var_mask(self):
        """
        generate_var_mask returns a mask tensor where invalid domain indexes
        for a given variable/VID has value -10e6.

        An invalid domain index is possible since domain indexes are expanded
        to the maximum domain size of a given VID: e.g. if a variable A has
        10 unique values and variable B has 6 unique values, then the last
        4 domain indexes (index 6-9) of variable B are invalid.

        :return: Torch.Tensor of size (# of variables) X (max domain)
            where tensor[i][j] = 0 iff the value corresponding to domain index 'j'
            is valid for the i-th VID and tensor[i][j] = -10e6 otherwise.
        """
        var_to_domsize = {}
        query = 'SELECT _vid_, domain_size FROM %s' % AuxTables.cell_domain.name
        res = self.ds.engine.execute_query(query)
        mask = torch.zeros(self.total_vars,self.classes)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            max_class = int(tuple[1])
            mask[vid, max_class:] = -10e6
            var_to_domsize[vid] = max_class
        return mask, var_to_domsize

    def get_training_data(self):
        train_idx = (self.weak_labels != -1).nonzero()[:,0]
        return TorchFeaturizedDataset(
            vids=train_idx,
            featurizers=self.featurizers,
            Y=self.weak_labels,
            var_mask=self.var_class_mask,
            batch_size=self.env['batch_size'],
            feature_norm =self.env['feature_norm']
        )

    def get_infer_data(self):
        infer_idx = (self.is_clean == 0).nonzero()[:, 0]
        return TorchFeaturizedDataset(
            vids=infer_idx,
            featurizers=self.featurizers,
            Y=self.weak_labels,
            var_mask=self.var_class_mask,
            batch_size=self.env['batch_size'],
            feature_norm =self.env['feature_norm']
        ), infer_idx

class TorchFeaturizedDataset(torch.utils.data.Dataset):
    def __init__(self, vids, featurizers, Y, var_mask, batch_size, feature_norm):
        self.vids = vids
        self.featurizers = featurizers
        self.Y = Y
        self.var_mask = var_mask
        self.batch_size = batch_size
        self.feature_norm = feature_norm
        self.num_examples = len(self.vids)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        X = torch.cat([featurizer.gen_feat_tensor(self.vids[idx]) for featurizer in self.featurizers],1)
        if self.feature_norm:
            logging.debug("normalizing features...")
            # normalize within each cell the features
            self.tensor = F.normalize(self.tensor, p=2, dim=1)
            logging.debug("DONE feature normalization.")

        Y = self.Y[self.vids[idx]]
        var_mask = self.var_mask[self.vids[idx]]
        return BatchedFeaturizedDataset(X, Y, var_mask)
#
#     # def reset(self, get_labels):
#     #     self.vids = np.random.permutation(vids)
#     #     self.get_labels = get_labels
#     #     self.curr_batch = 0
#
#     def next(self):
#         if self.curr_batch >= self.batch_size:
#             raise StopIteration
#         self.curr_batch += 1
#         start_ind, end_ind = (self.curr_batch - 1) * self.batch_size, min(len(self.vids), self.curr_batch * self.batch_size)
#         vids_to_featurize = self.vids[start_ind:end_ind]
#         X = [featurizer.gen_feat_tensor_for_vids(vids_to_featurize) for featurizer in self.featurizers]
#         tensor = torch.cat(X, 2)
#         if self.feature_norm:
#             logging.debug("normalizing features...")
#             # normalize within each cell the features
#             self.tensor = F.normalize(self.tensor, p=2, dim=1)
#             logging.debug("DONE feature normalization.")
#         if self.get_labels:
#             Y = self.Y.index_select(0, vids_to_featurize)
#         var_mask = self.var_mask.index_select(0, vids_to_featurize)
#         return BatchedFeaturizedDataset(X, Y, var_mask)
#
#
# class FeaturizedInferenceDataset()
#     # def get_tensor(self):
#     #     return self.tensor
#     #
#     # def get_training_data(self):
#     #     """
#     #     get_training_data returns a DatasetIterator which iterates over the
#     #     labelled training data.
#     #     """
#     #
#     #     # This assumes that we have a larger proportion of correct initial values
#     #     # and only a small amount of incorrect initial values which allow us
#     #     # to train to convergence
#     #     # """
#     #     train_idx = (self.weak_labels != -1).nonzero()[:,0]
#     #     return DatasetIterator(vids=train_idx, batch_size=self.env['batch_size'])
#     #     #
#     #     # X_train = self.tensor.index_select(0, train_idx)
#     #     # Y_train = self.weak_labels.index_select(0, train_idx)
#     #     # mask_train = self.var_class_mask.index_select(0, train_idx)
#     #     # return X_train, Y_train, mask_train
#     #
#     # def get_infer_data(self):
#     #     """
#     #     Retrieves the samples to be inferred i.e. DK cells.
#     #     """
#     #     # only infer on those that are DK cells
#     #     infer_idx = (self.is_clean == 0).nonzero()[:, 0]
#     #     X_infer = self.tensor.index_select(0, infer_idx)
#     #     mask_infer = self.var_class_mask.index_select(0, infer_idx)
#     #     return X_infer, mask_infer, infer_idx
#
# class DatasetIterator:
