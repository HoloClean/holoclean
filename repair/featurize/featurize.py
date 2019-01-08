import logging
import torch
from tqdm import tqdm
from collections import namedtuple
from dataset import AuxTables, CellStatus

FeatInfo = namedtuple('FeatInfo', ['name', 'size', 'learnable', 'init_weight', 'feature_names'])

class FeaturizedDataset:
    def __init__(self, dataset, env, featurizers):
        self.ds = dataset
        self.env = env
        self.total_vars, self.classes = self.ds.get_domain_info()
        self.processes = self.env['threads']
        for f in featurizers:
            f.setup_featurizer(self.ds, self.total_vars, self.classes, self.processes)
        tensors = [f.create_tensor() for f in featurizers]
        self.featurizer_info = [FeatInfo(featurizer.name,
            tensor.size()[2],
            featurizer.learnable,
            featurizer.init_weight,
            featurizer.feature_names())
            for tensor, featurizer in zip(tensors, featurizers)]
        tensor = torch.cat(tensors,2)
        # DEBUGING
        self.debugging = {}
        print("========== DEBUGGING ==========")
        for i, t in enumerate(tensors):
            debug = t[9324, :, :].numpy()
            feat = featurizers[i].name
            self.debugging[feat] = {}
            self.debugging[feat]['size'] = debug.shape
            self.debugging[feat]['weights'] = debug

        self.tensor = tensor
        # TODO: remove after we validate it is not needed.
        self.in_features = self.tensor.shape[2]
        self.weak_labels, self.labels_type = self.generate_weak_labels()
        self.var_class_mask, self.var_to_domsize = self.generate_var_mask()

    def generate_weak_labels(self):
        """
        generate_weak_labels returns a tensor where for each VID we have the
        domain index of the initial value.

        :return: Torch.Tensor of size (# of variables) X 1 where tensor[i][0]
            contains the domain index of the initial value for the i-th
            variable/VID.
        """
        logging.debug("Generating weak labels.")
        query = 'SELECT _vid_, weak_label_idx, fixed FROM %s AS t1 LEFT JOIN %s AS t2 ' \
                'ON t1._cid_ = t2._cid_ WHERE t2._cid_ is NULL OR t1.fixed != %d;' % (
        AuxTables.cell_domain.name, AuxTables.dk_cells.name, CellStatus.NOT_SET.value)
        res = self.ds.engine.execute_query(query)
        if len(res) == 0:
            raise Exception("No weak labels available. Reduce pruning threshold.")
        labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        labels_type = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            label = int(tuple[1])
            fixed = int(tuple[2])
            labels[vid] = label
            labels_type[vid] = fixed
        logging.debug("DONE generating weak labels.")
        return labels, labels_type

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
        logging.debug("Generating mask.")
        var_to_domsize = {}
        query = 'SELECT _vid_, domain_size FROM %s' % AuxTables.cell_domain.name
        res = self.ds.engine.execute_query(query)
        mask = torch.zeros(self.total_vars,self.classes)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            max_class = int(tuple[1])
            mask[vid, max_class:] = -10e6
            var_to_domsize[vid] = max_class
        logging.debug("DONE generating mask.")
        return mask, var_to_domsize

    def get_tensor(self):
        return self.tensor

    def get_training_data(self):
        """
        get_training_data returns X_train, y_train, and mask_train
        where each row of each tensor is a variable/VID and
        y_train are weak labels for each variable i.e. they are
        set as the initial values.

        This assumes that we have a larger proportion of correct initial values
        and only a small amount of incorrect initial values which allow us
        to train to convergence.
        """
        train_idx = (self.weak_labels != -1).nonzero()[:,0]
        X_train = self.tensor.index_select(0, train_idx)
        Y_train = self.weak_labels.index_select(0, train_idx)
        mask_train = self.var_class_mask.index_select(0, train_idx)
        return X_train, Y_train, mask_train

    def get_infer_data(self, infer_labeled):
        """
        :param infer_labeled: (bool) infer also for cells that have been used with weak labels
        """
        if infer_labeled:
            infer_idx = (self.labels_type <= CellStatus.SINGLE_VALUE.value).nonzero()[:, 0]
            X_infer = self.tensor.index_select(0, infer_idx)
            mask_infer = self.var_class_mask.index_select(0, infer_idx)
        else:
            infer_idx = (self.weak_labels == -1).nonzero()[:, 0]
            X_infer = self.tensor.index_select(0, infer_idx)
            mask_infer = self.var_class_mask.index_select(0, infer_idx)
        return X_infer, mask_infer, infer_idx
