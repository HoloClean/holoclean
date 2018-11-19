import torch
from tqdm import tqdm

from dataset import AuxTables


class FeaturizedDataset:
    def __init__(self, dataset, env, featurizers):
        self.ds = dataset
        self.env = env
        self.total_vars, self.classes = self.ds.get_domain_info()
        self.processes = self.env['threads']
        for f in featurizers:
            f.setup_featurizer(self.ds, self.total_vars, self.classes, self.processes)
        tensors = [f.create_tensor() for f in featurizers]
        # save size info of all featurizers
        self.featurizer_info = [(str(type(featurizers[i])), t.cpu().numpy().shape[2]) for i, t in enumerate(tensors)]
        tensor = torch.cat(tensors,2)
        self.tensor = tensor
        self.in_features = self.tensor.shape[2]
        self.weak_labels = self.generate_weak_labels()
        self.var_class_mask, self.var_to_domsize = self.generate_var_mask()

    def generate_weak_labels(self):
        if self.env['verbose']:
            print("Generating weak labels.")
        query = 'SELECT _vid_, init_index FROM %s AS t1 LEFT JOIN %s AS t2 ' \
                'ON t1._cid_ = t2._cid_ WHERE t2._cid_ is NULL OR t1.fixed = 1;' % (
        AuxTables.cell_domain.name, AuxTables.dk_cells.name)
        res = self.ds.engine.execute_query(query)
        if len(res) == 0:
            raise Exception("No weak labels available. Reduce pruning threshold.")
        labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            label = int(tuple[1])
            labels[vid] = label
        if self.env['verbose']:
            print("DONE generating weak labels.")
        return labels

    def generate_var_mask(self):
        if self.env['verbose']:
            print("Generating mask.")
        var_to_domsize = {}
        query = 'SELECT _vid_, domain_size FROM %s' % AuxTables.cell_domain.name
        res = self.ds.engine.execute_query(query)
        mask = torch.zeros(self.total_vars,self.classes)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            max_class = int(tuple[1])
            mask[vid, max_class:] = -10e6
            var_to_domsize[vid] = max_class
        if self.env['verbose']:
            print("DONE generating mask.")
        return mask, var_to_domsize

    def get_tensor(self):
        return self.tensor

    def get_training_data(self):
        train_idx = (self.weak_labels != -1).nonzero()[:,0]
        X_train = self.tensor.index_select(0, train_idx)
        Y_train = self.weak_labels.index_select(0, train_idx)
        mask_train = self.var_class_mask.index_select(0, train_idx)
        return X_train, Y_train, mask_train

    def get_infer_data(self):
        infer_idx = (self.weak_labels == -1).nonzero()[:, 0]
        X_infer = self.tensor.index_select(0, infer_idx)
        mask_infer = self.var_class_mask.index_select(0, infer_idx)
        return X_infer, mask_infer, infer_idx
