import pandas as pd
import time

from .featurize import FeaturizedDataset
from .learn import RepairModel
from dataset import AuxTables


class RepairEngine:
    def __init__(self, env, dataset):
        self.ds = dataset
        self.env = env

    def setup_featurized_ds(self, featurizers):
        tic = time.clock()
        self.feat_dataset = FeaturizedDataset(self.ds, self.env, featurizers)
        toc = time.clock()
        status = "DONE setting up featurized dataset."
        feat_time = toc - tic
        return status, feat_time

    def setup_repair_model(self):
        tic = time.clock()
        feat_info = self.feat_dataset.featurizer_info
        output_dim = self.feat_dataset.classes
        self.repair_model = RepairModel(self.env, feat_info, output_dim, bias=self.env['bias'])
        toc = time.clock()
        status = "DONE setting up repair model."
        setup_time = toc - tic
        return status, setup_time

    def fit_repair_model(self):
        tic = time.clock()
        X_train, Y_train, mask_train = self.feat_dataset.get_training_data()
        self.repair_model.fit_model(X_train, Y_train, mask_train)
        toc = time.clock()
        status = "DONE training repair model."
        train_time = toc - tic
        return status, train_time

    def infer_repairs(self):
        tic = time.clock()
        X_pred, mask_pred, infer_idx = self.feat_dataset.get_infer_data()
        Y_pred = self.repair_model.infer_values(X_pred, mask_pred)
        distr_df, infer_val_df = self.get_infer_dataframes(infer_idx, Y_pred)
        self.ds.generate_aux_table(AuxTables.cell_distr, distr_df, store=True, index_attrs=['_vid_'])
        self.ds.generate_aux_table(AuxTables.inf_values_idx, infer_val_df, store=True, index_attrs=['_vid_'])
        toc = time.clock()
        status = "DONE inferring repairs."
        infer_time = toc - tic
        return status, infer_time

    def get_infer_dataframes(self, infer_idx, Y_pred):
        distr = []
        infer_val = []
        Y_assign = Y_pred.data.numpy().argmax(axis=1)
        domain_size = self.feat_dataset.var_to_domsize
        for idx in range(Y_pred.shape[0]):
            vid = int(infer_idx[idx])
            rv_distr = list(Y_pred[idx].data.numpy())
            rv_value = int(Y_assign[idx])
            rv_prob = Y_pred[idx].data.numpy().max()
            d_size = domain_size[vid]
            distr.append({'_vid_': vid, 'distribution':[str(p) for p in rv_distr[:d_size]]})
            infer_val.append({'_vid_': vid, 'inferred_assignment':rv_value, 'prob':rv_prob})
        distr_df = pd.DataFrame(data=distr)
        infer_val_df = pd.DataFrame(data=infer_val)
        return distr_df, infer_val_df

    def get_featurizer_weights(self):
        tic = time.clock()
        report = self.repair_model.get_featurizer_weights(
            self.feat_dataset.featurizer_info,
            self.feat_dataset.debugging
        )
        toc = time.clock()
        report_time = toc - tic
        return report, report_time
