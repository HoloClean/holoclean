import logging
import time
import os

import pandas as pd

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
        training_data = self.feat_dataset.get_training_dataset()
        logging.info('training with %d training examples (cells)', training_data.num_examples)
        self.repair_model.fit_model(training_data)
        toc = time.clock()
        status = "DONE training repair model."
        train_time = toc - tic
        return status, train_time

    def infer_repairs(self):
        tic = time.clock()
        infer_data, infer_idx = self.feat_dataset.get_infer_dataset()
        Y_pred = self.repair_model.infer_values(infer_data)
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

        # Need to map the inferred value index of the random variable to the actual value
        # val_idx = val_id - 1 since val_id was numbered starting from 1 whereas
        # val_idx starts at 0.
        query = 'SELECT _vid_, val_id-1, rv_val FROM {pos_values}'.format(pos_values=AuxTables.pos_values.name)
        pos_values = self.ds.engine.execute_query(query)
        # dict mapping _vid_ --> val_idx --> value
        vid_to_val = {}
        for vid, val_idx, val in pos_values:
            vid_to_val[vid] = vid_to_val.get(vid, {})
            vid_to_val[vid][val_idx] = val

        for idx in range(Y_pred.shape[0]):
            vid = int(infer_idx[idx])
            rv_distr = list(Y_pred[idx].data.numpy())
            rv_val_idx = int(Y_assign[idx])
            rv_val = vid_to_val[vid][rv_val_idx]
            rv_prob = Y_pred[idx].data.numpy().max()
            d_size = domain_size[vid]
            distr.append({'_vid_': vid, 'distribution':[str(p) for p in rv_distr[:d_size]]})
            infer_val.append({'_vid_': vid, 'inferred_val_idx': rv_val_idx, 'inferred_val': rv_val, 'prob':rv_prob})
        distr_df = pd.DataFrame(data=distr)
        infer_val_df = pd.DataFrame(data=infer_val)
        return distr_df, infer_val_df

    def get_featurizer_weights(self):
        tic = time.clock()
        report = self.repair_model.get_featurizer_weights(self.feat_dataset.featurizer_info)
        toc = time.clock()
        report_time = toc - tic
        return report, report_time

    def clear_cache(self):
        tic = time.clock()
        directory_path = "%s/cache/" % os.environ['HOLOCLEANHOME']
        for file_name in os.listdir(directory_path):
            if "tensor" in file_name:
                file_path = os.path.join(directory_path, file_name)
                os.unlink(file_path)
        toc = time.clock()
        status = "DONE clearing cache."
        infer_time = toc - tic
        return status, infer_time
