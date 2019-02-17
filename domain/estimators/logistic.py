from abc import ABCMeta, abstractmethod
import logging

import pandas as pd
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from ..estimator import Estimator
from utils import dictify_df


class Logistic(Estimator, torch.nn.Module):
    """
    Logistic is an Estimator that approximates posterior of
    p(v_cur | v_init) by training a logistic regression model to predict the current
    value in a cell given all other initial values using features
    of the other initial values such as co-occurrence.
    """

    def __init__(self, env, dataset, domain_df, active_attrs, batch_size=32):
        """
        :param dataset: (Dataset) original dataset
        :param domain_df: (DataFrame) currently populated domain dataframe.
            Required columns are: _vid_, _tid_, attribute, domain, domain_size, init_value
        :param active_attrs: (list[str]) attributes that have random values
        """
        torch.nn.Module.__init__(self)
        Estimator.__init__(self, env, dataset)

        self.active_attrs = active_attrs

        # Sorted records of the currently populated domain. This helps us
        # align the final predicted probabilities.
        self.domain_records = domain_df.sort_values('_vid_')[['_vid_', '_tid_', 'attribute', 'domain', 'init_value']].to_records()

        # self.dom maps tid --> attr --> list of domain values
        # we need to find the number of domain values we will be generating
        # a training sample for.
        self.n_samples = int(domain_df['domain_size'].sum())

        # Create and initialize featurizers.
        self.featurizers = [CooccurAttrFeaturizer(self.ds)]
        for f in self.featurizers:
            f.setup()
        self.num_features = sum(feat.num_features() for feat in self.featurizers)

        # Construct the X and Y tensors.
        self._gen_training_data()

        # Use pytorch logistic regression model.
        self._W = torch.nn.Parameter(torch.zeros(self.num_features, 1))
        torch.nn.init.xavier_uniform_(self._W)
        self._B = torch.nn.Parameter(torch.Tensor([1e-6]))

        self._loss = torch.nn.BCELoss()
        if self.env['optimizer'] == 'sgd':
            self._optimizer = SGD(self.parameters(), lr=self.env['learning_rate'], momentum=self.env['momentum'],
                                  weight_decay=self.env['weight_decay'])
        else:
            self._optimizer = Adam(self.parameters(), lr=self.env['learning_rate'], weight_decay=self.env['weight_decay'])

    def _gen_training_data(self):
        """
        _gen_training_data memoizes the self._X and self._Y tensors
        used for training and prediction.
        """
        logging.debug('Logistic: featurizing training data...')
        tic = time.clock()
        # Each row corresponds to a possible value for a given attribute
        # and given TID
        self._X = torch.zeros(self.n_samples, self.num_features)
        self._Y = torch.zeros(self.n_samples)

        """
        Iterate through the domain for every cell and create a sample
        to use in training. We assign Y as 1 if the value is the initial value.
        """
        sample_idx = 0
        raw_data_dict = self.ds.raw_data.df.set_index('_tid_').to_dict('index')
        # Keep track of which indices correspond to a VID so we can re-use
        # self._X in prediction.
        self.vid_to_idxs = {}
        for rec in tqdm(list(self.domain_records)):
            init_row = raw_data_dict[rec['_tid_']]
            domain_vals = rec['domain'].split('|||')

            # Generate the feature tensor for all the domain values for this
            # cell.
            feat_tensor = self._gen_feat_tensor(init_row, rec['attribute'], domain_vals)
            assert(feat_tensor.shape[0] == len(domain_vals))
            self._X[sample_idx:sample_idx+len(domain_vals)] = feat_tensor

            # Assign the tensor corresponding to the initial value with
            # a target label of 1.
            try:
                init_idx = domain_vals.index(rec['init_value'])
                self._Y[sample_idx + init_idx] = 1
            except ValueError:
                logging.error('init value "%s" should be in domain "%s" for VID %d', rec['init_value'], rec['domain'], rec['_vid_'])
                raise

            self.vid_to_idxs[rec['_vid_']] = (sample_idx, sample_idx+len(domain_vals))
            sample_idx += len(domain_vals)

        logging.debug('Logistic: DONE featurization in %.2fs', time.clock() - tic)

    def _gen_feat_tensor(self, init_row, attr, domain_vals):
        """
        Generates the feature tensor for the list of :param`domain_vals` from
        all featurizers.

        :param init_row: (namedtuple or dict) current initial values
        :param attr: (str) attribute of row (i.e. cell) the :param values: correspond to
            and the cell to generate a feature tensor for.
        :param domain_vals: (list[str]) domain values to featurize for

        :return: Tensor with dimensions (len(values), total # of features across all featurizers)
        """

        return torch.cat([f.create_tensor(init_row, attr, domain_vals) for f in self.featurizers], dim=1)

    def forward(self, X):
        linear = X.matmul(self._W) + self._B
        return torch.sigmoid(linear)

    def train(self, num_epochs=3, batch_size=32):
        """
        Trains the LR model.

        :param num_epochs: (int) number of epochs.
        """
        batch_losses = []
        torch_ds = TensorDataset(self._X, self._Y)

        # Main training loop.
        for epoch_idx in range(1, num_epochs+1):
            logging.debug("Logistic: epoch %d", epoch_idx)
            batch_cnt = 0
            for batch_X, batch_Y in tqdm(DataLoader(torch_ds, batch_size=batch_size)):
                batch_pred = self.forward(batch_X)
                batch_loss = self._loss(batch_pred, batch_Y.reshape(-1,1))
                batch_losses.append(float(batch_loss))
                self.zero_grad()
                batch_loss.backward()
                self._optimizer.step()
                batch_cnt += 1
            logging.debug('Logistic: average batch loss: %f', sum(batch_losses[-1 * batch_cnt:]) / batch_cnt)
        return batch_losses

    def predict_pp(self, row, attr=None, values=None):
        """
        predict_pp generates posterior probabilities for the domain values
        corresponding to the cell/random variable row['_vid_'].

        That is: :param`attr` and :param`values` are ignored.

        predict_pp_batch is much faster for Logistic since it simply does
        a one-pass of the batch feature tensor.

        :return: (list[2-tuple]) 2-tuples corresponding to (value, proba)
        """
        start_idx, end_idx = self.vid_to_idxs[row['_vid_']]
        pred_X = self._X[start_idx:end_idx]
        pred_Y = self.forward(pred_X)
        values = self.domain_records[row['_vid_']]['domain'].split('|||')
        return zip(values, map(float, pred_Y))

    def predict_pp_batch(self, raw_records_by_tid=None, cell_domain_rows=None):
        """
        Performs batch prediction.

        Parameters are ignored: instead it returns the predictions for cells
        and domain values ordered by VID of the initial domain dataframe
        passed into the constructor.
        """
        pred_Y = self.forward(self._X)
        for rec in self.domain_records:
            values = rec['domain'].split('|||')
            start_idx, end_idx = self.vid_to_idxs[rec['_vid_']]
            yield zip(values, map(float, pred_Y[start_idx:end_idx]))

class Featurizer:
    """
    Feauturizer is an abstract class for featurizers that is able to generate
    real-valued tensors (features) for a row from raw data.
    Used in Logistic model.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def num_features(self):
        raise NotImplementedError

    @abstractmethod
    def create_tensor(self, row, attr, values):
        raise NotImplementedError


class CooccurAttrFeaturizer(Featurizer):
    """
    CooccurAttrFeaturizer computes the co-occurrence statistics for a cell
    and its possible domain values with the other initial values in the tuple.
    It breaks down each co-occurrence feature on a pairwise attr1 X attr2 basis.
    """
    name = 'CooccurAttrFeaturizer'

    def __init__(self, dataset):
        """
        :param data_df: (pandas.DataFrame) contains the data to compute co-occurrence features for.
        :param attrs: attributes in columns of :param data_df: to compute feautres for.
        :param freq: (dict { attr: { val: count } } }) if not None, uses these
            frequency statistics instead of computing it from data_df.
        :param cooccur_freq: (dict { attr1: { attr2: { val1: { val2: count } } } })
            if not None, uses these co-occurrence statistics instead of
            computing it from data_df.
        """
        self.ds = dataset
        self.attrs = self.ds.get_attributes()
        self.attr_to_idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.n_attrs = len(self.attrs)

    def num_features(self):
        return len(self.attrs) * len(self.attrs)

    def setup(self):
        _, self.freq, self.cooccur_freq = self.ds.get_statistics()

    def create_tensor(self, row, attr, values):
        """
        :param row: (namedtuple or dict) current initial values
        :param attr: (str) attribute of row (i.e. cell) the :param values: correspond to
            and the cell to generate a feature tensor for.
        :param values: (list[str]) values to generate

        :return: Tensor with dimensions (len(values), # of features)
        """
        tensor = torch.zeros(len(values), self.num_features())
        for val_idx, val in enumerate(values):
            for other_attr_idx, other_attr in enumerate(self.attrs):
                if attr == other_attr:
                    continue

                # calculate p(val | other_val)
                # there may not be co-occurrence frequencies for some value pairs since
                # our possible values were from correlation with only
                # one other attribute
                cooccur = self.cooccur_freq[attr][other_attr][val].get(row[other_attr], 0)
                freq = self.freq[other_attr][row[other_attr]]

                feat_idx = self.attr_to_idx[attr] * self.n_attrs + other_attr_idx

                tensor[val_idx,feat_idx] = float(cooccur) / float(freq)
        return tensor
