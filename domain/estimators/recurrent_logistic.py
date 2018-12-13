from abc import ABCMeta, abstractmethod
import logging
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from ..estimator import Estimator
from utils import dictify_df

class RecurrentLogistic(Estimator, torch.nn.Module):
    """
    RecurrentLogistic is an Estimator that approximates posterior of
    p(v_cur | v_init) by training a logistic regression model to predict the current
    value in a cell given all other initial values using features
    of the other initial values such as co-occurrence.
    """

    def __init__(self, dataset, pruned_domain, active_attrs):
        """
        :param dataset: (Dataset)
        :param pruned_domain: (dict) of tid -> attr -> value for the domain
        :param active_attrs: (list[str]) attributes that have random values
        """
        torch.nn.Module.__init__(self)
        Estimator.__init__(self, dataset, pruned_domain, active_attrs)

        # self.dom maps tid --> attr --> list of domain values
        # we need to find the number of domain values we will be generating
        # a training sample for.
        self.n_samples = sum(len(dom) for _, attrs in self.dom.items() for _, dom in attrs.items())

        # make a copy of the raw data as our current data
        self.cur_df = self.ds.get_raw_data().copy()

        # generate featurizers
        self._update_featurizers()

        # pytorch logistic regression model
        self._W = torch.nn.Parameter(torch.zeros(len(self.attrs),1))
        torch.nn.init.xavier_uniform_(self._W)
        self._B = torch.nn.Parameter(torch.Tensor([1e-6]))
        self._loss = torch.nn.BCELoss()
        self._optimizer = torch.optim.Adam(self.parameters())

    def _update_featurizers(self):
        """
        Reinitialize featurizers that depend on updated current values.
        """

        self.featurizers = [
                CooccurFeaturizer(self.cur_df, self.attrs)
                ]
        self.num_features = sum(feat.num_features() for feat in self.featurizers)

    def _update_training_data(self):

        # Each row corresponds to a possible value for a given attribute and given TID
        self._X = torch.zeros(self.n_samples, self.num_features)
        self._Y = torch.zeros(self.n_samples)

        logging.info('RecurrentLogistic: featurizing training data')

        sample_idx = 0
        domain_idx_df = []
        # Iterate through each row and attribute and generate a sample (corresponds
        # to one row of X and Y)
        for row in tqdm(self.cur_df.to_records()):
            for attr in self.active_attrs:
                domain_vals = self.dom[row['_tid_']][attr]
                domain_idx_df += [{'_tid_': row['_tid_'],
                    'attr': attr,
                    'val': val} for val in domain_vals]

                # Initialize our X matrix with features from featurizers
                feat_tensor = self._gen_feat_tensor(row, attr, domain_vals)
                assert(feat_tensor.shape[0] == len(domain_vals))
                self._X[sample_idx:sample_idx+len(domain_vals)] = feat_tensor

                # target label is our initial value
                if row[attr] in domain_vals:
                    init_val_idx = domain_vals.index(row[attr])
                    self._Y[sample_idx+init_val_idx] = 1

                sample_idx += len(domain_vals)

        # Map index (along first dimension) in self._X and self._Y to domain values
        self._domain_idx_df = pd.DataFrame(domain_idx_df)

    def _gen_feat_tensor(self, row, attr, values):
        """
        :param row: (namedtuple, recarray, dict) current values
        :param attr: (str) attribute for :param values:
        :param values: (list[str]) values to generate features for
        """
        out_tensor = torch.zeros(len(values), self.num_features)

        # iterate through each featurizer
        feat_idx = 0
        for featurizer in self.featurizers:
            feat_tensor = featurizer.create_tensor(row, attr, values)
            assert(feat_tensor.shape[0] == len(values))
            assert(feat_tensor.shape[1] == featurizer.num_features())


            out_tensor[:][feat_idx:feat_idx+featurizer.num_features()] = feat_tensor
            feat_idx += featurizer.num_features()

        return out_tensor

    def forward(self, X):
        linear = X.matmul(self._W) + self._B
        return torch.nn.functional.sigmoid(linear)

    def train(self, num_recur=1, num_epochs=3, batch_size=32):
        """
        :param num_recur: (int) number of times to train model.
        :param num_epochs: (int) number of epochs PER recurrent iteration.
        :param batch_size: (int) size of batch in stochastic gradient descent.

        We update the current values with the maximum a posteriori after each recurrent iteration.
        """

        batch_losses = []
        for recur_idx in range(1, num_recur+1):
            self._update_training_data()
            torch_ds = TensorDataset(self._X, self._Y)

            logging.info("RecurrentLogistic: training, recur iteration: %d", recur_idx)
            for epoch_idx in range(1, num_epochs+1):
                logging.info("RecurrentLogistic: epoch %d", epoch_idx)
                for batch_X, batch_Y in tqdm(DataLoader(torch_ds, batch_size=batch_size)):
                    batch_pred = self.forward(batch_X)
                    batch_loss = self._loss(batch_pred, batch_Y.reshape(-1,1))
                    batch_losses.append(float(batch_loss))
                    self.zero_grad()
                    batch_loss.backward()
                    self._optimizer.step()
                # TODO(richardwu): update cur_df with predictions

        return batch_losses

    def predict_pp(self, row, attr, values):
        pred_X = self._gen_feat_tensor(row, attr, values)
        pred_Y = self.forward(pred_X)
        return list(zip(values, pred_Y))


class Featurizer:
    """
    Feauturizer is an abstract class for featurizers that is able to generate
    real-valued tensors (features) for a row from raw data.
    Used in RecurrentLogistic model.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def num_features(self):
        raise NotImplementedError

    @abstractmethod
    def create_tensor(self, row, attr, values):
        raise NotImplementedError


class CooccurFeaturizer(Featurizer):
    def __init__(self, data_df, attrs):
        """
        :param data_df: (pandas.DataFrame) contains the data to compute co-occurrence features for.
        :param attrs: attributes in columns of :param data_df: to compute feautres for.
        """
        self.data_df = data_df
        self.attrs = attrs

        # Frequencies of values per each attribute
        self.freq = {}
        for attr in self.attrs:
            self.freq[attr] = self._get_freq(attr)

        # Co-occurrence counts per each attribute pair (and value pair)
        self.cooccur_freq = {}
        for attr1 in self.attrs:
            self.cooccur_freq[attr1] = {}
            for attr2 in self.attrs:
                if attr1 == attr2:
                    continue
                self.cooccur_freq[attr1][attr2] = self._get_cooccur_freq(attr1, attr2)

    def num_features(self):
        return len(self.attrs)

    def create_tensor(self, row, attr, values):
        """
        :param row: (namedtuple or dict) current initial values
        :param attr: (str) attribute of row (i.e. cell) the :param values: correspond to
            and the cell to generate a feature tensor for.
        :param values: (list[str]) values to generate
        """
        tensor = torch.zeros(len(values), len(self.attrs))
        for val_idx, val in enumerate(values):
            for attr_idx, other_attr in enumerate(self.attrs):
                if attr == other_attr:
                    continue

                # calculate p(val | other_val)
                # there may not be co-occurrence frequencies for some value pairs since
                # our possible values were from correlation with only
                # one other attribute
                cooccur = self.cooccur_freq[attr][other_attr][val].get(row[other_attr], 0)
                freq = self.freq[other_attr][row[other_attr]]

                tensor[val_idx][attr_idx] = float(cooccur) / float(freq)
        return tensor

    def _get_freq(self, attr):
        """
        _get_freq returns a dictionary where the keys possible values for :param attr: and
        the values contain the frequency count of that value for this attribute.
        """
        return self.data_df[[attr]].groupby([attr]).size().to_dict()

    def _get_cooccur_freq(self, attr1, attr2):
        """
        _get_cooccur_freq returns a dictionary {val1 -> {val2 -> count } } where:
            <val1>: all possible values for :param attr1:
            <val2>: all values for :param attr2: that appeared at least once with <val1>
            <count>: frequency (# of entities) where :param attr1 = <val1> AND :param attr2: = <val2>
        """
        tmp_df = self.data_df[[attr1,attr2]].groupby([attr1,attr2]).size().reset_index(name="count")
        return dictify_df(tmp_df)

