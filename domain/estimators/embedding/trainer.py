import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import NULL_REPR
from torch.utils.data import DataLoader, Sampler
from torch.nn import Softmax

NONNUMERICS = "[^0-9+-.e]"

class VidSampler(Sampler):
    def __init__(self, domain_df, raw_df, numerical_attr_groups,
            shuffle=True, train_only_clean=False):
        # No NULL targets
        domain_df = domain_df[domain_df['weak_label'] != NULL_REPR]

        # No NULL values in each cell's numerical group (all must be non-null
        # since target_numvals requires all numerical values.
        if numerical_attr_groups:
            raw_data_dict = raw_df.set_index('_tid_').to_dict('index')
            attr_to_group = {attr: group for group in numerical_attr_groups
                    for attr in group}
            def group_notnull(row):
                tid = row['_tid_']
                cur_attr = row['attribute']
                # Non-numerical cell: return true
                if cur_attr not in attr_to_group:
                    return True
                return all(raw_data_dict[tid][attr] != NULL_REPR
                        for attr in attr_to_group[cur_attr])
            fil_notnull = domain_df.apply(group_notnull, axis=1)
            if sum(fil_notnull) < domain_df.shape[0]:
                logging.warning('dropping %d targets where target\'s numerical group contain NULLs',
                        domain_df.shape[0] - sum(fil_notnull))
                domain_df = domain_df[fil_notnull]

        # Train on only clean cells
        if train_only_clean:
            self._vids = domain_df.loc[(domain_df['is_clean'] | domain_df['fixed'] >= 1), '_vid_']
        else:
            self._vids = domain_df['_vid_'].values

        if shuffle:
            self._vids = np.random.permutation(self._vids)

    def __iter__(self):
        return iter(self._vids.tolist())

    def __len__(self):
        return len(self._vids)

class Trainer:
    """
    Trainer class
    """
    def __init__(self, model, num_loss, cat_loss, optimizer, dataset, predictor, domain_df, 
    validate_fpath=None, validate_tid_col=None, validate_attr_col=None,
            validate_val_col=None, validate_epoch=None):

        """
        :param model: (torch.nn.Module) tuple embedding model object
        :param num_loss: (torch.nn.modules.loss) loss function for numerical attribute
        :param cat_loss: (torch.nn.modules.loss) loss function for categorical attribute
        :param optimizer: (torch.optim) optimizer used in training
        :param dataset: (DataSet) embedding dataset object
        :param predictor: (object) tuple embedding predictor class object
        :param domain_df: (DataFrame) dataframe containing domain values
        :param validate_fpath: (string) filepath to validation CSV
        :param validate_tid_col: (string) column containing TID
        :param validate_attr_col: (string) column containing attribute
        :param validate_val_col: (string) column containing correct value
        """
    
        self.model = model
        self._dataset = dataset
        self.domain_df = domain_df

        # parameters for trainning
        self._cat_loss = cat_loss
        self._num_loss = num_loss
        self._optimizer = optimizer

        # store numerical attributes
        self._numerical_attrs = dataset.get_num_attr()

        # initialize validation dataframe
        self._init_validation_df(validate_fpath, validate_tid_col, validate_attr_col, validate_val_col, validate_epoch)

        # initialize predictor that used in validation
        self.predictor = predictor

    def train(self, num_epochs, batch_size, weight_entropy_lambda, shuffle, train_only_clean):
        """
        :param num_epochs: (int) number of epochs to train for
        :param batch_size: (int) size of batches
        :param weight_entropy_lambda: (float) penalization strength for
            weights assigned to other attributes for a given attribute.
            A higher penalization strength means the model will depend
            on more attributes instead of putting all weight on a few
            attributes. Recommended values between 0 to 0.5.
        :param shuffle: (bool) shuffle the dataset while training
        :param train_only_clean: (bool) train only on clean cells not marked by
            error detection. Recommend False if error detector is very liberal.
        """

        # Returns VIDs to train on.
        sampler = VidSampler(self.domain_df, self._dataset.ds.get_raw_data(),
                self.model._numerical_attr_groups,
                shuffle=shuffle, train_only_clean=train_only_clean)

        logging.debug("%s: training (lambda = %f) on %d cells (%d cells in total) in:\n1) %d categorical columns: %s\n2) %d numerical columns: %s",
                      type(self).__name__,
                      weight_entropy_lambda,
                      len(sampler),
                      self.domain_df.shape[0],
                      self.model._n_train_cat_attrs,
                      self.model._train_cat_attrs,
                      self.model._n_train_num_attrs,
                      self.model._train_num_attrs)

        trainDataGenerator = DataLoader(self._dataset, batch_size=batch_size, sampler=sampler)

        num_batches = len(trainDataGenerator)
        num_steps = num_epochs * num_batches
        batch_losses = []
        
        # Main training loop.
        for epoch_idx in range(1, num_epochs+1):
            batch_cnt = 0
            logging.debug('%s: epoch %d of %d', type(self).__name__, epoch_idx, num_epochs)
            logging.debug('%s: using cosine LR scheduler with %d steps', type(self).__name__, num_batches)
            scheduler = CosineAnnealingLR(self._optimizer, num_batches)

            for vids, is_categorical, attr_idxs, \
                init_cat_idxs, init_numvals, init_nummasks, \
                domain_idxs, domain_masks, \
                target_numvals, cat_targets \
                in tqdm(trainDataGenerator):

                cat_preds, numval_preds, cat_mask, num_mask = self.model.forward(is_categorical, attr_idxs,
                        init_cat_idxs, init_numvals, init_nummasks,
                        domain_idxs, domain_masks)

                # Select out the appropriate targets
                cat_targets = cat_targets.view(-1)[cat_mask]
                target_numvals = target_numvals[num_mask]

                assert cat_preds.shape[0] == cat_targets.shape[0]
                assert numval_preds.shape == target_numvals.shape

                batch_loss = 0.
                if cat_targets.shape[0] > 0:
                    batch_loss += self._cat_loss(cat_preds, cat_targets)
                if target_numvals.shape[0] > 0:
                    # Note both numval_preds and target_numvals have 0-ed out
                    # values if the sample's dimension is < max dim.
                    # TODO: downweight samples that are part of a group of n attributes
                    # by 1/n.
                    batch_loss += self._num_loss(numval_preds, target_numvals)

                # Add the negative entropy of the attr_W to the cost: that is
                # we maximize entropy of the logits of attr_W to encourage
                # non-sparsity of logits.
                if weight_entropy_lambda != 0.:
                    attr_weights = Softmax(dim=1)(self.model.attr_W).view(-1)
                    neg_attr_W_entropy = attr_weights.dot(attr_weights.log()) / self.model.attr_W.shape[0]
                    batch_loss.add_(weight_entropy_lambda * neg_attr_W_entropy)

                batch_losses.append(float(batch_loss))
                self.model.zero_grad()
                batch_loss.backward()

                # Do not update weights for 0th reserved vectors.
                if self.model.in_W._grad is not None:
                    self.model.in_W._grad[0].zero_()
                if self.model.out_W._grad is not None:
                    self.model.out_W._grad[0].zero_()
                if self.model.out_B._grad is not None:
                    self.model.out_B._grad[0].zero_()

                self._optimizer.step()
                scheduler.step()
                batch_cnt += 1

            logging.debug('%s: average batch loss: %f',
                    type(self).__name__,
                    sum(batch_losses[-1 * batch_cnt:]) / batch_cnt)

            if self._do_validation and epoch_idx % self._validate_epoch == 0:
                res = self.validate()

        return batch_losses

    def validate(self):
        logging.debug('%s: running validation set...', type(self).__name__)

        # Construct DataFrame with inferred values
        validation_preds = list(self.predictor.predict_pp_batch(self._validate_df))
        df_pred = []
        for vid, is_cat, preds in tqdm(validation_preds):
            if is_cat:
                inf_val, inf_prob = max(preds, key=lambda t: t[1])
            else:
                # preds is just a float
                inf_val, inf_prob = preds, -1

            df_pred.append({'_vid_': vid,
                'is_cat': is_cat,
                'inferred_val': inf_val,
                'proba': inf_prob})
        df_pred = pd.DataFrame(df_pred)
        df_res = self._validate_df.merge(df_pred, on=['_vid_'])


        # General filters and metrics
        fil_dk = ~df_res['is_clean']
        fil_cat = df_res['is_cat']
        fil_grdth = df_res['_value_'].apply(lambda arr: arr != [NULL_REPR])

        if (~fil_grdth).sum():
            logging.debug('%s: there are %d cells with no validation ground truth',
                    type(self).__name__,
                    (~fil_grdth).sum())

        n_cat = fil_cat.sum()
        n_num = (~fil_cat).sum()

        n_cat_dk = (fil_dk & fil_cat).sum()
        n_num_dk = (fil_dk & ~fil_cat).sum()

        # Categorical filters and metrics
        fil_err = df_res.apply(lambda row: row['init_value'] not in row['_value_'],
                axis=1) & fil_cat & fil_grdth
        fil_noterr = ~fil_err & fil_cat & fil_grdth
        fil_cor = df_res.apply(lambda row: row['inferred_val'] in row['_value_'],
                axis=1) & fil_cat & fil_grdth
        fil_repair = (df_res['init_value'] != df_res['inferred_val']) & fil_cat

        total_err = fil_err.sum()
        detected_err = (fil_dk & fil_err).sum()

        n_repair = fil_repair.sum()
        n_repair_dk = (fil_dk & fil_repair).sum()
        n_cor_repair = (fil_cor & fil_repair).sum()
        n_cor_repair_dk = (fil_dk & fil_cor & fil_repair).sum()

        if total_err == 0:
            logging.warning('%s: total errors in validation set is 0', type(self).__name__)
        if detected_err == 0:
            logging.warning('%s: total detected errors in validation set is 0', type(self).__name__)

        # In-sample accuracy (predict init value that is already correcT)
        sample_acc = (fil_noterr & fil_cor).sum() / (fil_noterr).sum()

        precision = n_cor_repair / max(n_repair, 1)
        recall = n_cor_repair / max(total_err, 1)

        precision_dk = n_cor_repair_dk / max(n_repair_dk, 1)
        repair_recall = n_cor_repair_dk / max(detected_err, 1)

        # Numerical metrics (RMSE)
        rmse = 0
        rmse_dk = 0
        rmse_by_attr = {}
        rmse_dk_by_attr = {}
        if n_num:
            rmse = self._calc_rmse(df_res, ~fil_cat)
            rmse_dk = self._calc_rmse(df_res, ~fil_cat & fil_dk)
            for attr in self._numerical_attrs:
                fil_attr = df_res['attribute'] == attr
                rmse_by_attr[attr] = self._calc_rmse(df_res, fil_attr)
                rmse_dk_by_attr[attr] = self._calc_rmse(df_res, fil_attr & fil_dk)

        # Compile results
        val_res = {'n_cat': n_cat,
            'n_num': n_num,
            'n_cat_dk': n_cat_dk,
            'n_num_dk': n_num_dk,
            'total_err': total_err,
            'detected_err': detected_err,
            'n_repair': n_repair,
            'n_repair_dk': n_repair_dk,
            'sample_acc': sample_acc,
            'precision': precision,
            'recall': recall,
            'precision_dk': precision_dk,
            'repair_recall': repair_recall,
            'rmse': rmse,
            'rmse_dk': rmse_dk,
            'rmse_by_attr': rmse_by_attr,
            'rmse_dk_by_attr': rmse_dk_by_attr,
            }

        logging.debug("%s: # categoricals: (all) %d, (DK) %d",
                type(self).__name__, val_res['n_cat'], val_res['n_cat_dk'])
        logging.debug("%s: # numericals: (all) %d, (DK) %d",
                type(self).__name__, val_res['n_num'], val_res['n_num_dk'])

        logging.debug("%s: # of errors: %d, # of detected errors: %d",
                type(self).__name__, val_res['total_err'], val_res['detected_err'])

        logging.debug("%s: In-sample accuracy: %.3f",
                type(self).__name__, val_res['sample_acc'])

        logging.debug("%s: # repairs: (all) %d, (DK) %d",
                type(self).__name__, val_res['n_repair'], val_res['n_repair_dk'])

        logging.debug("%s: (Infer on all) Precision: %.3f, Recall: %.3f",
                type(self).__name__, val_res['precision'], val_res['recall'])
        logging.debug("%s: (Infer on DK) Precision: %.3f, Repair Recall: %.3f",
                type(self).__name__, val_res['precision_dk'], val_res['repair_recall'])

        if val_res['n_num']:
            logging.debug("%s: RMSE: (all) %f, (DK) %f", type(self).__name__,
                    val_res['rmse'], val_res['rmse_dk'])
            logging.debug("%s: RMSE per attr:", type(self).__name__)
            for attr in self._numerical_attrs:
                logging.debug("\t'%s': (all) %f, (DK) %f", attr,
                        val_res['rmse_by_attr'].get(attr, np.nan),
                        val_res['rmse_dk_by_attr'].get(attr, np.nan))

        return val_res

    def _init_validation_df(self, validate_fpath, validate_tid_col, validate_attr_col, validate_val_col, validate_epoch):
        self._do_validation = False
        if validate_fpath is not None \
            and validate_tid_col is not None \
            and validate_attr_col is not None \
            and validate_val_col is not None:
            self._validate_df = pd.read_csv(validate_fpath, dtype=str)
            self._validate_df.rename({validate_tid_col: '_tid_',
                validate_attr_col: '_attribute_',
                validate_val_col: '_value_',
                }, axis=1, inplace=True)
            self._validate_df['_tid_'] = self._validate_df['_tid_'].astype(int)
            self._validate_df['_value_'] = self._validate_df['_value_'].str.strip().str.lower()
            # Merge left so we can still get # of repairs for cells without
            # ground truth.
            self._validate_df = self.domain_df.merge(self._validate_df, how='left',
                    left_on=['_tid_', 'attribute'], right_on=['_tid_', '_attribute_'])
            self._validate_df['_value_'].fillna(NULL_REPR, inplace=True)
            # | separated correct values
            self._validate_df['_value_'] = self._validate_df['_value_'].str.split('\|')

            fil_notnull = self._validate_df['_value_'].apply(lambda arr: arr != [NULL_REPR])

            # Raise error if validation set has non-numerical values for numerical attrs
            if self._numerical_attrs is not None:
                fil_attr = self._validate_df['attribute'].isin(self._numerical_attrs)
                fil_notnumeric = self._validate_df['_value_'].apply(lambda arr: arr[0]).str.contains(NONNUMERICS)
                bad_numerics = fil_attr & fil_notnull & fil_notnumeric
                if bad_numerics.sum():
                    logging.error('%s: validation dataframe contains %d non-numerical values in numerical attrs %s',
                        type(self).__name__,
                        bad_numerics.sum(),
                        self._numerical_attrs)
                    raise Exception()

            # Log how many cells are actually repairable based on domain generated.
            # Cells without ground truth are "not repairable".
            fil_repairable = self._validate_df[fil_notnull].apply(lambda row: any(v in row['domain'] for v in row['_value_']), axis=1)
            logging.debug("%s: max repairs possible (# cells ground truth in domain): (DK) %d, (all): %d",
                        type(self).__name__,
                        (fil_repairable & ~self._validate_df['is_clean']).sum(),
                        fil_repairable.sum())

            self._validate_df = self._validate_df[['_vid_', 'attribute', 'init_value', '_value_', 'is_clean']]
            self._validate_epoch = validate_epoch or 1
            self._do_validation = True
        

        def _calc_rmse(df_res, df_filter):
            """
            calculate root mean square error for numerical attributes
            """
            if df_filter.sum() == 0:
                return 0
            X_cor = df_res.loc[df_filter, '_value_'].apply(lambda arr: arr[0]).values.astype(np.float)
            X_inferred = df_res.loc[df_filter, 'inferred_val'].values.astype(np.float)
            assert X_cor.shape == X_inferred.shape
            return np.sqrt(np.mean((X_cor - X_inferred) ** 2))
