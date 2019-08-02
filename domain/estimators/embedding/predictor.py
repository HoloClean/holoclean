import pandas as pd
import numpy as np
import torch
import math
import logging
from tqdm import tqdm
from torch.nn import Softmax
from torch.utils.data import DataLoader, Sampler

class IterSampler(Sampler):
    def __init__(self, iter):
        self.iter = iter

    def __iter__(self):
        return iter(self.iter)

    def __len__(self):
        return len(self.iter)

class Predictor:
    def __init__(self, model, dataset, domain_df, domain_recs):
        """
        Predictor class to evaluate for a given model/dataframe
        :param model: (torch.nn.Module) tuple embedding model object
        :param dataset: (DataSet) tuple embedding dataset object
        :param domain_df: (DataFrame) dataframe containing domain values
        """

        self._embed_size = model.get_embed_size()
        self.model = model
        self._dataset = dataset
        self.domain_df = domain_df
        self.domain_recs = domain_recs

    def predict_pp_batch(self, df=None):
        """
        Performs batch prediction.

        df must have column '_vid_'.
        One should only pass in VIDs that have been trained on (see
        :param:`train_attrs`).

        Returns (vid, is_categorical, list[(value, proba)] OR real value (np.array))
            where if is_categorical = True then list[(value, proba)]  is returned.
        """
        if df is None:
            df = self.domain_df

        train_idx_to_attr = {idx: attr for attr, idx in self._dataset._train_attr_idxs.items()}
        n_cats, n_nums = 0, 0

        # Limit max batch size to prevent memory explosion.
        batch_sz = int(1e5 / self._embed_size)
        num_batches = math.ceil(df.shape[0] / batch_sz)
        logging.debug('%s: starting batched (# batches = %d, size = %d) prediction...',
                type(self).__name__, num_batches, batch_sz)
        self.model.set_mode(inference_mode=True)

        # No gradients required.
        with torch.no_grad():
            for vids, is_categorical, attr_idxs, \
                init_cat_idxs, init_numvals, init_nummasks, \
                domain_idxs, domain_masks, \
                target_numvals, cat_targets in tqdm(DataLoader(self._dataset, batch_size=batch_sz, sampler=IterSampler(df['_vid_'].values))):
                pred_cats, pred_nums, cat_mask, num_mask = self.model.forward(is_categorical,
                        attr_idxs,
                        init_cat_idxs,
                        init_numvals,
                        init_nummasks,
                        domain_idxs,
                        domain_masks)

                pred_cat_idx = 0
                pred_num_idx = 0

                for idx, is_cat in enumerate(is_categorical.view(-1)):
                    vid = int(vids[idx, 0])
                    if is_cat:
                        logits = pred_cats[pred_cat_idx]
                        pred_cat_idx += 1
                        n_cats += 1
                        yield vid, bool(is_cat), zip(self._dataset.domain_values(vid), map(float, Softmax(dim=0)(logits)))
                        continue

                    # Real valued prediction

                    # Find the z-score and map it back to its actual value
                    attr = train_idx_to_attr[int(attr_idxs[idx,0])]
                    group_idx = self._dataset._train_num_attrs_group[attr].index(attr)
                    mean = self._dataset._num_attrs_mean[attr]
                    std = self._dataset._num_attrs_std[attr]
                    pred_num = float(pred_nums[pred_num_idx,group_idx]) * std + mean
                    pred_num_idx += 1
                    n_nums += 1
                    yield vid, False, pred_num

        self.model.set_mode(inference_mode=False)
        logging.debug('%s: done batch prediction on %d categorical and %d numerical VIDs.',
                type(self).__name__, n_cats, n_nums)

    def get_features(self, vids):
        """
        Returns three tensors:
            cat_probas: (# of vids, max domain)
            num_predvals: (# of vids, 1)
            is_categorical: (# of vids, 1)
        """
        # No gradients required.
        with torch.no_grad():
            ret_cat_probas = torch.zeros(len(vids), self.model.max_cat_domain)
            ret_num_predvals = torch.zeros(len(vids), 1)
            ret_is_categorical = torch.zeros(len(vids), 1, dtype=torch.uint8)

            batch_sz = int(1e5 / self._embed_size)
            num_batches = math.ceil(len(vids) / batch_sz)
            logging.debug('%s: getting features in batches (# batches = %d, size = %d) ...',
                    type(self).__name__, num_batches, batch_sz)

            mask_offset = 0

            self.model.set_mode(inference_mode=True)
            for vids, is_categorical, attr_idxs, \
                init_cat_idxs, init_numvals, init_nummasks, \
                domain_idxs, domain_masks, \
                target_numvals, cat_targets in tqdm(DataLoader(self._dataset, batch_size=batch_sz, sampler=IterSampler(vids))):

                # (# of cats, max cat domain), (# of num, max_num_dim)
                cat_logits, num_predvals, cat_masks, num_masks = self.model.forward(is_categorical,
                        attr_idxs,
                        init_cat_idxs,
                        init_numvals,
                        init_nummasks,
                        domain_idxs,
                        domain_masks)

                if cat_logits.nelement():
                    cat_probas = Softmax(dim=1)(cat_logits)
                else:
                    cat_probas = cat_logits

                # (# of cats), (# of num)
                cat_masks.add_(mask_offset)
                num_masks.add_(mask_offset)
                mask_offset += is_categorical.shape[0]
                # (# of num VIDs, 1)
                num_attr_idxs = self.model._num_attr_idxs(is_categorical, attr_idxs)
                num_attr_group_mask = self.model._num_attr_group_mask.index_select(0, num_attr_idxs.view(-1))
                # (# of num VIDS, 1)
                num_predvals_masked = (num_attr_group_mask * num_predvals).sum(dim=1, keepdim=True)

                # write values to return tensor
                ret_cat_probas.scatter_(0, cat_masks.unsqueeze(-1).expand(-1, self.model.max_cat_domain), cat_probas.data)
                ret_num_predvals.scatter_(0, num_masks.unsqueeze(-1), num_predvals_masked.data)
                ret_is_categorical[cat_masks] = 1

                del cat_probas, num_predvals_masked

            self.model.set_mode(inference_mode=False)

            return ret_cat_probas.detach(), ret_num_predvals.detach(), ret_is_categorical.detach()

    def dump_predictions(self, prefix, include_all=False):
        """
        Dump inference results to ":param:`prefix`_predictions.pkl" (if not None).
        Returns the dataframe of results.

        include_all = True will include all domain values and their prediction
        probabilities for categorical attributes.
        """
        preds = self.predict_pp_batch()

        logging.debug('%s: constructing and dumping predictions...',
                      type(self).__name__)
        results = []
        for ((vid, is_cat, pred), row) in zip(preds, self.domain_recs):
            assert vid == row['_vid_']
            if is_cat:
                # Include every domain value and their predicted probabilities
                if include_all:
                    for val, proba in pred:
                        results.append({'tid': row['_tid_'],
                            'vid': vid,
                            'attribute': row['attribute'],
                            'inferred_val': val,
                            'proba': proba})
                else:
                    max_val, max_proba = max(pred, key=lambda t: t[1])
                    results.append({'tid': row['_tid_'],
                        'vid': vid,
                        'attribute': row['attribute'],
                        'inferred_val': max_val,
                        'proba': max_proba})
            else:
                results.append({'tid': row['_tid_'],
                    'vid': vid,
                    'attribute': row['attribute'],
                    'inferred_val': pred,
                    'proba': -1})

        results = pd.DataFrame(results)

        if prefix is not None:
            fpath = '{}_predictions.pkl'.format(prefix)
            logging.debug('%s: dumping predictions to %s', type(self).__name__, fpath)
            results.to_pickle(fpath)
        return results