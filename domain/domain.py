import logging
import pandas as pd
import time

import itertools
import numpy as np
from pyitlib import discrete_random_variable as drv
from tqdm import tqdm

from dataset import AuxTables, CellStatus
from .estimators import NaiveBayes


class DomainEngine:
    def __init__(self, env, dataset, max_sample=5):
        """
        :param env: (dict) contains global settings such as verbose
        :param dataset: (Dataset) current dataset
        :param max_sample: (int) maximum # of domain values from a random sample
        """
        self.env = env
        self.ds = dataset
        self.domain_thresh_1 = env["domain_thresh_1"]
        self.weak_label_thresh = env["weak_label_thresh"]
        self.domain_thresh_2 = env["domain_thresh_2"]
        self.max_domain = env["max_domain"]
        self.setup_complete = False
        self.active_attributes = None
        self.domain = None
        self.total = None
        self.correlations = None
        self._corr_attrs = {}
        self.cor_strength = env["cor_strength"]
        self.max_sample = max_sample
        self.single_stats = {}
        self.pair_stats = {}
        self.all_attrs = {}

    def setup(self):
        """
        setup initializes the in-memory and Postgres auxiliary tables (e.g.
        'cell_domain', 'pos_values').
        """
        tic = time.time()
        self.compute_correlations()
        self.setup_attributes()
        domain = self.generate_domain()
        self.store_domains(domain)
        status = "DONE with domain preparation."
        toc = time.time()
        return status, toc - tic

    def compute_correlations(self):
        """
        compute_correlations memoizes to self.correlations; a data structure
        that contains pairwise correlations between attributes (values are treated as
        discrete categories).
        """
        self.correlations = self._compute_norm_cond_entropy_corr()

    def _compute_norm_cond_entropy_corr(self):
        """
        Computes the correlations between attributes by calculating
        the normalized conditional entropy between them. The conditional
        entropy is asymmetric, therefore we need pairwise computation.

        The computed correlations are stored in a dictionary in the format:
        {
          attr_a: { cond_attr_i: corr_strength_a_i,
                    cond_attr_j: corr_strength_a_j, ... },
          attr_b: { cond_attr_i: corr_strength_b_i, ...}
        }

        :return a dictionary of correlations
        """
        data_df = self.ds.get_raw_data()
        attrs = self.ds.get_attributes()

        corr = {}
        # Compute pair-wise conditional entropy.
        for x in attrs:
            corr[x] = {}
            x_vals = data_df[x]
            x_domain_size = x_vals.nunique()
            for y in attrs:
                # Set correlation to 0.0 if entropy of x is 1 (only one possible value).
                if x_domain_size == 1:
                    corr[x][y] = 0.0
                    continue

                # Set correlation to 1 for same attributes.
                if x == y:
                    corr[x][y] = 1.0
                    continue

                # Compute the conditional entropy H(x|y) = H(x,y) - H(y).
                # H(x,y) denotes H(x U y).
                # If H(x|y) = 0, then y determines x, i.e., y -> x.
                # Use the domain size of x as a log base for normalization.
                y_vals = data_df[y]
                x_y_entropy = drv.entropy_conditional(x_vals, y_vals, base=x_domain_size)

                # The conditional entropy is 0 for strongly correlated attributes and 1 for
                # completely independent attributes. We reverse this to reflect the correlation.
                corr[x][y] = 1.0 - x_y_entropy
        return corr

    def store_domains(self, domain):
        """
        store_domains stores the 'domain' DataFrame as the 'cell_domain'
        auxiliary table as well as generates the 'pos_values' auxiliary table,
        a long-format of the domain values, in Postgres.

        pos_values schema:
            _tid_: entity/tuple ID
            _cid_: cell ID
            _vid_: random variable ID (all cells with more than 1 domain value)
            _

        """
        if domain.empty:
            raise Exception("ERROR: Generated domain is empty.")
        else:
            self.ds.generate_aux_table(AuxTables.cell_domain, domain, store=True, index_attrs=['_vid_'])
            self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_tid_'])
            self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_cid_'])
            query = "SELECT _vid_, _cid_, _tid_, attribute, a.rv_val, a.val_id from %s , unnest(string_to_array(regexp_replace(domain,\'[{\"\"}]\',\'\',\'gi\'),\'|||\')) WITH ORDINALITY a(rv_val,val_id)" % AuxTables.cell_domain.name
            self.ds.generate_aux_table_sql(AuxTables.pos_values, query, index_attrs=['_tid_', 'attribute'])

    def setup_attributes(self):
        self.active_attributes = self.get_active_attributes()
        total, single_stats, pair_stats = self.ds.get_statistics()
        self.total = total
        self.single_stats = single_stats
        logging.debug("preparing pruned co-occurring statistics...")
        tic = time.clock()
        self.pair_stats = self._pruned_pair_stats(pair_stats)
        logging.debug("DONE with pruned co-occurring statistics in %.2f secs", time.clock() - tic)
        self.setup_complete = True

    def _pruned_pair_stats(self, pair_stats):
        """
        _pruned_pair_stats converts 'pair_stats' which is a dictionary mapping
            { attr1 -> { attr2 -> {val1 -> {val2 -> count } } } } where
              <val1>: all possible values for attr1
              <val2>: all values for attr2 that appeared at least once with <val1>
              <count>: frequency (# of entities) where attr1: <val1> AND attr2: <val2>

        to a flattened 4-level dictionary { attr1 -> { attr2 -> { val1 -> [pruned list of val2] } } }
        i.e. maps to the co-occurring values for attr2 that exceed
        the self.domain_thresh_1 co-occurrence probability for a given
        attr1-val1 pair.
        """

        out = {}
        for attr1 in tqdm(pair_stats.keys()):
            out[attr1] = {}
            for attr2 in pair_stats[attr1].keys():
                out[attr1][attr2] = {}
                for val1 in pair_stats[attr1][attr2].keys():
                    denominator = self.single_stats[attr1][val1]
                    # tau becomes a threshhold on co-occurrence frequency
                    # based on the co-occurrence probability threshold
                    # domain_thresh_1.
                    tau = float(self.domain_thresh_1*denominator)
                    top_cands = [val2 for (val2, count) in pair_stats[attr1][attr2][val1].items() if count > tau]
                    out[attr1][attr2][val1] = top_cands
        return out

    def get_active_attributes(self):
        """
        get_active_attributes returns the attributes to be modeled.
        These attributes correspond only to attributes that contain at least
        one potentially erroneous cell.
        """
        query = 'SELECT DISTINCT attribute as attribute FROM {}'.format(AuxTables.dk_cells.name)
        result = self.ds.engine.execute_query(query)
        if not result:
            raise Exception("No attribute contains erroneous cells.")
        # Sort the active attributes to maintain the order of the ids of random variable.
        return sorted(itertools.chain(*result))

    def get_corr_attributes(self, attr, thres):
        """
        get_corr_attributes returns attributes from self.correlations
        that are correlated with attr with magnitude at least self.cor_strength
        (init parameter).

        :param thres: (float) correlation threshold (absolute) for returned attributes.
        """
        # Not memoized: find correlated attributes from correlation dictionary.
        if (attr, thres) not in self._corr_attrs:
            self._corr_attrs[(attr,thres)] = []

            if attr in self.correlations:
                attr_correlations = self.correlations[attr]
                self._corr_attrs[(attr, thres)] = sorted([corr_attr
                                                   for corr_attr, corr_strength in attr_correlations.items()
                                                   if corr_attr != attr and corr_strength > thres])

        return self._corr_attrs[(attr, thres)]

    def generate_domain(self):
        """
        Generates the domain for each cell in the active attributes as well
        as assigns variable IDs (_vid_) (increment key from 0 onwards, depends on
        iteration order of rows/entities in raw data and attributes.

        Note that _vid_ has a 1-1 correspondence with _cid_.

        See get_domain_cell for how the domain is generated from co-occurrence
        and correlated attributes.

        If no values can be found from correlated attributes, return a random
        sample of domain values.

        :return: DataFrame with columns
            _tid_: entity/tuple ID
            _cid_: cell ID (unique for every entity-attribute)
            _vid_: variable ID (1-1 correspondence with _cid_)
            attribute: attribute name
            domain: ||| separated string of domain values
            domain_size: length of domain
            init_value: initial value for this cell
            init_value_idx: domain index of init_value
            fixed: 1 if a random sample was taken since no correlated attributes/top K values
        """

        if not self.setup_complete:
            raise Exception(
                "Call <setup_attributes> to setup active attributes. Error detection should be performed before setup.")

        logging.debug('generating initial set of un-pruned domain values...')
        tic = time.clock()
        # Iterate over dataset rows.
        cells = []
        vid = 0
        records = self.ds.get_raw_data().to_records()
        self.all_attrs = list(records.dtype.names)
        for row in tqdm(list(records)):
            tid = row['_tid_']
            app = []
            for attr in self.active_attributes:
                init_value, init_value_idx, dom = self.get_domain_cell(attr, row)
                # We will use an estimator model for additional weak labelling
                # below, which requires an initial pruned domain first.
                weak_label = init_value
                weak_label_idx = init_value_idx
                if len(dom) > 1:
                    cid = self.ds.get_cell_id(tid, attr)
                    app.append({"_tid_": tid,
                                "attribute": attr,
                                "_cid_": cid,
                                "_vid_": vid,
                                "domain": "|||".join(dom),
                                "domain_size": len(dom),
                                "init_value": init_value,
                                "init_index": init_value_idx,
                                "weak_label": weak_label,
                                "weak_label_idx": weak_label_idx,
                                "fixed": CellStatus.NOT_SET.value})
                    vid += 1
                else:
                    add_domain = self.get_random_domain(attr, init_value)
                    # Check if attribute has more than one unique values.
                    if len(add_domain) > 0:
                        dom.extend(add_domain)
                        cid = self.ds.get_cell_id(tid, attr)
                        app.append({"_tid_": tid,
                                    "attribute": attr,
                                    "_cid_": cid,
                                    "_vid_": vid,
                                    "domain": "|||".join(dom),
                                    "domain_size": len(dom),
                                    "init_value": init_value,
                                    "init_index": init_value_idx,
                                    "weak_label": init_value,
                                    "weak_label_idx": init_value_idx,
                                    "fixed": CellStatus.SINGLE_VALUE.value})
                        vid += 1
            cells.extend(app)
        domain_df = pd.DataFrame(data=cells).sort_values('_vid_')
        logging.debug('DONE generating initial set of domain values in %.2f', time.clock() - tic)

        # Skip estimator model since we do not require any weak labelling or domain
        # pruning based on posterior probabilities.
        if self.env['weak_label_thresh'] == 1 and self.env['domain_thresh_2'] == 0:
            return domain_df

        # Run pruned domain values from correlated attributes above through
        # posterior model for a naive probability estimation.
        logging.debug('training posterior model for estimating domain value probabilities...')
        tic = time.clock()
        estimator = NaiveBayes(self.env, self.ds, domain_df, self.correlations)
        logging.debug('DONE training posterior model in %.2fs', time.clock() - tic)

        # Predict probabilities for all pruned domain values.
        logging.debug('predicting domain value probabilities from posterior model...')
        tic = time.clock()
        preds_by_cell = estimator.predict_pp_batch()
        logging.debug('DONE predictions in %.2f secs, re-constructing cell domain...', time.clock() - tic)

        logging.debug('re-assembling final cell domain table...')
        tic = time.clock()
        # iterate through raw/current data and generate posterior probabilities for
        # weak labelling
        num_weak_labels = 0
        updated_domain_df = []
        for preds, row in tqdm(zip(preds_by_cell, domain_df.to_records())):
            # Do not re-label single valued cells.
            if row['fixed'] == CellStatus.SINGLE_VALUE.value:
                updated_domain_df.append(row)
                continue

            # prune domain if any of the values are above our domain_thresh_2
            preds = [[val, proba] for val, proba in preds if proba >= self.domain_thresh_2] or preds

            # cap the maximum # of domain values to self.max_domain based on probabilities.
            domain_values = [val for val, proba in sorted(preds, key=lambda pred: pred[1], reverse=True)[:self.max_domain]]

            # ensure the initial value is included even if its probability is low.
            if row['init_value'] not in domain_values:
                domain_values.append(row['init_value'])
            domain_values = sorted(domain_values)
            # update our memoized domain values for this row again
            row['domain'] = '|||'.join(domain_values)
            row['domain_size'] = len(domain_values)
            row['weak_label_idx'] = domain_values.index(row['weak_label'])
            row['init_index'] = domain_values.index(row['init_value'])

            weak_label, weak_label_prob = max(preds, key=lambda pred: pred[1])

            # Assign weak label if it is not the same as init AND domain value
            # exceeds our weak label threshold.
            if weak_label != row['init_value'] and weak_label_prob >= self.weak_label_thresh:
                num_weak_labels += 1

                weak_label_idx = domain_values.index(weak_label)
                row['weak_label'] = weak_label
                row['weak_label_idx'] = weak_label_idx
                row['fixed'] = CellStatus.WEAK_LABEL.value

            updated_domain_df.append(row)

        # update our cell domain df with our new updated domain
        domain_df = pd.DataFrame.from_records(updated_domain_df, columns=updated_domain_df[0].dtype.names).drop('index', axis=1).sort_values('_vid_')
        logging.debug('DONE assembling cell domain table in %.2fs', time.clock() - tic)

        logging.info('number of (additional) weak labels assigned from posterior model: %d', num_weak_labels)

        logging.debug('DONE generating domain and weak labels')
        return domain_df

    def get_domain_cell(self, attr, row):
        """
        get_domain_cell returns a list of all domain values for the given
        entity (row) and attribute.

        We define domain values as values in 'attr' that co-occur with values
        in attributes ('cond_attr') that are correlated with 'attr' at least in
        magnitude of self.cor_strength (init parameter).

        For example:

                cond_attr       |   attr
                H                   B                   <-- current row
                H                   C
                I                   D
                H                   E

        This would produce [B,C,E] as domain values.

        :return: (initial value of entity-attribute, domain values for entity-attribute).
        """

        domain = set()
        correlated_attributes = self.get_corr_attributes(attr, self.cor_strength)
        # Iterate through all attributes correlated at least self.cor_strength ('cond_attr')
        # and take the top K co-occurrence values for 'attr' with the current
        # row's 'cond_attr' value.
        for cond_attr in correlated_attributes:
            # Ignore correlations with index, tuple id or the same attribute.
            if cond_attr == attr or cond_attr == '_tid_':
                continue
            if not self.pair_stats[cond_attr][attr]:
                logging.warning("domain generation could not find pair_statistics between attributes: {}, {}".format(cond_attr, attr))
                continue
            cond_val = row[cond_attr]
            # Ignore correlations with null values.
            if cond_val == '_nan_':
                continue
            s = self.pair_stats[cond_attr][attr]
            try:
                candidates = s[cond_val]
                domain.update(candidates)
            except KeyError as missing_val:
                if row[attr] != '_nan_':
                    # Error since co-occurrence must be at least 1 (since
                    # the current row counts as one co-occurrence).
                    logging.error('value missing from statistics: {}'.format(missing_val))
                    raise

        # Add the initial value to the domain.
        init_value = row[attr]
        domain.add(init_value)

        # Remove _nan_ if added due to correlated attributes, only if it was not the initial value.
        if init_value != '_nan_':
            domain.discard('_nan_')

        # Convert to ordered list to preserve order.
        domain_lst = sorted(list(domain))

        # Get the index of the initial value. This should never raise a ValueError since we made sure
        # that 'init_value' was added.
        init_value_idx = domain_lst.index(init_value)

        return init_value, init_value_idx, domain_lst

    def get_random_domain(self, attr, cur_value):
        """
        get_random_domain returns a random sample of at most size
        'self.max_sample' of domain values for 'attr' that is NOT 'cur_value'.
        """
        domain_pool = set(self.single_stats[attr].keys())
        domain_pool.discard(cur_value)
        domain_pool = sorted(list(domain_pool))
        size = len(domain_pool)
        if size > 0:
            k = min(self.max_sample, size)
            additional_values = np.random.choice(domain_pool, size=k, replace=False)
        else:
            additional_values = []
        return sorted(additional_values)
