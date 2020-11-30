import logging
import pandas as pd
import time

import itertools
import numpy as np
from pyitlib import discrete_random_variable as drv
from tqdm import tqdm

from dataset import AuxTables, CellStatus
from .estimators import NaiveBayes
from utils import NULL_REPR


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
        tic = time.time()
        self.pair_stats = self._pruned_pair_stats(pair_stats)
        logging.debug("DONE with pruned co-occurring statistics in %.2f secs", time.time() - tic)
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

        :param attr: (string) the original attribute to get the correlated attributes for.
        :param thres: (float) correlation threshold (absolute) for returned attributes.
        """
        # Not memoized: find correlated attributes from correlation dictionary.
        if (attr, thres) not in self._corr_attrs:
            self._corr_attrs[(attr, thres)] = []

            if attr in self.correlations:
                attr_correlations = self.correlations[attr]
                self._corr_attrs[(attr, thres)] = sorted([corr_attr
                                                   for corr_attr, corr_strength in attr_correlations.items()
                                                   if corr_attr != attr and corr_strength > thres])

        return self._corr_attrs[(attr, thres)]

    def generate_domain(self):
        """
        Generates the domain for each cell in the active attributes as well
        as assigns a random variable ID (_vid_) for cells that have
        a domain of size >= 2.

        See get_domain_cell for how the domain is generated from co-occurrence
        and correlated attributes.

        If no values can be found from correlated attributes, return a random
        sample of domain values.

        :return: DataFrame with columns
            _tid_: entity/tuple ID
            _cid_: cell ID (one for every cell in the raw data in active attributes)
            _vid_: random variable ID (one for every cell with a domain of at least size 2)
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
        tic = time.time()
        # Iterate over dataset rows.
        cells = []
        vid = 0
        records = self.ds.get_raw_data().to_records()
        self.all_attrs = list(records.dtype.names)
        for row in tqdm(list(records)):
            tid = row['_tid_']
            for attr in self.active_attributes:
                init_value, init_value_idx, dom = self.get_domain_cell(attr, row)
                # We will use an estimator model for additional weak labelling
                # below, which requires an initial pruned domain first.
                # Weak labels will be trained on the init values.
                cid = self.ds.get_cell_id(tid, attr)

                # Originally, all cells have a NOT_SET status to be considered
                # in weak labelling.
                cell_status = CellStatus.NOT_SET.value

                if len(dom) <= 1:
                    # Initial  value is NULL and we cannot come up with
                    # a domain; a random domain probably won't help us so
                    # completely ignore this cell and continue.
                    # Note if len(dom) == 1, then we generated a single correct
                    # value (since NULL is not included in the domain).
                    # This would be a "SINGLE_VALUE" example and we'd still
                    # like to generate a random domain for it.
                    if init_value == NULL_REPR and len(dom) == 0:
                        continue

                    # Not enough domain values, we need to get some random
                    # values (other than 'init_value') for training. However,
                    # this might still get us zero domain values.
                    rand_dom_values = self.get_random_domain(attr, init_value)

                    # rand_dom_values might still be empty. In this case,
                    # there are no other possible values for this cell. There
                    # is not point to use this cell for training and there is no
                    # point to run inference on it since we cannot even generate
                    # a random domain. Therefore, we just ignore it from the
                    # final tensor.
                    if len(rand_dom_values) == 0:
                        continue

                    # Otherwise, just add the random domain values to the domain
                    # and set the cell status accordingly.
                    dom.extend(rand_dom_values)

                    # Set the cell status that this is a single value and was
                    # randomly assigned other values in the domain. These will
                    # not be modified by the estimator.
                    cell_status = CellStatus.SINGLE_VALUE.value

                cells.append({"_tid_": tid,
                              "attribute": attr,
                              "_cid_": cid,
                              "_vid_": vid,
                              "domain": "|||".join(dom),
                              "domain_size": len(dom),
                              "init_value": init_value,
                              "init_index": init_value_idx,
                              "weak_label": init_value,
                              "weak_label_idx": init_value_idx,
                              "fixed": cell_status})
                vid += 1
        domain_df = pd.DataFrame(data=cells).sort_values('_vid_')
        logging.debug('DONE generating initial set of domain values in %.2f', time.time() - tic)

        # Skip estimator model since we do not require any weak labelling or domain
        # pruning based on posterior probabilities.
        if self.env['weak_label_thresh'] == 1 and self.env['domain_thresh_2'] == 0:
            return domain_df

        # Run pruned domain values from correlated attributes above through
        # posterior model for a naive probability estimation.
        logging.debug('training posterior model for estimating domain value probabilities...')
        tic = time.time()
        estimator = NaiveBayes(self.env, self.ds, domain_df, self.correlations)
        logging.debug('DONE training posterior model in %.2fs', time.time() - tic)

        # Predict probabilities for all pruned domain values.
        logging.debug('predicting domain value probabilities from posterior model...')
        tic = time.time()
        preds_by_cell = estimator.predict_pp_batch()
        logging.debug('DONE predictions in %.2f secs, re-constructing cell domain...', time.time() - tic)

        logging.debug('re-assembling final cell domain table...')
        tic = time.time()
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
            if row['init_value'] not in domain_values and row['init_value'] != NULL_REPR:
                domain_values.append(row['init_value'])
            domain_values = sorted(domain_values)
            # update our memoized domain values for this row again
            row['domain'] = '|||'.join(domain_values)
            row['domain_size'] = len(domain_values)
            # update init index based on new domain
            if row['init_value'] in domain_values:
                row['init_index'] = domain_values.index(row['init_value'])
            # update weak label index based on new domain
            if row['weak_label'] != NULL_REPR:
                row['weak_label_idx'] = domain_values.index(row['weak_label'])

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
        logging.debug('DONE assembling cell domain table in %.2fs', time.time() - tic)

        logging.info('number of (additional) weak labels assigned from posterior model: %d', num_weak_labels)

        logging.debug('DONE generating domain and weak labels')
        return domain_df

    def get_domain_cell(self, attr, row):
        """
        get_domain_cell returns a list of all domain values for the given
        entity (row) and attribute. The domain never has null as a possible value.

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
        init_value = row[attr]
        correlated_attributes = self.get_corr_attributes(attr, self.cor_strength)
        # Iterate through all correlated attributes and take the top K co-occurrence values
        # for 'attr' with the current row's 'cond_attr' value.
        for cond_attr in correlated_attributes:
            # Ignore correlations with index, tuple id or the same attribute.
            if cond_attr == attr or cond_attr == '_tid_':
                continue
            if not self.pair_stats[cond_attr][attr]:
                logging.warning("domain generation could not find pair_statistics between attributes: {}, {}".format(cond_attr, attr))
                continue
            cond_val = row[cond_attr]
            # Ignore co-occurrence with a NULL cond init value since we do not
            # store them.
            # Also it does not make sense to retrieve the top co-occuring
            # values with a NULL value.
            # It is possible for cond_val to not be in pair stats if it only co-occurs
            # with NULL values.
            if cond_val == NULL_REPR or cond_val not in self.pair_stats[cond_attr][attr]:
                continue

            # Update domain with top co-occuring values with the cond init value.
            candidates = self.pair_stats[cond_attr][attr][cond_val]
            domain.update(candidates)

        # We should not have any NULLs since we do not store co-occurring NULL
        # values.
        assert NULL_REPR not in domain

        # Add the initial value to the domain if it is not NULL.
        if init_value != NULL_REPR:
            domain.add(init_value)

        # Convert to ordered list to preserve order.
        domain_lst = sorted(list(domain))

        # Get the index of the initial value.
        # NULL values are not in the domain so we set their index to -1.
        init_value_idx = -1
        if init_value != NULL_REPR:
            init_value_idx = domain_lst.index(init_value)

        return init_value, init_value_idx, domain_lst

    def get_random_domain(self, attr, cur_value):
        """
        get_random_domain returns a random sample of at most size
        'self.max_sample' of domain values for 'attr' that is NOT 'cur_value'.
        """
        domain_pool = set(self.single_stats[attr].keys())
        # We should not have any NULLs since we do not keep track of their
        # counts.
        assert NULL_REPR not in domain_pool
        domain_pool.discard(cur_value)
        domain_pool = sorted(list(domain_pool))
        size = len(domain_pool)
        if size > 0:
            k = min(self.max_sample, size)
            additional_values = np.random.choice(domain_pool, size=k, replace=False)
        else:
            additional_values = []
        return sorted(additional_values)
