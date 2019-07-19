import collections
from functools import lru_cache
import logging
import pandas as pd
import time

import itertools
import numpy as np
from tqdm import tqdm

from dataset import AuxTables, CellStatus
from .estimators import *
from .correlations import compute_norm_cond_entropy_corr
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
        self.cor_strength = env["cor_strength"]
        self.estimator_type = env["estimator_type"]

        self.setup_complete = False
        self.domain = None
        self.total = None
        self.correlations = None
        self.do_quantization = False
        self._corr_attrs = {}
        self.max_sample = max_sample
        self.single_stats = {}
        self.pair_stats = {}

    def setup(self):
        """
        setup initializes the in-memory and Postgres auxiliary tables (e.g.
        'cell_domain', 'pos_values').
        """
        tic = time.time()

        if self.correlations is None:
            self.compute_correlations()
        self.setup_attributes()
        self.domain_df = self.generate_domain()
        self.store_domains(self.domain_df)
        status = "DONE with domain preparation."
        toc = time.time()
        return status, toc - tic

    # TODO(richardwu): move this to Dataset after loading data.
    def compute_correlations(self):
        """
        compute_correlations memoizes to self.correlations; a data structure
        that contains pairwise correlations between attributes (values are treated as
        discrete categories).
        """
        logging.debug("Computing correlations...")
        data_df = self.ds.get_quantized_data() if self.do_quantization \
            else self.ds.get_raw_data()
        self.correlations = compute_norm_cond_entropy_corr(data_df,
                                                           self.ds.get_attributes(),
                                                           self.ds.get_attributes())
        corrs_df = pd.DataFrame.from_dict(self.correlations, orient='columns')
        corrs_df.index.name = 'cond_attr'
        corrs_df.columns.name = 'attr'
        pd.set_option('display.max_columns', len(corrs_df.columns))
        pd.set_option('display.max_rows', len(corrs_df.columns))
        logging.debug("correlations:\n%s", corrs_df)
        logging.debug("summary of correlations:\n%s", corrs_df.describe())

    def store_domains(self, domain):
        """
        store_domains stores the 'domain' DataFrame as the 'cell_domain'
        auxiliary table as well as generates the 'pos_values' auxiliary table,
        a long-format of the domain values, in Postgres.

        pos_values schema:
            _tid_: entity/tuple ID
            _cid_: cell ID
            _vid_: random variable ID (all cells with more than 1 domain value)
        """
        if domain.empty:
            raise Exception("ERROR: Generated domain is empty.")
        self.ds.generate_aux_table(AuxTables.cell_domain, domain, store=True)
        self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_vid_'])
        self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_tid_'])
        self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_cid_'])
        query = "SELECT _vid_, _cid_, _tid_, attribute, a.rv_val, a.val_id from %s , unnest(string_to_array(regexp_replace(domain,\'[{\"\"}]\',\'\',\'gi\'),\'|||\')) WITH ORDINALITY a(rv_val,val_id)" % AuxTables.cell_domain.name
        self.ds.generate_aux_table_sql(AuxTables.pos_values, query, index_attrs=['_tid_', 'attribute'])

    def setup_attributes(self):
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
                    top_cands = [(val2, count/denominator) for (val2, count) in pair_stats[attr1][attr2][val1].items() if count > tau]
                    out[attr1][attr2][val1] = top_cands
        return out

    @lru_cache(maxsize=None)
    def get_corr_attributes(self, attr, thres):
        """
        get_corr_attributes returns attributes from self.correlations
        that are correlated with attr with magnitude at least self.cor_strength
        (init parameter).

        :param attr: (string) the original attribute to get the correlated attributes for.
        :param thres: (float) correlation threshold (absolute) for returned attributes.
        """
        if attr not in self.correlations:
            return []
        attr_correlations = self.correlations[attr]
        return sorted([corr_attr
            for corr_attr, corr_strength in attr_correlations.items()
            if corr_attr != attr and corr_strength >= thres])

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
            init_index: domain index of init_value
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
        raw_df = self.ds.get_quantized_data() if self.do_quantization else self.ds.get_raw_data()
        records = raw_df.to_records()

        dk_lookup = {(val[0], val[1]) for val in self.ds.aux_table[AuxTables.dk_cells].df[['_tid_', 'attribute']].values}

        for row in tqdm(list(records)):
            tid = row['_tid_']
            for attr in self.ds.get_active_attributes():
                init_value, init_value_idx, dom = self.get_domain_cell(attr, row)

                # We will use an estimator model for additional weak labelling
                # below, which requires an initial pruned domain first.
                # Weak labels will be trained on the init values.
                cid = self.ds.get_cell_id(tid, attr)

                # Originally, all cells have a NOT_SET status to be considered
                # in weak labelling.
                cell_status = CellStatus.NOT_SET.value

                if len(dom) <= 1:
                    # Initial value is NULL and we cannot come up with
                    # a domain (note that NULL is not included in the domain)
                    if init_value == NULL_REPR and len(dom) == 0:
                       continue

                    # Not enough domain values, we need to get some random
                    # values (other than 'init_value') for training. However,
                    # this might still get us zero domain values.
                    rand_dom_values = self.get_random_domain(attr, dom)

                    # We still want to add cells with only 1 single value and no
                    # additional random domain # they are required in the output.

                    # Otherwise, just add the random domain values to the domain
                    # and set the cell status accordingly.
                    dom.extend(rand_dom_values)

                    # Set the cell status that this is a single value and was
                    # randomly assigned other values in the domain. These will
                    # not be modified by the estimator.
                    cell_status = CellStatus.SINGLE_VALUE.value

                dom_vals = "|||".join(dom)
                cells.append({"_tid_": tid,
                              "attribute": attr,
                              "_cid_": cid,
                              "_vid_": vid,
                              "domain": dom_vals,
                              "domain_size": len(dom),
                              "init_value": init_value,
                              "init_index": init_value_idx,
                              "weak_label": init_value,
                              "weak_label_idx": init_value_idx,
                              "fixed": cell_status,
                              "is_dk": (tid, attr) in dk_lookup,
                              })
                vid += 1
        domain_df = pd.DataFrame(data=cells).sort_values('_vid_')
        logging.debug('domain size stats: %s', domain_df['domain_size'].describe())
        logging.debug('domain count by attr: %s', domain_df['attribute'].value_counts())
        logging.debug('DONE generating initial set of domain values in %.2fs', time.clock() - tic)

        return domain_df

    def get_domain_cell(self, attr, row):
        """
        get_domain_cell returns a list of all domain values for the given
        entity (row) and attribute. the domain never has null as a possible value.

        we define domain values as values in 'attr' that co-occur with values
        in attributes ('cond_attr') that are correlated with 'attr' at least in
        magnitude of self.cor_strength (init parameter).

        for example:

                cond_attr       |   attr
                h                   b                   <-- current row
                h                   c
                i                   d
                h                   e

        this would produce [b,c,e] as domain values.

        :return: (initial value of entity-attribute, domain values for entity-attribute).
        """

        domain = collections.OrderedDict()
        init_value = row[attr]
        correlated_attributes = self.get_corr_attributes(attr, self.cor_strength)
        # iterate through all correlated attributes and take the top k co-occurrence values
        # for 'attr' with the current row's 'cond_attr' value.
        for cond_attr in correlated_attributes:
            # ignore correlations with index, tuple id or the same attribute.
            if cond_attr == attr or cond_attr == '_tid_':
                continue
            if not self.pair_stats[cond_attr][attr]:
                logging.warning("domain generation could not find pair_statistics between attributes: {}, {}".format(cond_attr, attr))
                continue
            cond_val = row[cond_attr]
            # ignore co-occurrence with a null cond init value since we do not
            # store them.
            # also it does not make sense to retrieve the top co-occuring
            # values with a null value.
            # it is possible for cond_val to not be in pair stats if it only co-occurs
            # with null values.
            if cond_val == NULL_REPR or cond_val not in self.pair_stats[cond_attr][attr]:
                continue

            # update domain with top co-occuring values with the cond init value.
            candidates = self.pair_stats[cond_attr][attr][cond_val]
            for val, freq in candidates:
                if val in domain and domain[val] > freq:
                    continue
                domain[val] = freq

        # We should not have any NULLs since we do not store co-occurring NULL
        # values.
        assert NULL_REPR not in domain

        # Add the initial value to the domain if it is not NULL.
        if init_value != NULL_REPR:
            domain[init_value] = 1

        domain = [val for (val, freq) in reversed(sorted(domain.items(), key=lambda t: t[1]))][:self.max_domain]

        # Convert to ordered list to preserve order.
        domain_lst = sorted(list(domain))

        # Get the index of the initial value.
        # NULL values are not in the domain so we set their index to -1.
        init_value_idx = -1
        if init_value != NULL_REPR:
            init_value_idx = domain_lst.index(init_value)

        return init_value, init_value_idx, domain_lst

    def get_random_domain(self, attr, cur_dom):
        """
        get_random_domain returns a random sample of at most size
        'self.max_sample' of domain values for 'attr' that is NOT in 'cur_dom'.
        """
        domain_pool = set(self.single_stats[attr].keys())
        # We should not have any NULLs since we do not keep track of their
        # counts.
        assert NULL_REPR not in domain_pool
        domain_pool = domain_pool.difference(cur_dom)
        domain_pool = sorted(list(domain_pool))
        size = len(domain_pool)
        if size > 0:
            k = min(self.max_sample, size)
            additional_values = np.random.choice(domain_pool, size=k, replace=False)
        else:
            additional_values = []
        return sorted(additional_values)

    def generate_domain_embedding(self, domain_attrs):
        """
        Simple version of generate_domain (for TupleEmbedding) (no random
        sampling).

        Generates domains for the attributes in domain_attrs.

        :return: DataFrame with columns
            _tid_: entity/tuple ID
            attribute: attribute name
            _cid_: cell ID (one for every cell in the raw data in active attributes)
            _vid_: random variable ID (one for every cell with a domain of at least size 2)
            domain: ||| separated string of domain values
            domain_size: length of domain
            init_value: initial value for this cell
            init_index: domain index of init_value
        """
        self.compute_correlations()
        self.setup_attributes()

        logging.debug('generating initial set of un-pruned domain values...')
        records = self.ds.get_raw_data().to_records()
        vid = 0
        domain_df = None

        cells = []
        for row in tqdm(list(records)):
            tid = row['_tid_']
            for attr in domain_attrs:
                init_value, init_value_idx, dom = self.get_domain_cell(attr, row)
                cid = self.ds.get_cell_id(tid, attr)
                cells.append({"_tid_": tid,
                              "attribute": attr,
                              "_cid_": cid,
                              "_vid_": vid,
                              "domain": "|||".join(dom),
                              "domain_size": len(dom),
                              "init_value": init_value,
                              "init_index": init_value_idx,
                              })
                vid += 1

        if domain_df is not None:
            domain_df = pd.concat([domain_df, pd.DataFrame(data=cells)]).reset_index(drop=True)
        else:
            domain_df = pd.DataFrame(data=cells)

        domain_df = domain_df.sort_values('_vid_')
        return domain_df

    def run_estimator(self):
        """
        Runs weak labelling and domain pruning using estimator on domain
        dataframe.
        """
        self.domain_df = self._run_estimator()
        self.store_domains(self.domain_df)

    def _run_estimator(self):
        # Skip estimator model since we do not require any weak labelling or domain
        # pruning based on estimator's posterior probabilities.
        if self.env['estimator_type'] is None \
                or (self.env['weak_label_thresh'] == 1 \
                and self.env['domain_thresh_2'] == 0):
            return self.domain_df

        domain_df = self.domain_df.sort_values('_vid_')

        # Run pruned domain values from correlated attributes above through
        # estimator model for a naive probability estimation.
        logging.debug('training estimator for estimating domain value probabilities...')
        tic = time.clock()

        logging.debug('using estimator: %s', self.env['estimator_type'])
        estimator = None
        if self.env['estimator_type'] == 'NaiveBayes':
            estimator = NaiveBayes(self.env, self.ds, domain_df, self.correlations)
        elif self.env['estimator_type'] == 'Logistic':
            estimator = Logistic(self.env, self.ds, domain_df)
        elif self.env['estimator_type'] == 'TupleEmbedding':
            estimator = TupleEmbedding(self.env, self.ds, domain_df)
            # Memoize embedding model for later use (e.g. in featurizers).
            self.ds.load_embedding_model(estimator)
        else:
            raise Exception('estimator_type must be one of {NaiveBayes, Logistic, TupleEmbedding}')
        estimator.train(self.env['estimator_epochs'], self.env['estimator_batch_size'])
        logging.debug('DONE training estimator in %.2fs', time.clock() - tic)

        # Predict probabilities for all pruned domain values.
        logging.debug('predicting domain value probabilities from estimator...')
        tic = time.clock()
        preds_by_cell = estimator.predict_pp_batch()
        logging.debug('DONE predictions in %.2f secs, re-constructing cell domain...', time.clock() - tic)

        logging.debug('re-assembling final cell domain table...')
        tic = time.clock()
        # iterate through raw/current data and generate posterior probabilities for
        # weak labelling
        num_weak_labels = 0
        updated_domain_df = []

        # TODO(richardwu): we currently do not do anything with is_cat.
        for (vid, is_cat, preds), row in tqdm(list(zip(preds_by_cell, domain_df.to_records()))):
            # Do not re-label single valued cells OR clean cells.
            if row['fixed'] == CellStatus.SINGLE_VALUE.value or not row['is_dk']:
                updated_domain_df.append(row)
                continue

            # prune domain if any of the values are above our domain_thresh_2
            preds = [[val, proba] for val, proba in preds if proba >= self.domain_thresh_2] or preds

            # cap the maximum # of domain values to self.max_domain based on probabilities.
            domain_values = [val for val, proba in sorted(preds, key=lambda pred: pred[1], reverse=True)[:self.max_domain]]

            # ensure the initial value is included even if its probability is low.
            init_val = row['init_value']
            if init_val not in domain_values and init_val != NULL_REPR:
                domain_values.append(init_val)
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
        domain_df = pd.DataFrame.from_records(updated_domain_df,
                columns=updated_domain_df[0].dtype.names)\
                        .drop('index', axis=1).sort_values('_vid_')
        logging.debug('DONE assembling cell domain table in %.2fs', time.clock() - tic)

        logging.info('number of (additional) weak labels assigned from estimator: %d', num_weak_labels)

        logging.debug('DONE generating domain and weak labels')
        return domain_df
