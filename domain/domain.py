import logging
import pandas as pd
import time
from tqdm import tqdm
import itertools
import random
import math

from dataset import AuxTables, CellStatus
from .estimators import RecurrentLogistic


class DomainEngine:
    def __init__(self, env, dataset, sampling_prob=1.0, max_sample=5):
        """
        :param env: (dict) contains global settings such as verbose
        :param dataset: (Dataset) current dataset
        :param sampling_prob: (float) probability of using a random sample if domain cannot be determined
            from correlated co-attributes
        :param max_sample: (int) maximum # of domain values from a random sample
        """
        self.env = env
        self.ds = dataset
        self.topk = env["pruning_topk"]
        self.weak_label_thresh = env["weak_label_thresh"]
        self.domain_prune_thresh = env["domain_prune_thresh"]
        self.max_domain = env["max_domain"]
        self.setup_complete = False
        self.active_attributes = None
        self.domain = None
        self.total = None
        self.correlations = None
        self.cor_strength = env["cor_strength"]
        self.sampling_prob = sampling_prob
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
        random.seed(self.env['seed'])
        self.find_correlations()
        self.setup_attributes()
        domain = self.generate_domain()
        self.store_domains(domain)
        del domain
        status = "DONE with domain preparation."
        toc = time.time()
        return status, toc - tic

    def find_correlations(self):
        """
        find_correlations memoizes to self.correlations; a DataFrame containing
        the pairwise correlations between attributes (values are treated as
        discrete categories).
        """
        df = self.ds.get_raw_data()[self.ds.get_attributes()].copy()
        # Convert dataset to categories/factors.
        for attr in df.columns:
            df[attr] = df[attr].astype('category').cat.codes
        # Drop columns with only one value and tid column.
        df = df.loc[:, (df != 0).any(axis=0)]
        # Compute correlation across attributes.
        m_corr = df.corr()
        self.correlations = m_corr

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
        tic = time.clock()
        self.pair_stats = self._topk_pair_stats(pair_stats)
        toc = time.clock()
        prep_time = toc - tic
        logging.debug("DONE with pair stats preparation in %.2f secs", prep_time)
        self.setup_complete = True

    def _topk_pair_stats(self, pair_stats):
        """
        _topk_pair_stats converts 'pair_stats' which is a dictionary mapping
            { attr1 -> { attr2 -> {val1 -> {val2 -> count } } } } where
                DataFrame contains 3 columns:
                  <val1>: all possible values for attr1
                  <val2>: all values for attr2 that appeared at least once with <val1>
                  <count>: frequency (# of entities) where attr1: <val1> AND attr2: <val2>

        to a flattened 4-level dictionary { attr1 -> { attr2 -> { val1 -> [Top K list of val2] } } }
        i.e. maps to the Top K co-occurring values for attr2 for a given
        attr1-val1 pair.
        """

        out = {}
        for attr1 in tqdm(pair_stats.keys()):
            out[attr1] = {}
            for attr2 in pair_stats[attr1].keys():
                out[attr1][attr2] = {}
                for val1 in pair_stats[attr1][attr2].keys():
                    denominator = self.single_stats[attr1][val1]
                    # TODO(richardwu): while this computes tau as the topk % of
                    # count, we are actually not guaranteed any pairwise co-occurrence
                    # thresholds exceed this count.
                    # For example suppose topk = 0.1 ("top 10%") but we have
                    # 20 co-occurrence counts i.e. 20 unique val2 with
                    # uniform counts, therefore each co-occurrence count
                    # is actually 1 / 20 = 0.05 of the frequency for val1.
                    tau = float(self.topk*denominator)
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
        return set(itertools.chain(*result))

    def get_corr_attributes(self, attr, thres):
        """
        get_corr_attributes returns attributes from self.correlations
        that are correlated with attr with magnitude at least self.cor_strength
        (init parameter).

        :param thres: (float) correlation threshold (absolute) for returned attributes.
        """
        if attr not in self.correlations:
            return []

        d_temp = self.correlations[attr]
        d_temp = d_temp.abs()
        cor_attrs = [rec[0] for rec in d_temp[d_temp > thres].iteritems() if rec[0] != attr]
        return cor_attrs

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
        # Iterate over dataset rows.
        cells = []
        vid = 0
        records = self.ds.get_raw_data().to_records()
        self.all_attrs = list(records.dtype.names)
        for row in tqdm(list(records)):
            tid = row['_tid_']
            app = []
            for attr in self.active_attributes:
                # TODO(richardwu): relax domain prune here: simply take all
                #   values with at least one co-occurrence. This can be
                #   simulated by setting cor_strength = 0.0
                init_value, dom = self.get_domain_cell(attr, row)
                init_value_idx = dom.index(init_value)
                # We will use an Estimator model for weak labelling below, which requires
                # the full pruned domain first.
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
                        dom.extend(self.get_random_domain(attr, init_value))
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
        domain_df = pd.DataFrame(data=cells)

        logging.info('Generating posteriors for domain values using RecurrentLogistic model...')

        # Use pruned domain from correlated attributes above for Logistic model.
        pruned_domain = {}
        for row in domain_df[['_tid_', 'attribute', 'domain']].to_records():
            pruned_domain[row['_tid_']] = pruned_domain.get(row['_tid_'], {})
            pruned_domain[row['_tid_']][row['attribute']] = row['domain'].split('|||')

        estimator = RecurrentLogistic(self.ds, pruned_domain, self.active_attributes)
        estimator.train(num_recur=1, num_epochs=3, batch_size=self.env['batch_size'])

        logging.info('Generating weak labels from posterior model...')

        # raw records indexed by tid
        raw_records_by_tid = {row['_tid_']: row for row in records}

        logging.info('Predicting posterior probabilities in batch...')
        tic = time.clock()
        domain_records = domain_df.to_records()
        preds_by_cell = estimator.predict_pp_batch(raw_records_by_tid, domain_records)
        logging.info('DONE predictions in %.2f secs, re-constructing cell domain...', time.clock() - tic)

        # iterate through raw/current data and generate posterior probabilities for
        # weak labelling
        num_weak_labels = 0
        updated_domain_df = []
        for preds, row in tqdm(zip(preds_by_cell, domain_records)):
            # no need to modify single value cells
            if row['fixed'] == CellStatus.SINGLE_VALUE.value:
                updated_domain_df.append(row)
                continue

            # prune domain if any of the values are above our domain_prune_thresh
            preds = [[val, proba] for val, proba in preds if proba >= self.domain_prune_thresh] or preds

            # cap the maximum # of domain values to self.max_domain
            domain_values = [val for val, proba in sorted(preds, key=lambda pred: pred[1], reverse=True)[:self.max_domain]]

            # ensure the initial value is included
            if row['init_value'] not in domain_values:
                domain_values.append(row['init_value'])
            # update our memoized domain values for this row again
            row['domain'] = '|||'.join(domain_values)
            row['domain_size'] = len(domain_values)
            row['weak_label_idx'] = domain_values.index(row['weak_label'])
            row['init_index'] = domain_values.index(row['init_value'])

            # Assign weak label if domain value exceeds our weak label threshold
            weak_label, weak_label_prob = max(preds, key=lambda pred: pred[1])

            if weak_label_prob >= self.weak_label_thresh:
                num_weak_labels+=1

                weak_label_idx = domain_values.index(weak_label)
                row['weak_label'] = weak_label
                row['weak_label_idx'] = weak_label_idx
                row['fixed'] = CellStatus.WEAK_LABEL.value

            updated_domain_df.append(row)

        del domain_records
        del domain_df
        del preds_by_cell
        # update our cell domain df with our new updated domain
        domain_df = pd.DataFrame.from_records(updated_domain_df, columns=updated_domain_df[0].dtype.names).drop('index', axis=1).sort_values('_vid_')

        logging.info('number of weak labels assigned: %d', num_weak_labels)

        logging.info('DONE generating domain and weak labels')
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

        domain = set([])
        correlated_attributes = self.get_corr_attributes(attr, self.cor_strength)
        # Iterate through all attributes correlated at least self.cor_strength ('cond_attr')
        # and take the top K co-occurrence values for 'attr' with the current
        # row's 'cond_attr' value.
        for cond_attr in correlated_attributes:
            if cond_attr == attr or cond_attr == 'index' or cond_attr == '_tid_':
                continue
            cond_val = row[cond_attr]
            if not pd.isnull(cond_val):
                if not self.pair_stats[cond_attr][attr]:
                    break
                s = self.pair_stats[cond_attr][attr]
                try:
                    candidates = s[cond_val]
                    domain.update(candidates)
                except KeyError as missing_val:
                    if not pd.isnull(row[attr]):
                        # Error since co-occurrence must be at least 1 (since
                        # the current row counts as one co-occurrence).
                        logging.error('Missing value: {}'.format(missing_val))
                        raise

        # Remove _nan_ if added due to correlated attributes.
        domain.discard('_nan_')
        # Add initial value in domain
        if pd.isnull(row[attr]):
            domain.update({'_nan_'})
            init_value = '_nan_'
        else:
            domain.update({row[attr]})
            init_value = row[attr]
        return init_value, list(domain)

    def get_random_domain(self, attr, cur_value):
        """
        get_random_domain returns a random sample of at most size
        'self.max_sample' of domain values for 'attr' that is NOT 'cur_value'.
        """

        if random.random() > self.sampling_prob:
            return []
        domain_pool = set(self.single_stats[attr].keys())
        domain_pool.discard(cur_value)
        size = len(domain_pool)
        if size > 0:
            k = min(self.max_sample, size)
            additional_values = random.sample(domain_pool, k)
        else:
            additional_values = []
        return additional_values
