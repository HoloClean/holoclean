import pandas as pd
import time
from tqdm import tqdm
import itertools
import random

from dataset import AuxTables
from dataset.dataset import dictify


class DomainEngine:
    def __init__(self, env, dataset, cor_strength = 0.1, sampling_prob=0.3, max_sample=5):
        """
        :param env: (dict) contains global settings such as verbose
        :param dataset: (Dataset) current dataset
        :param cor_strength: (float) correlation magnitude threshold for determining domain values
            from correlated co-attributes
        :param sampling_prob: (float) probability of using a random sample if domain cannot be determined
            from correlated co-attributes
        :param max_sample: (int) maximum # of domain values from a random sample
        """
        self.env = env
        self.ds = dataset
        self.topk = env["pruning_topk"]
        self.verbose = env['verbose']
        self.setup_complete = False
        self.active_attributes = []
        self.raw_data = None
        self.domain = None
        self.total = None
        self.correlations = None
        self.cor_strength = cor_strength
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
        status = "DONE with domain preparation."
        toc = time.time()
        return status, toc - tic

    def find_correlations(self):
        """
        find_correlations memoizes to self.correlations a DataFrame containing
        the pairwise correlations between attributes (values are treated as
        discrete categories).
        """
        self.raw_data = self.ds.get_raw_data().copy()
        # convert dataset to numberic categories
        df = pd.DataFrame()
        for attr in self.raw_data.columns.values:
            df[attr] = self.raw_data[attr].astype('category').cat.codes
        # drop columns with only one value and tid column
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.drop(['_tid_'], axis=1)
        # Computer correlation across attributes
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

        self.ds.generate_aux_table(AuxTables.cell_domain, domain, store=True, index_attrs=['_vid_'])
        self.ds.aux_tables[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_tid_'])
        self.ds.aux_tables[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_cid_'])
        query = "SELECT _vid_, _cid_, _tid_, attribute, a.rv_val, a.val_id from %s , unnest(string_to_array(regexp_replace(domain,\'[{\"\"}]\',\'\',\'gi\'),\'|||\')) WITH ORDINALITY a(rv_val,val_id)" % AuxTables.cell_domain.name
        self.ds.generate_aux_table_sql(AuxTables.pos_values, query, index_attrs=['_tid_', 'attribute'])

    def setup_attributes(self):
        try:
            self.active_attributes = self.get_active_attributes()
        except Exception:
            print("ERROR in domain generation")
            raise
        total, single_stats, pair_stats = self.ds.get_statistics()
        self.total = total
        self.single_stats = single_stats
        tic = time.clock()
        self.pair_stats = self.preproc_pair_stats(pair_stats)
        toc = time.clock()
        if self.verbose:
            prep_time = toc - tic
            print("DONE with pair stats preparation in %.2f secs"%prep_time)
        self.setup_complete = True

    def preproc_pair_stats(self, pair_stats):
        """
        preproc_pair_stats converts 'pair_stats' which is a dictionary mapping
            { attr1 -> { attr2 -> DataFrame } } where
                DataFrame contains 3 columns:
                  <attr1>: all possible values for attr1 ('val1')
                  <attr2>: all values for attr2 that appeared at least once with <val1> ('val2')
                  <count>: frequency (# of entities) where attr1: val1 AND attr2: val2

        to a flattened 4-level dictionary { attr1 -> { attr2 -> { val1 -> [Top K list of val2] } } }
        i.e. maps to the Top K co-occurring values for attr2 for a given
        attr1-val1 pair.
        """

        out = {}
        for key1 in tqdm(pair_stats):
            if key1 == '_tid_':
                continue
            out[key1] = {}
            for key2 in pair_stats[key1]:
                if key1 == '_tid_':
                    continue
                df = pair_stats[key1][key2]
                if not df.empty:
                    out[key1][key2] = dictify(df)
                    for val in out[key1][key2]:
                        denominator = self.single_stats[key1][val]
                        tau = float(self.topk*denominator)
                        top_cands = [k for (k, v) in out[key1][key2][val].items() if v > tau]
                        out[key1][key2][val] = top_cands
                else:
                    out[key1][key2] = {}
        return out

    def get_active_attributes(self):
        """
        get_active_attributes returns the attributes to be modeled.

        These attributes correspond only to attributes that contain at least
        one potentially erroneous cell if error detection was ran.
        """

        # No error detection: fallback to all attributes
        if not self.ds.aux_table_exists(AuxTables.dk_cells):
            return self.ds.get_attributes()

        # Error detector was used: only return attributes that have DK cells.

        query = 'SELECT DISTINCT attribute as attribute FROM %s'%AuxTables.dk_cells.name
        result = self.ds.engine.execute_query(query)
        if not result:
            raise Exception("No attribute contains erroneous cells.")
        # concatenate all lists of lists of attributes
        return list(itertools.chain(*result))

    def get_corr_attributes(self, attr):
        """
        get_corr_attributes returns attributes from self.correlations
        that are correlated with attr with magnitude at least self.cor_strength
        (init parameter).
        """
        if attr not in self.correlations:
            return []

        d_temp = self.correlations[attr]
        d_temp = d_temp.abs()
        cor_attrs = [rec[0] for rec in d_temp[d_temp > self.cor_strength].iteritems() if rec[0] != attr]
        return cor_attrs

    def generate_domain(self):
        """
        Generate the domain for each cell in the active attributes as well
        as assigns variable IDs (_vid_) (increment key from 0 onwards, depends on
        iteration order of rows/entities in raw data and attributes.

        Note that _vid_ has a 1-1 correspondence with _cid_.

        See _get_init_domain for how the domain is generated from co-occurrence
        and correlated attributes.

        If no values can be found from correlated attributes, return a random
        sample of domain values.

        :return: DataFrame with columns
            _tid_: entity/tuple ID
            _cid_: cell ID (unique for every entity-attribute)
            _vid_: variable ID (1-1 correspondence with _cid_)
            attribute: attribute name
            domain: ||| seperated string of domain values
            domain_size: length of domain
            init_value: initial value for this cell
            init_value_idx: domain index of init_value
            fixed: 1 if a random sample was taken since no correlated attributes/top K values
        """

        if not self.setup_complete:
            raise Exception(
                "Call <setup_attributes> to setup active attributes. Error detection should be performed before setup.")

        all_cells = []
        vid = 0
        df_raw = self.ds.get_raw_data().set_index('_tid_')
        self.all_attrs = self.ds.get_attributes()
        # Iterate through every entity in the raw data and create their
        # corresponding row in cell_domain.
        for tid in tqdm(list(df_raw.index.unique())):
            for attr in self.active_attributes:
                init_value, dom = self._get_init_domain(df_raw, attr, tid)
                init_value_idx = dom.index(init_value)
                cid = self.ds.get_cell_id(tid, attr)
                if len(dom) > 1:
                    all_cells.append({"_tid_": tid, "attribute": attr, "_cid_": cid, "_vid_":vid, "domain": "|||".join(dom),  "domain_size": len(dom),
                                "init_value": init_value, "init_index": init_value_idx, "fixed":0})
                    vid += 1
                else:
                    add_domain = self._get_random_domain(attr,init_value)
                    # Check if attribute has more than one unique values
                    if len(add_domain) > 0:
                        dom.extend(add_domain)
                        init_value_idx = dom.index(init_value)
                        all_cells.append({"_tid_": tid, "attribute": attr, "_cid_": cid, "_vid_": vid, "domain": "|||".join(dom),
                                    "domain_size": len(dom),
                                    "init_value": init_value, "init_index": init_value_idx, "fixed": 1})
                        vid += 1
        domain_df = pd.DataFrame(data=all_cells)
        print('DONE generating domain')
        return domain_df

    def _get_init_domain(self, df_raw, attr, tid):
        """
        _get_init_domain returns the initial (i.e. assumed to be "correct")
        value and a list of all domain values for the given entity and
        attribute.

        :param df_raw: (pandas.DataFrame) each row corresponds to a mention of values for
        an entity. Must be indexed on the TID.

        :return: (initial value of entity-attribute, domain values for entity-attribute)
        """
        rows = df_raw.loc[tid]

        # Only one row for this entity: assume we need to repair this cell's value
        # TODO(richardwu): should fusion datasets with only one mention for
        # a given entity really default to repairing? (i.e. current logic)
        if isinstance(rows, pd.Series):
            return self._get_repair_domain(attr, rows)

        # More than one row for this entity: default to fusion task
        return self._get_fusion_domain(attr, rows)

    def _get_repair_domain(self, attr, row):
        """
        _get_repair_domain returns the initial value and a list of all domain
        values for the given entity and attribute where the primary goal
        is to repair the entity value for :param attr:.

        We define domain values as values in :param attr: that co-occur with
        values in attributes ('cond_attr') that are correlated with :param
        attr: at least in magnitude of self.cor_strength (init parameter).

        For example:

                cond_attr       |   attr
                H                   B                   <-- current row
                H                   C
                I                   D
                H                   E

        This would produce [B,C,E] as domain values.

        :param attr: (str) attribute to generate initial value and domain values for.
        :param row: (pandas.Series) row containing this entity's values indexed on attribute.

        :return: (initial value of entity-attribute, domain values for entity-attribute)
        """

        domain = set([])
        correlated_attributes = self.get_corr_attributes(attr)
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
                    domain.update(set(candidates))
                except KeyError as missing_val:
                    if pd.isnull(row[attr]):
                        pass
                    else:
                        # error since co-occurrence must be at least 1 (since the
                        # current row counts as one co-occurrence).
                        raise Exception(missing_val)

        # Remove _nan_ if added due to correlated attributes
        domain.discard('_nan_')
        # Add initial value in domain
        if pd.isnull(row[attr]):
            domain.update(set(['_nan_']))
            init_value = '_nan_'
        else:
            domain.update(set([row[attr]]))
            init_value = row[attr]
        return init_value, list(domain)

    def _get_fusion_domain(self, attr, rows):
        """
        _get_fusion_domain returns the initial value and a list of all domain
        values for the given entity and attribute where the primary goal
        is to fuse the multiple mention of values for :param attr:.

        Domain values are simply the set of values in amongst all mentions for
        a given entity-attribute pair.

        Initial value is taken as the majority value (randomly sample one
        for tiebreakers).

        :param attr: (str) attribute to generate initial value and domain values for.
        :param rows: (pandas.DataFrame) rows each corresponding to a mention of
            an entity's values. Columns should be attributes. The '_tid_' column
            must exist.

        :return: (initial value of entity-attribute, domain values for entity-attribute)
        """
        # Get the majority value (sample if multiple)
        init_values = rows[attr].value_counts(sort=True, ascending=False)
        init_val = init_values[init_values == init_values.max()].sample(random_state=self.env['seed'])
        return init_val.index[0], list(rows[attr].unique())

    def _get_random_domain(self, attr, cur_value):
        """
        _get_random_domain returns a random sample of at most size
        'self.max_sample' of domain values for 'attr' that is NOT 'cur_value'.
        """

        if random.random() > self.sampling_prob:
            return []
        domain_pool = set(self.single_stats[attr].index.astype(str))
        domain_pool.discard(cur_value)
        size = len(domain_pool)
        if size > 0:
            k = min(self.max_sample, size)
            additional_values = random.sample(domain_pool, k)
        else:
            additional_values = []
        return additional_values
