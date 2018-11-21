import logging
import time
from enum import Enum
import os
import time
import pandas as pd

from .dbengine import DBengine
from .table import Table, Source


def _dictify(frame):
    """
    _dictify converts a frame with columns

      col1    | col2    | .... | coln   | value
      ...
    to a dictionary that maps values valX from colX

    { val1 -> { val2 -> { ... { valn -> value } } } }
    """
    ret = {}
    for row in frame.values:
        cur_level = ret
        for elem in row[:-2]:
            if elem not in cur_level:
                cur_level[elem] = {}
            cur_level = cur_level[elem]
        cur_level[row[-2]] = row[-1]
    return ret


class AuxTables(Enum):
    c_cells        = 1
    dk_cells       = 2
    cell_domain    = 3
    pos_values     = 4
    cell_distr     = 5
    inf_values_idx = 6
    inf_values_dom = 7

class Dataset:
    """
    This class keeps all dataframes and tables for a HC session
    """

    def __init__(self, name, env):
        self.id = name
        self.raw_data = None
        self.repaired_data = None
        self.constraints = None
        self.aux_table = {}
        for tab in AuxTables:
            self.aux_table[tab] = None
        # start dbengine
        self.engine = DBengine(env['db_user'], env['db_pwd'], env['db_name'], env['db_host'], pool_size=env['threads'],
                               timeout=env['timeout'])
        # members to convert (tuple_id, attribute) to cell_id
        self.attr_to_idx = {}
        self.attr_count = 0
        # dataset statistics
        self.stats_ready = False
        # Total tuples
        self.total_tuples = 0
        # Domain stats for single attributes
        self.single_attr_stats = {}
        # Domain stats for attribute pairs
        self.pair_attr_stats = {}

    # TODO(richardwu): load more than just CSV files
    def load_data(self, name, fpath, na_values=None, entity_col=None, src_col=None):
        """
        load_data takes a CSV file of the initial data, adds tuple IDs (_tid_)
        to each row to uniquely identify an 'entity', and generates unique
        index numbers for each attribute/column.

        Creates a table with the user supplied 'name' parameter (e.g. 'hospital').

        :param name: (str) name to initialize dataset with.
        :param fpath: (str) filepath to CSV file.
        :param na_values: (str) value that identifies a NULL value
        :param entity_col: (str) column containing the unique
            identifier/ID of an entity.  For fusion tasks, rows with
            the same ID will be fused together in the output.
            If None, assumes every row is a unique entity.
        :param src_col: (str) if not None, for fusion tasks
            specifies the column containing the source for each "mention" of an
            entity.
        """
        tic = time.clock()
        try:
            # Do not include TID and source column as trainable attributes
            exclude_attr_cols = ['_tid_']
            if src_col is not None:
                exclude_attr_cols.append(src_col)

            # Load raw CSV file/data into a Postgres table 'name' (param).
            self.raw_data = Table(name, Source.FILE, na_values=na_values,
                    exclude_attr_cols=exclude_attr_cols, fpath=fpath)

            df = self.raw_data.df
            # Add _tid_ column to dataset that uniquely identifies an entity.
            # If entity_col is not supplied, use auto-incrementing values.
            # Otherwise we use the entity values directly as _tid_'s.
            if entity_col is None:
                # auto-increment
                df.insert(0, '_tid_', range(0,len(df)))
            else:
                # use entity IDs as _tid_'s directly
                df.rename({entity_col: '_tid_'}, axis='columns', inplace=True)

            # Use '_nan_' to represent NULL values
            df.fillna('_nan_', inplace=True)

            # Call to store to database

            self.raw_data.store_to_db(self.engine.engine)
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))

            # Generate indexes on attribute columns for faster queries
            for attr in self.raw_data.get_attributes():
                # Generate index on attribute
                self.raw_data.create_db_index(self.engine,[attr])

            # Create attr_to_idx dictionary (assign unique index for each attribute)
            # and attr_count (total # of attributes)

            self.attr_to_idx = {attr: idx for idx, attr in enumerate(self.raw_data.get_attributes())}
            self.attr_count = len(self.attr_to_idx)
        except Exception:
            logging.error('loading data for table %s', name)
            raise
        toc = time.clock()
        load_time = toc - tic
        return status, load_time

    def set_constraints(self, constraints):
        self.constraints = constraints

    def generate_aux_table(self, aux_table, df, store=False, index_attrs=False):
        """
        generate_aux_table writes/overwrites the auxiliary table specified by
        'aux_table'.

        It does:
          1. stores/replaces the specified aux_table into Postgres (store=True), AND/OR
          2. sets an index on the aux_table's internal Pandas DataFrame (index_attrs=[<columns>]), AND/OR
          3. creates Postgres indexes for aux_table (store=True and index_attrs=[<columns>])

        :param aux_table: (AuxTable) auxiliary table to generate
        :param df: (DataFrame) dataframe to memoize/store for this auxiliary table
        :param store: (bool) if true, creates/replaces Postgres table for this auxiliary table
        :param index_attrs: (list[str]) list of attributes to create indexes on. If store is true,
        also creates indexes on Postgres table.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name, Source.DF, df=df)
            if store:
                self.aux_table[aux_table].store_to_db(self.engine.engine)
            if index_attrs:
                self.aux_table[aux_table].create_df_index(index_attrs)
            if store and index_attrs:
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
        except Exception:
            logging.error('generating aux_table %s', aux_table.name)
            raise

    def generate_aux_table_sql(self, aux_table, query, index_attrs=False):
        """
        :param aux_table: (AuxTable) auxiliary table to generate
        :param query: (str) SQL query whose result is used for generating the auxiliary table.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name, Source.SQL, table_query=query, db_engine=self.engine)
            if index_attrs:
                self.aux_table[aux_table].create_df_index(index_attrs)
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
        except Exception:
            logging.error('generating aux_table %s', aux_table.name)
            raise

    def get_raw_data(self):
        """
        get_raw_data returns a pandas.DataFrame containing the raw data as it was initially loaded.
        """
        if self.raw_data is None:
            raise Exception('ERROR No dataset loaded')
        return self.raw_data.df

    def get_attributes(self):
        """
        get_attributes return the trainable/learnable attributes (i.e. exclude meta
        columns like _tid_).
        """
        if self.raw_data is None:
            raise Exception('ERROR No dataset loaded')
        return self.raw_data.get_attributes()

    def get_cell_id(self, tuple_id, attr_name):
        """
        get_cell_id returns cell ID: a unique ID for every cell.

        Cell ID: _tid_ * (# of attributes) + attr_idx
        """
        vid = tuple_id*self.attr_count + self.attr_to_idx[attr_name]

        return vid

    def get_statistics(self):
        """
        get_statistics returns:
            1. self.total_tuples (total # of tuples)
            2. self.single_attr_stats ({ attribute -> { value -> count } })
              the frequency (# of entities) of a given attribute-value
            3. self.pair_attr_stats ({ attr1 -> { attr2 -> {val1 -> {val2 -> count } } } })
              where DataFrame contains 3 columns:
                <attr1>: all possible values for attr1 ('val1')
                <attr2>: all values for attr2 that appeared at least once with <val1> ('val2')
                <count>: frequency (# of entities) where attr1: val1 AND attr2: val2
        """
        if not self.stats_ready:
            self.collect_stats()
        stats = (self.total_tuples, self.single_attr_stats, self.pair_attr_stats)
        self.stats_ready = True
        return stats

    def collect_stats(self):
        """
        collect_stats memoizes:
          1. self.single_attr_stats ({ attribute -> { value -> count } })
            the frequency (# of entities) of a given attribute-value
          2. self.pair_attr_stats ({ attr1 -> { attr2 -> {val1 -> {val2 -> count } } } })
            where DataFrame contains 3 columns:
              <attr1>: all possible values for attr1 ('val1')
              <attr2>: all values for attr2 that appeared at least once with <val1> ('val2')
              <count>: frequency (# of entities) where attr1: val1 AND attr2: val2
            Also known as co-occurrence count.
        """

        self.total_tuples = self.get_raw_data().shape[0]
        # Single attribute-value frequency
        for attr in self.get_attributes():
            self.single_attr_stats[attr] = self.get_stats_single(attr)
        # Co-occurence frequency
        for cond_attr in self.get_attributes():
            self.pair_attr_stats[cond_attr] = {}
            for trg_attr in self.get_attributes():
                if trg_attr != cond_attr:
                    self.pair_attr_stats[cond_attr][trg_attr] = self.get_stats_pair(cond_attr,trg_attr)

    def get_stats_single(self, attr):
        """
        Returns a dictionary where the keys possible values for :param attr: and
        the values contain the frequency count of that value for this attribute.
        """
        # need to decode values into unicode strings since we do lookups via
        # unicode strings from Postgres
        return self.get_raw_data()[[attr]].groupby([attr]).size().to_dict()

    def get_stats_pair(self, first_attr, second_attr):
        """
        Returns a dictionary {first_val -> {second_val -> count } } where:
            <first_val>: all possible values for first_attr
            <second_val>: all values for second_attr that appeared at least once with <first_val>
            <count>: frequency (# of entities) where first_attr: <first_val> AND second_attr: <second_val>
        """
        tmp_df = self.get_raw_data()[[first_attr,second_attr]].groupby([first_attr,second_attr]).size().reset_index(name="count")
        return _dictify(tmp_df)

    def get_domain_info(self):
        """
        Returns (number of random variables, count of distinct values across all attributes).
        """

        query = 'SELECT count(_vid_), max(domain_size) FROM %s'%AuxTables.cell_domain.name
        res = self.engine.execute_query(query)
        total_vars = int(res[0][0])
        classes = int(res[0][1])
        return total_vars, classes

    def get_inferred_values(self):
        tic = time.clock()
        query = """
        SELECT  t1._tid_,
                t1.attribute,
                domain[inferred_assignment + 1] AS rv_value
        FROM (
            SELECT  _tid_,
                    attribute,
                    _vid_,
                    current_value,
                    string_to_array(regexp_replace(domain, \'[{{\"\"}}]\', \'\', \'gi\'), \'|||\') AS domain
            FROM {cell_domain}) AS t1,
            {inf_values_idx} AS t2
        WHERE t1._vid_ = t2._vid_
        """.format(cell_domain=AuxTables.cell_domain.name,
                inf_values_idx=AuxTables.inf_values_idx.name)
        self.generate_aux_table_sql(AuxTables.inf_values_dom, query, index_attrs=['_tid_'])
        self.aux_table[AuxTables.inf_values_dom].create_db_index(self.engine, ['attribute'])
        status = "DONE collecting the inferred values."
        toc = time.clock()
        total_time = toc - tic
        return status, total_time

    def get_repaired_dataset(self):
        tic = time.clock()
        init_records = self.raw_data.df.sort_values(['_tid_']).to_records(index=False)
        t = self.aux_table[AuxTables.inf_values_dom]
        repaired_vals = _dictify(t.df.reset_index())
        for tid in repaired_vals:
            for attr in repaired_vals[tid]:
                init_records[tid][attr] = repaired_vals[tid][attr]
        repaired_df = pd.DataFrame.from_records(init_records)
        name = self.raw_data.name+'_repaired'
        self.repaired_data = Table(name, Source.DF, df=repaired_df)
        self.repaired_data.store_to_db(self.engine.engine)
        status = "DONE generating repaired dataset"
        toc = time.clock()
        total_time = toc - tic
        return status, total_time


