import time
from enum import Enum
import pandas as pd
from .dbengine import DBengine
from .table import Table, Source


def dictify(frame):
    """
    dictify converts a frame with columns

      col1    | col2    | .... | coln   | value
      ...
    to a dictionary that maps values valX from colX

    { val1 -> { val2 -> { ... { valn -> value } } } }
    """
    d = {}
    for row in frame.values:
        here = d
        for elem in row[:-2]:
            if elem not in here:
                here[elem] = {}
            here = here[elem]
        here[row[-2]] = row[-1]
    return d


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
                               verbose=env['verbose'], timeout=env['timeout'])
        # members to convert (tuple_id, attribute) to cell_id
        self.attr_to_idx = {}
        self.attr_number = 0
        # dataset statistics
        self.stats_ready = False
        # Total tuples
        self.total_tuples = 0
        # Domain stats for single attributes
        self.single_attr_stats = {}
        # Domain stats for attribute pairs
        self.pair_attr_stats = {}

    # Fixed to load data from a CSV file at the moment.
    def load_data(self, name, f_path, f_name, na_values=None):
        """
        load_data takes a CSV file of the initial data, adds tuple IDs (_tid_)
        to each row to uniquely identify an 'entity', and generates unique
        index numbers for each attribute/column.

        Creates a table with the user supplied 'name' parameter (e.g. 'hospital').
        """

        tic = time.clock()
        try:
            # Load raw CSV file/data into the Postgres 'init_X' table.

            self.raw_data = Table(name, Source.FILE, f_path, f_name, na_values)
            # Add _tid_ column to dataset
            df = self.raw_data.df
            df.insert(0,'_tid_', range(0,len(df)))
            df.fillna('_nan_',inplace=True)
            # Call to store to database
            self.raw_data.store_to_db(self.engine.engine)
            status = 'DONE Loading '+f_name

            # Generate indexes on attribute columns for faster queries

            for attr in self.raw_data.get_attributes():
                # Generate index on attribute
                self.raw_data.create_db_index(self.engine,[attr])

            # Create attr_to_idx dictionary (assign unique index for each attribute)
            # and attr_number (total # of attributes)

            tmp_attr_list = self.raw_data.get_attributes()
            tmp_attr_list.remove('_tid_')
            for idx, attr in enumerate(tmp_attr_list):
                # Map attribute to index
                self.attr_to_idx[attr] = idx
            self.attr_number = len(self.raw_data.get_attributes())

        except Exception as e:
            status = ' '.join(['For table:', name, str(e)])
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

        :param aux_table: (str) name of auxiliary table (see AuxTables)
        :param df: (DataFrame) dataframe to memoize/store for this auxiliary table
        :param store: (bool) if true, creates/replaces Postgres table for this auxiliary table
        :param index_attrs: (list[str]) list of attributes to create indexes on. If store is true,
        also creates indexes on Postgres table.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name, Source.DF, df)
            if store:
                self.aux_table[aux_table].store_to_db(self.engine.engine)
            if index_attrs:
                self.aux_table[aux_table].create_df_index(index_attrs)
            if store and index_attrs:
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
        except Exception as e:
            raise Exception(' '.join(['For table:',aux_table.name,str(e)]))

    def generate_aux_table_sql(self, aux_table, query, index_attrs=False):
        try:
            self.aux_table[aux_table] = Table(aux_table.name, Source.SQL, query, self.engine)
            if index_attrs:
                self.aux_table[aux_table].create_df_index(index_attrs)
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
        except Exception as e:
            raise Exception(' '.join(['For table:',aux_table.name,str(e)]))

    def get_raw_data(self):
        """
        Is this guaranteed sorted by TID?
        """
        if self.raw_data:
            return self.raw_data.df
        else:
            raise Exception('ERROR No dataset loaded')

    def get_attributes(self):
        if self.raw_data:
            attrs = self.raw_data.get_attributes()
            attrs.remove('_tid_')
            return attrs
        else:
            raise Exception('ERROR No dataset loaded')

    def get_cell_id(self, tuple_id, attr_name):
        """
        get_cell_id returns cell ID: a unique ID for every cell.

        Cell ID: _tid_ * (# of attributes) + attr_idx
        """
        vid = tuple_id*self.attr_number + self.attr_to_idx[attr_name]

        return vid

    def get_statistics(self):
        if not self.stats_ready:
            self.collect_stats()
        stats = (self.total_tuples, self.single_attr_stats, self.pair_attr_stats)
        self.stats_ready = True
        return stats

    def collect_stats(self):
        """
        collect_stats memoizes:
          1. self.single_attr_stats ({ attribute -> Series (value -> count) })
            the frequency (# of entities) of a given attribute-value
          2. self.pair_attr_stats ({ attr1 -> { attr2 -> DataFrame } } where
            DataFrame contains 3 columns:
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
        Returns a Series indexed on possible values for 'attr' and contains the frequency.
        """
        tmp_df = self.get_raw_data()[[attr]].groupby([attr]).size()
        return tmp_df

    def get_stats_pair(self, cond_attr, trg_attr):
        """
        Returns a DataFrame containing 3 columns:
            <cond_attr>: all possible values for cond_attr ('val1')
            <trg_attr>: all values for trg_attr that appeared at least once with <val1> ('val2')
            <count>: frequency (# of entities) where cond_attr: val1 AND trg_attr: val2
        """
        tmp_df = self.get_raw_data()[[cond_attr,trg_attr]].groupby([cond_attr,trg_attr]).size().reset_index(name="count")
        return tmp_df

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
        query = "SELECT t1._tid_, t1.attribute, domain[inferred_assignment + 1] as rv_value " \
                "FROM " \
                "(SELECT _tid_, attribute, " \
                "_vid_, init_value, string_to_array(regexp_replace(domain, \'[{\"\"}]\', \'\', \'gi\'), \'|||\') as domain " \
                "FROM %s) as t1, %s as t2 " \
                "WHERE t1._vid_ = t2._vid_"%(AuxTables.cell_domain.name, AuxTables.inf_values_idx.name)
        try:
            self.generate_aux_table_sql(AuxTables.inf_values_dom, query, index_attrs=['_tid_'])
            self.aux_table[AuxTables.inf_values_dom].create_db_index(self.engine, ['attribute'])
            status = "DONE colleting the inferred values."
        except Exception as e:
            status = "ERROR when colleting the inferred values: %s"%str(e)
        toc = time.clock()
        total_time = toc - tic
        return status, total_time

    def get_repaired_dataset(self):
        tic = time.clock()
        try:
            init_records = self.raw_data.df.sort_values(['_tid_']).to_records(index=False)
            t = self.aux_table[AuxTables.inf_values_dom]
            repaired_vals = dictify(t.df.reset_index())
            for tid in repaired_vals:
                for attr in repaired_vals[tid]:
                    init_records[tid][attr] = repaired_vals[tid][attr]
            repaired_df = pd.DataFrame.from_records(init_records)
            name = self.raw_data.name+'_repaired'
            self.repaired_data = Table(name, Source.DF, repaired_df)
            self.repaired_data.store_to_db(self.engine.engine)
            status = "DONE generating repaired dataset"
        except Exception as e:
            status = "ERROR when generating repaired dataset: %s"
        toc = time.clock()
        total_time = toc - tic
        return status, total_time


