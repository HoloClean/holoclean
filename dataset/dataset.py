import time
from enum import Enum
import pandas as pd
from .dbengine import DBengine
from .table import Table, Source


def dictify(frame):
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
        tic = time.clock()
        try:
            self.raw_data = Table(name, Source.FILE, f_path, f_name, na_values)
            # Add _tid_ column to dataset
            df = self.raw_data.df
            df.insert(0,'_tid_', range(0,len(df)))
            df.fillna('_nan_',inplace=True)
            self.raw_data.store_to_db(self.engine.engine)
            status = 'DONE Loading '+f_name
            for attr in self.raw_data.get_attributes():
                # Generate index on attribute
                self.raw_data.create_db_index(self.engine,[attr])
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
        vid = tuple_id*self.attr_number + self.attr_to_idx[attr_name]

        return vid

    def get_statistics(self):
        if not self.stats_ready:
            self.collect_stats()
        stats = (self.total_tuples, self.single_attr_stats, self.pair_attr_stats)
        self.stats_ready = True
        return stats

    def collect_stats(self):
        self.total_tuples = self.get_raw_data().shape[0]
        for attr in self.get_attributes():
            self.single_attr_stats[attr] = self.get_stats_single(attr)
        for cond_attr in self.get_attributes():
            self.pair_attr_stats[cond_attr] = {}
            for trg_attr in self.get_attributes():
                if trg_attr != cond_attr:
                    self.pair_attr_stats[cond_attr][trg_attr] = self.get_stats_pair(cond_attr,trg_attr)

    def get_stats_single(self, attr):
        tmp_df = self.get_raw_data()[[attr]].groupby([attr]).size()
        return tmp_df

    def get_stats_pair(self, cond_attr, trg_attr):
        tmp_df = self.get_raw_data()[[cond_attr,trg_attr]].groupby([cond_attr,trg_attr]).size().reset_index(name="count")
        return tmp_df

    def get_domain_info(self):
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


