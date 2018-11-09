import time
import os
import pandas as pd
from string import Template

from ..dataset import AuxTables
from ..dataset.table import Table, Source

errors_template = Template('SELECT count(*) '\
                            'FROM $init_table as t1, $grdt_table as t2 '\
                            'WHERE t1._tid_ = t2._tid_ '\
                              'AND t2._attribute_ = \'$attr\' '\
                              'AND t1.$attr != t2._value_')

correct_repairs_template = Template('SELECT COUNT(*) FROM'\
                            '(SELECT t2._tid_, t2._attribute_, t2._value_ '\
                             'FROM $init_table as t1, $grdt_table as t2 '\
                             'WHERE t1._tid_ = t2._tid_ '\
                               'AND t2._attribute_ = \'$attr\' '\
                               'AND t1.$attr != t2._value_ ) as errors, $inf_dom as repairs '\
                              'WHERE errors._tid_ = repairs._tid_ '\
                                'AND errors._attribute_ = repairs.attribute '\
                                'AND errors._value_ = repairs.rv_value')


class EvalEngine:
    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset

    def load_data(self, name, f_path, f_name, get_tid, get_attr, get_val, na_values=None):
        tic = time.clock()
        try:
            raw_data = pd.read_csv(os.path.join(f_path,f_name), na_values=na_values)
            raw_data.fillna('_nan_',inplace=True)
            raw_data['_tid_'] = raw_data.apply(get_tid, axis=1)
            raw_data['_attribute_'] = raw_data.apply(get_attr, axis=1)
            raw_data['_value_'] = raw_data.apply(get_val, axis=1)
            raw_data = raw_data[['_tid_', '_attribute_', '_value_']]
            # Normalize string to lower-case and strip whitespaces.
            raw_data['_attribute_'] = raw_data['_attribute_'].apply(lambda x: x.lower())
            raw_data['_value_'] = raw_data['_value_'].apply(lambda x: x.strip())
            self.clean_data = Table(name, Source.DF, raw_data)
            self.clean_data.store_to_db(self.ds.engine.engine)
            self.clean_data.create_db_index(self.ds.engine, ['_tid_'])
            self.clean_data.create_db_index(self.ds.engine, ['_attribute_'])
            status = 'DONE Loading '+f_name
        except Exception as e:
            status = ' '.join(['For table:', name, str(e)])
        toc = time.clock()
        load_time = toc - tic
        return status, load_time

    def evaluate_repairs(self):
        self.compute_total_repairs()
        self.compute_total_repairs_grdt()
        self.compute_total_errors()
        self.compute_detected_errors()
        self.compute_correct_repairs()
        prec = self.compute_precision()
        rec = self.compute_recall()
        rep_recall = self.compute_repairing_recall()
        f1 = self.compute_f1()
        rep_f1 = self.compute_repairing_f1()
        return prec, rec, rep_recall, f1, rep_f1

    def eval_report(self):
        tic = time.clock()
        try:
            prec, rec, rep_recall, f1, rep_f1 = self.evaluate_repairs()
            report = "Precision = %.2f, Recall = %.2f, Repairing Recall = %.2f, F1 = %.2f, Repairing F1 = %.2f, Detected Errors = %d, Total Errors = %d, Correct Repairs = %d, Total Repairs = %d, Total Repairs (Grdth present) = %d" % (
                      prec, rec, rep_recall, f1, rep_f1, self.detected_errors, self.total_errors, self.correct_repairs, self.total_repairs, self.total_repairs_grdt)
        except Exception as e:
            report = "ERROR generating evaluation report: %s"%str(e)
        toc = time.clock()
        report_time = toc - tic
        return report, report_time

    def compute_total_repairs(self):
        query = "SELECT count(*) FROM " \
                "(SELECT _vid_ " \
                 "FROM %s as t1, %s as t2 " \
                 "WHERE t1._tid_ = t2._tid_ " \
                   "AND t1.attribute = t2.attribute " \
                   "AND t1.init_value != t2.rv_value) AS t"\
                %(AuxTables.cell_domain.name, AuxTables.inf_values_dom.name)
        res = self.ds.engine.execute_query(query)
        self.total_repairs = float(res[0][0])

    def compute_total_repairs_grdt(self):
        query = "SELECT count(*) FROM " \
                "(SELECT _vid_ " \
                 "FROM %s as t1, %s as t2, %s as t3 " \
                 "WHERE t1._tid_ = t2._tid_ " \
                   "AND t1.attribute = t2.attribute " \
                   "AND t1.init_value != t2.rv_value " \
                   "AND t1._tid_ = t3._tid_ " \
                   "AND t1.attribute = t3._attribute_) AS t"\
                %(AuxTables.cell_domain.name, AuxTables.inf_values_dom.name, self.clean_data.name)
        res = self.ds.engine.execute_query(query)
        self.total_repairs_grdt = float(res[0][0])

    def compute_total_errors(self):
        queries = []
        total_errors = 0.0
        for attr in self.ds.get_attributes():
            query = errors_template.substitute(init_table=self.ds.raw_data.name, grdt_table=self.clean_data.name,
                        attr=attr)
            queries.append(query)
        results = self.ds.engine.execute_queries(queries)
        for res in results:
            total_errors += float(res[0][0])
        self.total_errors = total_errors

    def compute_total_errors_grdt(self):
        queries = []
        total_errors = 0.0
        for attr in self.ds.get_attributes():
            query = errors_template.substitute(init_table=self.ds.raw_data.name, grdt_table=self.clean_data.name,
                        attr=attr)
            queries.append(query)
        results = self.ds.engine.execute_queries(queries)
        for res in results:
            total_errors += float(res[0][0])
        self.total_errors = total_errors

    def compute_detected_errors(self):
        query = "SELECT count(*) FROM " \
                "(SELECT _vid_ " \
                "FROM %s as t1, %s as t2, %s as t3 " \
                "WHERE t1._tid_ = t2._tid_ AND t1._cid_ = t3._cid_ " \
                "AND t1.attribute = t2._attribute_ " \
                "AND t1.init_value != t2._value_) AS t" \
                % (AuxTables.cell_domain.name, self.clean_data.name, AuxTables.dk_cells.name)
        res = self.ds.engine.execute_query(query)
        self.detected_errors = float(res[0][0])

    def compute_correct_repairs(self):
        queries = []
        correct_repairs = 0.0
        for attr in self.ds.get_attributes():
            query = correct_repairs_template.substitute(init_table=self.ds.raw_data.name, grdt_table=self.clean_data.name,
                        attr=attr, inf_dom=AuxTables.inf_values_dom.name)
            queries.append(query)
        results = self.ds.engine.execute_queries(queries)
        for res in results:
            correct_repairs += float(res[0][0])
        self.correct_repairs = correct_repairs

    def compute_recall(self):
        return self.correct_repairs / self.total_errors

    def compute_repairing_recall(self):
        return self.correct_repairs / self.detected_errors

    def compute_precision(self):
        return self.correct_repairs / self.total_repairs_grdt

    def compute_f1(self):
        prec = self.compute_precision()
        rec = self.compute_recall()
        try:
            f1 = 2*(prec*rec)/(prec+rec)
        except ZeroDivisionError as e:
            f1 = -1.0
        return f1

    def compute_repairing_f1(self):
        prec = self.compute_precision()
        rec = self.compute_repairing_recall()
        try:
            f1 = 2*(prec*rec)/(prec+rec)
        except ZeroDivisionError as e:
            f1 = -1.0
        return f1
