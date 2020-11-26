from collections import namedtuple
import logging
import os
from string import Template
import time

import pandas as pd

from ..dataset import AuxTables
from ..dataset.table import Table, Source
from ..utils import NULL_REPR

report_name_list = ['precision', 'recall', 'repair_recall',
                    'f1', 'repair_f1', 'detected_errors', 'total_errors', 'correct_repairs', 'total_repairs',
                    'total_repairs_grdt', 'total_repairs_grdt_correct', 'total_repairs_grdt_incorrect',
                    'rmse']

EvalReport = namedtuple('EvalReport', report_name_list)
EvalReport.__new__.__defaults__ = (0,) * len(report_name_list)

errors_template = Template('SELECT count(*) ' \
                           'FROM  "$init_table" as t1, "$grdt_table" as t2 ' \
                           'WHERE t1._tid_ = t2._tid_ ' \
                           '  AND t2._attribute_ = \'$attr\' ' \
                           '  AND t1."$attr" != t2._value_')

"""
The 'errors' aliased subquery returns the (_tid_, _attribute_, _value_)
from the ground truth table for all cells that have an error in the original
raw data.

The 'repairs' aliased table contains the cells and values we've inferred.

We then count the number of cells that we repaired to the correct ground
truth value.
"""
correct_repairs_template = Template('SELECT COUNT(*) FROM '
                                    '  (SELECT t2._tid_, t2._attribute_, t2._value_ '
                                    '     FROM "$init_table" as t1, "$grdt_table" as t2 '
                                    '    WHERE t1._tid_ = t2._tid_ '
                                    '      AND t2._attribute_ = \'$attr\' '
                                    '      AND t1."$attr" != t2._value_) as errors, "$inf_dom" as repairs '
                                    'WHERE errors._tid_ = repairs._tid_ '
                                    '  AND errors._attribute_ = repairs.attribute '
                                    '  AND repairs.rv_value = errors._value_')


"""
Calculates RMSE error for the given attribute.
"""
rmse_template = Template('SELECT SQRT(AVG(POWER(grdt._value_::NUMERIC -repairs.rv_value::NUMERIC, 2))) FROM '
                        '"$grdt_table" as grdt, "$inf_dom" as repairs '
                        'WHERE grdt._tid_ = repairs._tid_ '
                        '  AND grdt._attribute_ = repairs.attribute'
                        '  AND grdt._attribute_ IN $attrs_list')


class EvalEngine:
    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset

    def load_data(self, name, fpath, tid_col, attr_col, val_col, na_values=None):
        tic = time.clock()
        try:
            raw_data = pd.read_csv(fpath, na_values=na_values, dtype=str, encoding='utf-8')
            # We drop any ground truth values that are NULLs since we follow
            # the closed-world assumption (if it's not there it's wrong).
            # TODO: revisit this once we allow users to specify which
            # attributes may be NULL.
            raw_data.dropna(subset=[val_col], inplace=True)
            raw_data.rename({tid_col: '_tid_',
                             attr_col: '_attribute_',
                             val_col: '_value_'},
                            axis='columns',
                            inplace=True)
            raw_data = raw_data[['_tid_', '_attribute_', '_value_']]
            raw_data['_tid_'] = raw_data['_tid_'].astype(int)

            # Normalize string to whitespaces.
            categorical_attrs = self.ds.categorical_attrs
            if categorical_attrs:
                cat_cells = raw_data['_attribute_'].isin(categorical_attrs)
                raw_data.loc[cat_cells, '_value_'] = \
                    raw_data.loc[cat_cells, '_value_'].astype(str).str.strip().str.lower()

            self.clean_data = Table(name, Source.DF, df=raw_data)
            self.clean_data.store_to_db(self.ds.engine.engine, schema=self.ds.engine.dbschema)
            self.clean_data.create_db_index(self.ds.engine, ['_tid_'])
            self.clean_data.create_db_index(self.ds.engine, ['_attribute_'])
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))
        except Exception:
            logging.error('load_data for table %s', name)
            raise
        toc = time.clock()
        load_time = toc - tic
        return status, load_time

    def eval_report(self, attr=None):
        """
        Returns an EvalReport named tuple containing the experiment results.
        :param attr: if attr is not None, compute results for attr:
                        if attr is numerical, then only report rmse
                        if attr is categorical, then report precision, recall etc.
                     if attr is None, compute results for all attrs
        """
        tic = time.clock()
        eval_report_dict = {}
        # attr is not None and is numerical
        # or attr is None(query on all attrs) and no categorical
        if attr is None or attr in self.ds.numerical_attrs:
            eval_report_dict['rmse'] = self.compute_rmse(attr) or 0.

        if attr is None or attr in self.ds.categorical_attrs:
            # if attr in self.ds.categorical_attrs or attr is None
            # attr is not None and is categorical
            # attr is None and should query on all
            correct_repairs = self.compute_correct_repairs(attr)
            total_repairs = self.compute_total_repairs(attr)
            detected_errors = self.compute_detected_errors(attr)
            total_errors = self.compute_total_errors(attr)
            total_repairs_grdt_correct, total_repairs_grdt_incorrect, \
                total_repairs_grdt = self.compute_total_repairs_grdt(attr)

            eval_report_dict['detected_errors'] = detected_errors
            eval_report_dict['total_errors'] = total_errors
            eval_report_dict['correct_repairs'] = correct_repairs
            eval_report_dict['total_repairs'] = total_repairs
            eval_report_dict['total_repairs_grdt'] = total_repairs_grdt
            eval_report_dict['total_repairs_grdt_correct'] = total_repairs_grdt_correct
            eval_report_dict['total_repairs_grdt_incorrect'] = total_repairs_grdt_incorrect

            eval_report_dict['precision'] = self.compute_precision(correct_repairs, total_repairs_grdt)
            eval_report_dict['recall'] = self.compute_recall(correct_repairs, total_errors)
            eval_report_dict['repair_recall'] = self.compute_repairing_recall(correct_repairs, detected_errors)
            eval_report_dict['f1'] = self.compute_f1(correct_repairs, total_errors, total_repairs_grdt)
            eval_report_dict['repair_f1'] = self.compute_repairing_f1(correct_repairs, detected_errors,
                                                                      total_repairs_grdt)

        report = EvalReport(**eval_report_dict)
        report_str = "Precision = %.2f, Recall = %.2f, Repairing Recall = %.2f, " \
                     "F1 = %.2f, Repairing F1 = %.2f, Detected Errors = %d, " \
                     "Total Errors = %d, Correct Repairs = %d, Total Repairs = %d, " \
                     "Total Repairs on correct cells (Grdth present) = %d, " \
                     "Total Repairs on incorrect cells (Grdth present) = %d, " \
                     "RMSE = %.2f" % (report.precision, report.recall, report.repair_recall, report.f1, report.repair_f1,
                                     report.detected_errors, report.total_errors, report.correct_repairs,
                                     report.total_repairs,
                                     report.total_repairs_grdt_correct, report.total_repairs_grdt_incorrect,
                                     report.rmse)

        if attr:
            report_str = "# Attribute:{};{}".format(attr, report_str)

        toc = time.clock()
        report_time = toc - tic
        return report_str, report_time, report

    """
    All these compute_xxx methods are aimed at categorical attributes
    All numerical attrs should not be involved in computing precision, recall, etc.
    """
    def get_categorical_clause(self, attr):
        """
        Get where condition on which attr should be queried
        :param attr: if attr is None, generate condition on all categorical attrs
        :return: str type attr_clause showing: 'AND t1.attribute = attr' to only do query on target attr
                if attr is None, then 'AND (t1.attribute = attr1 OR t1.attribute = attr2 ...) for all categorical attrs
        """
        query_attrs = [attr] if attr else self.ds.categorical_attrs
        query_attrs_str = ["\'{}\'".format(attr) for attr in query_attrs]
        categorical_where = 't1.attribute IN (%s)' % ','.join(query_attrs_str)

        return categorical_where

    def compute_total_repairs(self, attr=None):
        """
        compute_total_repairs memoizes the number of repairs:
        the # of cells that were inferred and where the inferred value
        is not equal to the initial value.
        :param attr: if attr is not None, it must be categorical
        """
        assert attr is None or attr in self.ds.categorical_attrs

        if not self.ds.categorical_attrs:
            return 0.
        # do not query on numerical attrs (when attr is None, indicating we want query on all categorical attrs)
        # if there are no numerical attrs, then no condition should be added, just query on all attributes
        attr_clause = self.get_categorical_clause(attr) if self.ds.numerical_attrs else "TRUE"
        query = "SELECT count(*) FROM " \
                "  (SELECT _vid_ " \
                '     FROM "{}" as t1, "{}" as t2 ' \
                "    WHERE t1._tid_ = t2._tid_ " \
                "      AND t1.attribute = t2.attribute " \
                "      AND t1.init_value != t2.rv_value " \
                "      AND {}) AS t".format(AuxTables.cell_domain.name,
                                            AuxTables.inf_values_dom.name,
                                            attr_clause)
        res = self.ds.engine.execute_query(query)
        return res[0][0]

    def compute_total_repairs_grdt(self, attr=None):
        """
        compute_total_repairs_grdt memoizes the number of repairs for cells
        that are specified in the clean/ground truth data. Otherwise repairs
        are defined the same as compute_total_repairs.

        We also distinguish between repairs on correct cells and repairs on
        incorrect cells (correct cells are cells where init == ground truth).
        :param attr: if attr is not None, it must be categorical
        """

        assert attr is None or attr in self.ds.categorical_attrs
        if not self.ds.categorical_attrs:
            return 0.
        # do not query on numerical attrs (when attr is None, indicating we want query on all categorical attrs)
        # if there are no numerical attrs, then no condition should be added, just query on all attributes
        attr_clause = self.get_categorical_clause(attr) if self.ds.numerical_attrs else "TRUE"

        query = """
            SELECT
                t1.init_value = t3._value_ AS is_correct,
                count(*)
            FROM   "{}" as t1, "{}" as t2, "{}" as t3
            WHERE  t1._tid_ = t2._tid_
              AND  t1.attribute = t2.attribute
              AND  t1.init_value != t2.rv_value
              AND  t1._tid_ = t3._tid_
              AND  t1.attribute = t3._attribute_
              AND  {}
            GROUP BY is_correct
              """.format(AuxTables.cell_domain.name,
                         AuxTables.inf_values_dom.name,
                         self.clean_data.name,
                         attr_clause)

        res = self.ds.engine.execute_query(query)

        # Memoize the number of repairs on correct cells and incorrect cells.
        # Since we do a GROUP BY we need to check which row of the result
        # corresponds to the correct/incorrect counts.
        total_repairs_grdt_correct, total_repairs_grdt_incorrect = 0, 0
        if not res:
            return 0, 0, 0

        if res[0][0]:
            correct_idx, incorrect_idx = 0, 1
        else:
            correct_idx, incorrect_idx = 1, 0
        if correct_idx < len(res):
            total_repairs_grdt_correct = float(res[correct_idx][1])
        if incorrect_idx < len(res):
            total_repairs_grdt_incorrect = float(res[incorrect_idx][1])
        total_repairs_grdt = total_repairs_grdt_correct + total_repairs_grdt_incorrect

        return total_repairs_grdt_correct, total_repairs_grdt_incorrect, total_repairs_grdt

    def compute_total_errors(self, attr=None):
        """
        compute_total_errors memoizes the number of cells that have a
        wrong initial value: requires ground truth data.
        :param attr: if attr is not None, it must be categorical
        """
        assert attr is None or attr in self.ds.categorical_attrs
        if not self.ds.categorical_attrs:
            return 0.

        queries = []
        total_errors = 0.0

        query_attrs = [attr] if attr else self.ds.categorical_attrs
        for attr in query_attrs:
            query = errors_template.substitute(init_table=self.ds.raw_data.name,
                                               grdt_table=self.clean_data.name,
                                               attr=attr)
            queries.append(query)
        results = self.ds.engine.execute_queries(queries)

        for i in range(len(results)):
            res = results[i]
            total_errors += float(res[0][0])

        return total_errors

    def compute_detected_errors(self, attr=None):
        """
        compute_detected_errors memoizes the number of error cells that
        were detected in error detection: requires ground truth.

        This value is always equal or less than total errors (see
        compute_total_errors).
        :param attr: if attr is not None, it must be categorical
        """
        assert attr is None or attr in self.ds.categorical_attrs
        if not self.ds.categorical_attrs:
            return 0.
        # do not query on numerical attrs (when attr is None, indicating we want query on all categorical attrs)
        # if there are no numerical attrs, then no condition should be added, just query on all attributes
        attr_clause = self.get_categorical_clause(attr) if self.ds.numerical_attrs else "TRUE"

        query = "SELECT count(*) FROM " \
                "  (SELECT _vid_ " \
                '   FROM "{}" as t1, "{}" as t2, "{}" as t3 ' \
                "   WHERE t1._tid_ = t2._tid_ " \
                "     AND t1._cid_ = t3._cid_ " \
                "     AND t1.attribute = t2._attribute_ " \
                "     AND t1.init_value != t2._value_" \
                "     AND {}) AS t".format(AuxTables.cell_domain.name,
                                           self.clean_data.name,
                                           AuxTables.dk_cells.name,
                                           attr_clause)
        res = self.ds.engine.execute_query(query)
        return float(res[0][0])

    def compute_correct_repairs(self, attr=None):
        """
        compute_correct_repairs memoizes the number of error cells
        that were correctly inferred.

        This value is always equal or less than total errors (see
        compute_total_errors).
        :param attr： if attr is not None, it must be categorical
        """
        assert attr is None or attr in self.ds.categorical_attrs
        if not self.ds.categorical_attrs:
            return 0.

        queries = []
        correct_repairs = 0.0

        query_attrs = [attr] if attr else self.ds.categorical_attrs
        for attr in query_attrs:
            query = correct_repairs_template.substitute(init_table=self.ds.raw_data.name,
                                                        grdt_table=self.clean_data.name,
                                                        attr=attr, inf_dom=AuxTables.inf_values_dom.name)
            queries.append(query)
        results = self.ds.engine.execute_queries(queries)

        for i in range(len(results)):
            res = results[i]
            correct_repairs += float(res[0][0])

        return correct_repairs

    def compute_rmse(self, attr=None):
        """
        Should check all the dk_cells in numerical attributes
        compute RMS error for all dk_cells in numerical attributes
        :return:
        """
        assert attr is None or attr in self.ds.numerical_attrs
        if not self.ds.numerical_attrs:
            return 0.

        query_attrs = [attr] if attr else self.ds.numerical_attrs
        query_attrs_str = ["\'{}\'".format(attr) for attr in query_attrs]
        query_attrs_sql = '(%s)' % ','.join(query_attrs_str)
        query = rmse_template.substitute(grdt_table=self.clean_data.name, inf_dom=AuxTables.inf_values_dom.name,
                                        attrs_list=query_attrs_sql)
        res = self.ds.engine.execute_query(query)

        return res[0][0]

    @staticmethod
    def compute_recall(correct_repairs, total_errors):
        """
        Computes the recall (# of correct repairs / # of total errors).
        """
        if total_errors == 0:
            return 0
        return correct_repairs / total_errors

    @staticmethod
    def compute_repairing_recall(correct_repairs, detected_errors):
        """
        Computes the _repairing_ recall (# of correct repairs / # of total
        _detected_ errors).
        """
        if detected_errors == 0:
            return 0
        return correct_repairs / detected_errors

    @staticmethod
    def compute_precision(correct_repairs, total_repairs_grdt):
        """
        Computes precision (# correct repairs / # of total repairs w/ ground truth)
        """
        if total_repairs_grdt == 0:
            return 0
        return correct_repairs / total_repairs_grdt

    @staticmethod
    def compute_f1(correct_repairs, total_errors, total_repairs_grdt):
        prec = EvalEngine.compute_precision(correct_repairs, total_repairs_grdt)
        rec = EvalEngine.compute_recall(correct_repairs, total_errors)
        if prec + rec == 0:
            f1 = 0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)
        return f1

    @staticmethod
    def compute_repairing_f1(correct_repairs, detected_errors, total_repairs_grdt):
        prec = EvalEngine.compute_precision(correct_repairs, total_repairs_grdt)
        rec = EvalEngine.compute_repairing_recall(correct_repairs, detected_errors)
        if prec == 0 and rec == 0:
            f1 = 0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)

        return f1

    def log_weak_label_stats(self):
        query = """
        select
            (t3._tid_ is NULL) as clean,
            (t1.fixed) as status,
            (t1.init_value =  t2._value_) as init_eq_grdth,
            (t1.weak_label = t2._value_) as wl_eq_grdth,
            (t1.weak_label = t4.rv_value) as wl_eq_infer,
            (t4.rv_value = t2._value_) as infer_eq_grdth,
            count(*) as count
        from
            "{cell_domain}" as t1,
            "{clean_data}" as t2
            left join "{dk_cells}" as t3 on t2._tid_ = t3._tid_ and t2._attribute_ = t3.attribute
            left join "{inf_values_dom}" as t4 on t2._tid_ = t4._tid_ and t2._attribute_ = t4.attribute where t1._tid_ = t2._tid_ and t1.attribute = t2._attribute_
        group by
            clean,
            status,
            init_eq_grdth,
            wl_eq_grdth,
            wl_eq_infer,
            infer_eq_grdth
        """.format(cell_domain=AuxTables.cell_domain.name,
                clean_data=self.clean_data.name,
                dk_cells=AuxTables.dk_cells.name,
                inf_values_dom=AuxTables.inf_values_dom.name)

        res = self.ds.engine.execute_query(query)

        df_stats = pd.DataFrame(res,
                columns=["is_clean", "cell_status",
                    "init = grdth", "wlabel = grdth", "wlabel = infer",
                    "infer = grdth", "count"])
        df_stats = df_stats.sort_values(list(df_stats.columns)).reset_index(drop=True)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', len(df_stats))
        pd.set_option('display.max_colwidth', -1)
        logging.debug("weak label statistics: (cell_status: 0 - none, 1 - wlabelled, 2 - single value)\n%s", df_stats)
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_colwidth')
