from string import Template

import pandas as pd

from .detector import Detector

unary_template = Template('SELECT t1._tid_ FROM "$table" as t1 WHERE $cond')
multi_template = Template('SELECT t1._tid_ FROM "$table" as t1 WHERE $cond1 $c EXISTS (SELECT t2._tid_ FROM "$table" as t2 WHERE $cond2)')


class ViolationDetector(Detector):
    """
    Detector to detect violations of integrity constraints (mainly denial constraints).
    """

    def __init__(self, name='ViolationDetector'):
        super(ViolationDetector, self).__init__(name)

    def setup(self, dataset, env):
        self.ds = dataset
        self.env = env
        self.constraints = dataset.constraints

    def detect_noisy_cells(self):
        """
        Returns a pandas.DataFrame containing all cells that
         violate denial constraints contained in self.dataset.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute violating any denial constraint.
        """
        # Convert  Constraints to SQL queries
        tbl = self.ds.raw_data.name
        queries = []
        attrs = []
        for c in self.constraints:
            q = self.to_sql(tbl, c)
            queries.append(q)
            attrs.append(c.components)
        # Execute Queries over the DBEngine of Dataset
        results = self.ds.engine.execute_queries(queries)

        # Generate final output
        errors = []
        for i in range(len(attrs)):
            res = results[i]
            attr_list = attrs[i]
            tmp_df = self.gen_tid_attr_output(res, attr_list)
            errors.append(tmp_df)
        errors_df = pd.concat(errors, ignore_index=True).drop_duplicates().reset_index(drop=True)
        return errors_df

    def to_sql(self, tbl, c):
        # Check tuples in constraint
        unary = len(c.tuple_names)==1
        if unary:
            query = self.gen_unary_query(tbl, c)
        else:
            query = self.gen_mult_query(tbl, c)
        return query

    def gen_unary_query(self, tbl, c):
        query = unary_template.substitute(table=tbl, cond=c.cnf_form)
        return query

    def gen_mult_query(self, tbl, c):
        # Iterate over constraint predicates to identify cond1 and cond2
        cond1_preds = []
        cond2_preds = []
        for pred in c.predicates:
            if 't1' in pred.cnf_form:
                if 't2' in pred.cnf_form:
                    cond2_preds.append(pred.cnf_form)
                else:
                    cond1_preds.append(pred.cnf_form)
            elif 't2' in pred.cnf_form:
                cond2_preds.append(pred.cnf_form)
            else:
                raise Exception("ERROR in violation detector. Cannot ground mult-tuple template.")
        cond1 = " AND ".join(cond1_preds)
        cond2 = " AND ".join(cond2_preds)
        a = ','.join(c.components)
        a = []
        for b in c.components:
            a.append("'"+b+"'")
        a = ','.join(a)
        if cond1 != '':
            query = multi_template.substitute(table=tbl, cond1=cond1, c='AND', cond2=cond2)
        else:
            query = multi_template.substitute(table=tbl, cond1=cond1, c='', cond2=cond2)
        return query

    def gen_tid_attr_output(self, res, attr_list):
        errors = []
        for tuple in res:
            tid = int(tuple[0])
            for attr in attr_list:
                errors.append({'_tid_': tid, 'attribute': attr})
        error_df  = pd.DataFrame(data=errors)
        return error_df
