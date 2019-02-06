import math

from tqdm import tqdm

from ..estimator import Estimator


class NaiveBayes(Estimator):
    """
    NaiveBayes is an estimator of posterior probabilities using the naive
    independence assumption where
        p(v_cur | v_init) = p(v_cur) * \prod_i (v_init_i | v_cur)
    where v_init_i is the init value for corresponding to attribute i. This
    probability is normalized over all values passed into predict_pp.
    """
    def __init__(self, dataset, correlations, cor_strength):
        self._n_tuples, self._freq, self._cooccur_freq = dataset.get_statistics()
        self._correlations = correlations
        self._cor_strength = cor_strength
        self._corr_attrs = {}

    def train(self):
        pass

    def predict_pp(self, row, attr, values):
        nb_score = []
        for val1 in values:
            val1_count = self._freq[attr][val1]
            log_prob = math.log(float(val1_count)/float(self._n_tuples))
            correlated_attributes = self._get_corr_attributes(attr)
            total_log_prob = 0.0
            for at in correlated_attributes:
                if at != attr:
                    val2 = row[at]
                    val2_count = self._freq[at][val2]
                    val2_val1_count = 0.1
                    if val1 in self._cooccur_freq[attr][at]:
                        if val2 in self._cooccur_freq[attr][at][val1]:
                            val2_val1_count = max(self._cooccur_freq[attr][at][val1][val2] - 1.0, 0.1)
                    p = float(val2_val1_count)/float(val1_count)
                    log_prob += math.log(p)
            nb_score.append((val1, log_prob))

        denom = sum(map(math.exp, [log_prob for _, log_prob in nb_score]))

        for val, log_prob in nb_score:
            yield (val, math.exp(log_prob) / denom)

    def predict_pp_batch(self, raw_records_by_tid, cell_domain_rows):
        """
        Performs batch prediction.

        This technically invokes predict_pp underneath.

        :param raw_records_by_tid: (dict) maps TID to its corresponding row (record) in the raw data
        :param cell_domain_rows: (list[pd.record]) list of records from the cell domain DF
        """
        for row in tqdm(cell_domain_rows):
            yield self.predict_pp(raw_records_by_tid[row['_tid_']], row['attribute'], row['domain'].split('|||'))

    def _get_corr_attributes(self, attr):
        if (attr, self._cor_strength) not in self._corr_attrs:
            self._corr_attrs[(attr,self._cor_strength)] = []

            if attr in self._correlations:
                d_temp = self._correlations[attr]
                d_temp = d_temp.abs()
                self._corr_attrs[(attr,self._cor_strength)] = [rec[0] for rec in d_temp[d_temp > self._cor_strength].iteritems() if rec[0] != attr]

        return self._corr_attrs[(attr, self._cor_strength)]
