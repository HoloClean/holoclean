from ..estimator import Estimator
import math

class NaiveBayes(Estimator):
    """
    NaiveBayes is an estimator of posterior probabilities using the naive
    independence assumption where
        p(v_cur | v_init) = p(v_cur) * PI_i (v_cur | v_init_i)
    where v_init_i is the init value for corresponding to attribute i. This
    probability is normalized over all values passed into predict_pp.
    """
    def __init__(self, dataset, freq, cooccur_freq, correlations, corr_strength):
        self._freq = freq
        self._cooccur_freq = cooccur_freq
        self._correlations = correlations
        self._corr_strength = corr_strength

    def train(self):
        pass

    def predict_pp(self, row, attr, values):
        nb_score = []
        for val1 in values:
            val1_count = self._freq[attr][val1]
            log_prob = math.log(float(val1_count)/float(self.total))
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

        return [(val, math.exp(log_prob) / denom) for val, log_ in nb_score]

    def _get_corr_attributes(attr):
        if attr not in self._correlations:
            return []

        d_temp = self._correlations[attr]
        d_temp = d_temp.abs()
        cor_attrs = [rec[0] for rec in d_temp[d_temp > self._corr_strength].iteritems() if rec[0] != attr]
        return cor_attrs
