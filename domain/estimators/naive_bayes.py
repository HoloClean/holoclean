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
    def __init__(self, env, dataset, domain_df, correlations):
        Estimator.__init__(self, env, dataset)

        self._n_tuples, self._freq, self._cooccur_freq = self.ds.get_statistics()
        self.domain_df = domain_df
        self._correlations = correlations
        self._cor_strength = self.env['nb_cor_strength']
        self._corr_attrs = {}

        # TID to raw data tuple for prediction.
        self._raw_records_by_tid = {}
        for row in self.ds.get_raw_data().to_records():
            self._raw_records_by_tid[row['_tid_']] = row

    def train(self):
        pass

    def predict_pp(self, row, attr, values):
        nb_score = []
        correlated_attributes = self._get_corr_attributes(attr)
        for val1 in values:
            val1_count = self._freq[attr][val1]
            log_prob = math.log(float(val1_count) / float(self._n_tuples))
            for at in correlated_attributes:
                # Ignore same attribute, index, and tuple id.
                if at == attr or at == '_tid_':
                    continue
                val2 = row[at]
                val2_val1_count = 0.1
                if val1 in self._cooccur_freq[attr][at]:
                    if val2 in self._cooccur_freq[attr][at][val1]:
                        val2_val1_count = max(self._cooccur_freq[attr][at][val1][val2] - 1.0, 0.1)
                p = float(val2_val1_count) / float(val1_count)
                log_prob += math.log(p)
            nb_score.append((val1, log_prob))

        denom = sum(map(math.exp, [log_prob for _, log_prob in nb_score]))

        for val, log_prob in nb_score:
            yield (val, math.exp(log_prob) / denom)

    def predict_pp_batch(self):
        """
        Performs batch prediction.

        This technically invokes predict_pp underneath.

        Returns a List[List[Tuple]] where each List[Tuple] corresponds to
        a cell (ordered by the order a cell appears in `self.domain_df`
        during construction) and each Tuple is (val, proba) where
        val is the domain value and proba is the estimator's posterior probability estimate.
        """
        for row in tqdm(self.domain_df.to_records()):
            yield self.predict_pp(self._raw_records_by_tid[row['_tid_']], row['attribute'], row['domain'].split('|||'))

    def _get_corr_attributes(self, attr):
        """
        TODO: refactor this with Domain::get_corr_attributes().
        """
        if (attr, self._cor_strength) not in self._corr_attrs:
            self._corr_attrs[(attr, self._cor_strength)] = []

            if attr in self._correlations:
                attr_correlations = self._correlations[attr]
                self._corr_attrs[(attr, self._cor_strength)] = [corr_attr
                                                                for corr_attr, corr_strength in attr_correlations.items()
                                                                if corr_attr != attr and corr_strength > self._cor_strength]

        return self._corr_attrs[(attr, self._cor_strength)]
