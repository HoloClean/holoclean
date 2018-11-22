import unittest

import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import CurrentFeaturizer
from repair.featurize import CurrentAttrFeaturizer
from repair.featurize import CurrentSimFeaturizer
from repair.featurize import FreqFeaturizer
from repair.featurize import OccurFeaturizer
from repair.featurize import ConstraintFeat
from repair.featurize import LangModelFeat


class TestHolocleanRepair(unittest.TestCase):
    def test_hospital(self):
        # 1. Setup a HoloClean session.
        hc = holoclean.HoloClean(pruning_topk=0.1, epochs=10, weight_decay=0.01, threads=20, batch_size=1, verbose=True, timeout=3*60000).session

        # 2. Load training data and denial constraints.
        hc.load_data('hospital', '../testdata/hospital.csv')
        hc.load_dcs('../testdata/hospital_constraints_att.txt')
        hc.ds.set_constraints(hc.get_dcs())

        # 3. Detect erroneous cells using these two detectors.
        detectors = [NullDetector(), ViolationDetector()]
        hc.detect_errors(detectors)

        # 4. Repair errors utilizing the defined features.
        hc.setup_domain()
        featurizers = [CurrentAttrFeaturizer(), CurrentSimFeaturizer(), FreqFeaturizer(), OccurFeaturizer(), LangModelFeat(), ConstraintFeat()]

        # 5. Repair and evaluate the correctness of the results.
        eval_func = lambda: hc.evaluate('../testdata/hospital_clean.csv', 'tid', 'attribute', 'correct_val')
        hc.repair_errors(featurizers, em_iterations=2, em_iter_func=eval_func)


if __name__ == '__main__':
    unitttest.main()
