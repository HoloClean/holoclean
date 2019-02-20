from detect import NullDetector, ViolationDetector
import holoclean
from repair.featurize import *

from tests.testutils import random_database, delete_database

TOL = 1e-9


def test_hospital_with_init():
    db_name = random_database()

    try:
        # 1. Setup a HoloClean session.
        hc = holoclean.HoloClean(
            db_name=db_name,
            domain_thresh_1=0.0,
            domain_thresh_2=0.0,
            weak_label_thresh=0.99,
            max_domain=10000,
            cor_strength=0.6,
            nb_cor_strength=0.8,
            epochs=10,
            weight_decay=0.01,
            learning_rate=0.001,
            threads=1,
            batch_size=1,
            verbose=True,
            timeout=3 * 60000,
            feature_norm=False,
            weight_norm=False,
            print_fw=True
        ).session

        # 2. Load training data and denial constraints.
        hc.load_data('hospital', '../testdata/hospital.csv')
        hc.load_dcs('../testdata/hospital_constraints.txt')
        hc.ds.set_constraints(hc.get_dcs())

        # 3. Detect erroneous cells using these two detectors.
        detectors = [NullDetector(), ViolationDetector()]
        hc.detect_errors(detectors)

        # 4. Repair errors utilizing the defined features.
        hc.setup_domain()
        featurizers = [
            InitAttrFeaturizer(),
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer(),
        ]

        hc.repair_errors(featurizers)

        # 5. Evaluate the correctness of the results.
        report = hc.evaluate(fpath='../testdata/hospital_clean.csv',
                    tid_col='tid',
                    attr_col='attribute',
                    val_col='correct_val')

        # We assert that our key metrics are exactly as tested for hospital.
        # If these assertions ever fail in a new change, the results should
        # be comparable if not better than before, unless a clear and correct
        # reason can be given.
        assert report.correct_repairs == 232
        assert report.total_repairs == 232
        assert abs(report.precision - 1.) < TOL
        assert abs(report.recall - 232. / 509) < TOL
        assert abs(report.repair_recall - 232. / 435) < TOL
        assert report.total_repairs_grdt_correct == 0
    finally:
        delete_database(db_name)

def test_hospital_without_init():
    db_name = random_database()

    try:
        # 1. Setup a HoloClean session.
        hc = holoclean.HoloClean(
            db_name='holo',
            domain_thresh_1=0.0,
            domain_thresh_2=0.0,
            weak_label_thresh=0.99,
            max_domain=10000,
            cor_strength=0.6,
            nb_cor_strength=0.8,
            epochs=10,
            weight_decay=0.01,
            learning_rate=0.001,
            threads=1,
            batch_size=1,
            verbose=True,
            timeout=3 * 60000,
            feature_norm=False,
            weight_norm=False,
            print_fw=True
        ).session

        # 2. Load training data and denial constraints.
        hc.load_data('hospital', '../testdata/hospital.csv')
        hc.load_dcs('../testdata/hospital_constraints.txt')
        hc.ds.set_constraints(hc.get_dcs())

        # 3. Detect erroneous cells using these two detectors.
        detectors = [NullDetector(), ViolationDetector()]
        hc.detect_errors(detectors)

        # 4. Repair errors utilizing the defined features.
        hc.setup_domain()
        featurizers = [
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer(),
        ]

        hc.repair_errors(featurizers)

        # 5. Evaluate the correctness of the results.
        report = hc.evaluate(fpath='../testdata/hospital_clean.csv',
                    tid_col='tid',
                    attr_col='attribute',
                    val_col='correct_val')

        # We assert that our key metrics are exactly as tested for hospital.
        # If these assertions ever fail in a new change, the results should
        # be comparable if not better than before, unless a clear and correct
        # reason can be given.
        assert report.correct_repairs == 434
        assert report.total_repairs == 456
        assert abs(report.precision - 434. / 456) < TOL
        assert abs(report.recall - 434. / 509) < TOL
        assert abs(report.repair_recall - 434. / 435) < TOL
        assert report.total_repairs_grdt_correct == 22
    finally:
        delete_database(db_name)
