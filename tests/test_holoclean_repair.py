from detect import NullDetector, ViolationDetector
import holoclean
from repair.featurize import *

from tests.testutils import random_database, delete_database

TOL = 1e-9

def template(featurizers, estimator_type):
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
            print_fw=True,
            estimator_type=estimator_type,
        ).session

        # 2. Load training data and denial constraints.
        hc.load_data('hospital', '../testdata/hospital/hospital.csv')
        hc.load_dcs('../testdata/hospital/hospital_constraints.txt')
        hc.ds.set_constraints(hc.get_dcs())

        # 3. Detect erroneous cells using these two detectors.
        detectors = [NullDetector(), ViolationDetector()]
        hc.detect_errors(detectors)

        # 4. Repair errors utilizing the defined features.
        hc.generate_domain()
        hc.run_estimator()
        hc.repair_errors(featurizers)

        # 5. Evaluate the correctness of the results.
        report = hc.evaluate(fpath='../testdata/hospital/hospital_clean.csv',
                    tid_col='tid',
                    attr_col='attribute',
                    val_col='correct_val')

        return report

    finally:
        delete_database(db_name)

# We assert that our key metrics are exactly as tested for hospital.
# If these assertions ever fail in a new change, the results should
# be comparable if not better than before, unless a clear and correct
# reason can be given.

def test_hospital_with_init():
    report = template([InitAttrFeaturizer(),
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer()], 'NaiveBayes')

    assert report.correct_repairs == 232
    assert report.total_repairs == 232
    assert abs(report.precision - 1.) < TOL
    assert abs(report.recall - 232. / 509) < TOL
    assert abs(report.repair_recall - 232. / 435) < TOL
    assert report.total_repairs_grdt_correct == 0

def test_hospital_without_init():
    report = template([OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer()], 'NaiveBayes')

    assert report.correct_repairs == 434
    assert report.total_repairs == 456
    assert abs(report.precision - 434. / 456) < TOL
    assert abs(report.recall - 434. / 509) < TOL
    assert abs(report.repair_recall - 434. / 435) < TOL
    assert report.total_repairs_grdt_correct == 22

def test_hospital_logistic():
    report = template([OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer()], 'Logistic')

    assert report.correct_repairs == 434
    assert report.total_repairs == 456
    assert abs(report.precision - 434. / 456) < TOL
    assert abs(report.recall - 434. / 509) < TOL
    assert abs(report.repair_recall - 434. / 435) < TOL
    assert report.total_repairs_grdt_correct == 22

def test_hospital_embedding():
    report = template([OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer()], 'TupleEmbedding')

    assert report.correct_repairs == 434
    assert report.total_repairs == 456
    assert abs(report.precision - 434. / 456) < TOL
    assert abs(report.recall - 434. / 509) < TOL
    assert abs(report.repair_recall - 434. / 435) < TOL
    assert report.total_repairs_grdt_correct == 22
