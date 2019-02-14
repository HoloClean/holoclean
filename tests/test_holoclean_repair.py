import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *


def test_hospital():
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
        weight_decay=0.1,
        threads=1,
        batch_size=1,
        verbose=True,
        timeout=3 * 60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    ).session

    # 2. Load training data and denial constraints.
    hc.load_data('hospital_100', '../testdata/hospital_100.csv')
    hc.load_dcs('../testdata/hospital_constraints.txt')
    hc.ds.set_constraints(hc.get_dcs())

    # 3. Detect erroneous cells using these two detectors.
    detectors = [NullDetector(), ViolationDetector()]
    hc.detect_errors(detectors)

    # 4. Repair errors utilizing the defined features.
    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        InitSimFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        ConstraintFeaturizer(),
        LangModelFeaturizer(),
    ]

    hc.repair_errors(featurizers)

    # 5. Evaluate the correctness of the results.
    hc.evaluate(fpath='../testdata/hospital_100_clean.csv',
                tid_col='tid',
                attr_col='attribute',
                val_col='correct_val')

    # Just to make sure pipeline ran with no errors.
    assert True
