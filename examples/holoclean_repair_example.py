import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *


# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    pruning_topk=0.0,
    weak_label_thresh=0.90,
    domain_prune_thresh=0,
    max_domain=100,
    cor_strength=0.0,
    epochs=20,
    weight_decay=0.1,
    threads=1,
    batch_size=32,
    verbose=True,
    timeout=3*60000,
    print_fw=True
).session

# 2. Load training data and denial constraints.
hc.load_data('hospital_100', '../testdata/hospital_100.csv')
hc.load_dcs('../testdata/hospital_constraints_att.txt')
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
    ConstraintFeat()
]

infer_labeled = False
hc.repair_errors(featurizers, infer_labeled=infer_labeled)

# 5. Evaluate the correctness of the results.
hc.evaluate(fpath='../testdata/hospital_100_clean.csv',
            tid_col='tid',
            attr_col='attribute',
            val_col='correct_val',
            infer_labeled=infer_labeled)
