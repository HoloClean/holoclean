import holoclean

from holoclean.detect import NullDetector, ViolationDetector
from holoclean.repair.featurize import InitFeaturizer
from holoclean.repair.featurize import OccurFeaturizer
from holoclean.repair.featurize import ConstraintFeat

def get_tid(row):
    return row['tupleid'] - 1

def get_attr(row):
    return row['attribute'].lower()

def get_value(row):
    return row['correct_value'].lower()

hc = holoclean.HoloClean(pruning_topk=0.3, epochs=30, momentum=0.0, l=0.001, weight_decay=0.9, threads=20, batch_size=1, normalize=True, verbose=True, bias=True, timeout=3 * 60000).session
hc.load_data('hospital', 'data', 'hospital.csv')
hc.load_dcs('data', 'hospital_constraints_att.txt')
hc.ds.set_constraints(hc.get_dcs())
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)
hc.setup_domain()
featurizers = [InitFeaturizer(), OccurFeaturizer(), ConstraintFeat()]
hc.repair_errors(featurizers)
hc.evaluate('data', 'hospital_clean.csv', get_tid, get_attr, get_value)
