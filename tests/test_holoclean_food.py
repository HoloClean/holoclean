import holoclean

from holoclean.detect import NullDetector, ViolationDetector
from holoclean.repair.featurize import InitFeaturizer
from holoclean.repair.featurize import InitAttFeaturizer
from holoclean.repair.featurize import FreqFeaturizer
from holoclean.repair.featurize import InitSimFeaturizer
from holoclean.repair.featurize import OccurFeaturizer
from holoclean.repair.featurize import ConstraintFeat
from holoclean.repair.featurize import LangModelFeat

def get_tid(row):
    return row['tupleid'] - 1

def get_attr(row):
    return row['attribute'].lower()

def get_value(row):
    return row['correct_value'].lower()

hc = holoclean.HoloClean(pruning_topk=0.3, epochs=30, weight_decay=0.01, threads=20, batch_size=1, verbose=True, timeout=3*60000).session
hc.load_data('food','data','food.csv')
hc.load_dcs('data','food_constraints.txt')
hc.ds.set_constraints(hc.get_dcs())
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)
hc.setup_domain()
featurizers = [InitAttFeaturizer(), InitSimFeaturizer(), FreqFeaturizer(), OccurFeaturizer(), LangModelFeat(), ConstraintFeat()]
hc.repair_errors(featurizers)
hc.evaluate('data','food_clean.csv', get_tid, get_attr, get_value)
