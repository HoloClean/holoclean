import holoclean
from holoclean.detect import NullDetector, ViolationDetector
from holoclean.repair.featurize import InitAttFeaturizer
from holoclean.repair.featurize import InitSimFeaturizer
from holoclean.repair.featurize import FreqFeaturizer
from holoclean.repair.featurize import OccurFeaturizer
from holoclean.repair.featurize import ConstraintFeat
from holoclean.repair.featurize import LangModelFeat

def get_tid(row):
    return row['tid']

def get_attr(row):
    return row['attr_id'].lower()

def get_value(row):
    return row['attr_val']

hc = holoclean.HoloClean(pruning_topk=0.9, epochs=10, weight_decay=0.1, threads=20, batch_size=1, normalize=False, verbose=True, timeout=3*60000).session
hc.load_data('adult','/Users/thodoris/Documents/Research/SrcCode/Profiler/ProfilerData/adult_1','adult_1.csv', na_values='?')
hc.load_dcs('/Users/thodoris/Documents/Research/SrcCode/Profiler/ProfilerData/adult_1/attention/joint/decomposition/knn/multiple_topdown','1025150539_adult_1_preFalse_joint_osc0dot02_topk_6_s400_k5_b001_multiple_knn10_euclidean_dc.txt')
hc.ds.set_constraints(hc.get_dcs())
detectors = [NullDetector(),ViolationDetector()]
hc.detect_errors(detectors)
hc.setup_domain()
featurizers = [InitAttFeaturizer(), InitSimFeaturizer(), FreqFeaturizer(), LangModelFeat(), OccurFeaturizer(), ConstraintFeat()]
hc.repair_errors(featurizers)
hc.evaluate('/Users/thodoris/Documents/Research/SrcCode/Profiler/ProfilerData/adult_1','adult_1_clean.csv', get_tid, get_attr, get_value,na_values='?')
