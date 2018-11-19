#python test_holoclean.py dataset_name data_diretory dirty_set clean_set constraints_directory constraints
import holoclean
import os
import time
import argparse
from detect import NullDetector, ViolationDetector
from repair.featurize import InitFeaturizer
from repair.featurize import InitAttFeaturizer
from repair.featurize import InitSimFeaturizer
from repair.featurize import FreqFeaturizer
from repair.featurize import OccurFeaturizer
from repair.featurize import ConstraintFeat
from repair.featurize import LangModelFeat
from repair.featurize import OccurAttrFeaturizer

parser = argparse.ArgumentParser(description='configurations')
parser.add_argument('-dataname', type = str, help ='name of the data, no appendix', required = True)
parser.add_argument('-dcpath', type = str, help ='dc path', required = True)
parser.add_argument('-datapath', type = str, help ='data path', default="/fastdisk/Datasets")
parser.add_argument('-dc', type = str, help ='dc file', required = True)
parser.add_argument('-data', type = str, help ='input data', default="")
parser.add_argument('-clean', type = str, help ='clean data', default="")
parser.add_argument('-outpath', type = str, help ='path for output', default="/fastdisk/Evaluation_Results/")
parser.add_argument('-k', type = float, help ='pruning_topk', default=0.1)
parser.add_argument('-w', type = float, help ='weight decay', default=0.01)
parser.add_argument('--normalize', help = 'if set true, normalize', action = 'store_true')
parser.add_argument('--wlog', help = 'if set true, log weight of featurizers', action = 'store_true')
parser.add_argument('--bias', help = 'if set true, bias set to true', action = 'store_true')
parser.add_argument('-msg', type = str, help ='msg appended to result file name', default='')
parser.add_argument('-notes', type = str, help ='notes appended to result', default='')
parser.add_argument('-omit', nargs='+', help='omit featurizer')

args = parser.parse_args()

if args.data == "":
    args.data = "{}.csv".format(args.dataname)
if args.clean == "":
    args.clean = "{}_clean.csv".format(args.dataname)


def get_tid(row):
    return row['tid']


# if clean data starts with tuple 1
def get_tid1(row):
    return row['tid'] - 1


def get_tupleid1(row):
    return row['tupleid'] - 1


def get_tupleid(row):
    return row['tupleid']


def get_attr_census(row):
    return row['attr_name'].lower()


def get_value_census(row):
    return row['attr_val'].strip()


def get_attr_adult(row):
    return row['attr_id'].lower()


def get_value_adult(row):
    return row['attr_val']


def get_attr_hospital(row):
    return row['attribute'].lower()


def get_value_hospital(row):
    return row['correct_val'].lower()


def get_value_food(row):
    return row['correct_value'].lower()


def get_value_physician(row):
    return row['correct_value']


# select featurizer used for this experiment
all_featurizer = {'init': InitFeaturizer(),
                  'freq': FreqFeaturizer(),
                  'initattr': InitAttFeaturizer(),
                  'constraint':ConstraintFeat(),
                  'occur': OccurFeaturizer(),
                  'lang':LangModelFeat(),
                  'initsim':InitSimFeaturizer(),
                  'occurattr': OccurAttrFeaturizer()}
if args.omit is not None:
    for o in args.omit:
        all_featurizer.pop(o)

# record runtime
start = time.time()

hc = holoclean.HoloClean(pruning_topk=args.k, epochs=30, momentum=0.0, l=0.01, weight_decay=args.w,
                         threads=50, batch_size=1, timeout=3*60000,
                         db_name = "{}_{}".format(args.dataname, args.notes),
                         normalize=args.normalize, verbose=True, bias=args.bias).session
if "adult" in args.dataname.lower():
    hc.load_data(args.dataname, args.datapath, args.data, na_values='?')
if "census" in args.dataname.lower():
    hc.load_data(args.dataname, args.datapath, args.data, na_values='empty')
else:
    hc.load_data(args.dataname, args.datapath, args.data)
hc.load_dcs(args.dcpath, args.dc)
hc.ds.set_constraints(hc.get_dcs())
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)
hc.setup_domain()
featurizers = all_featurizer.values()
featurizer_weights = hc.repair_errors(featurizers)
if "hospital" in args.dataname.lower():
    report = hc.evaluate(args.datapath, args.clean, get_tid1, get_attr_hospital, get_value_hospital)
elif "census" in args.dataname.lower():
    report = hc.evaluate(args.datapath, args.clean, get_tid, get_attr_census, get_value_census)
elif "adult" in args.dataname.lower():
    report = hc.evaluate(args.datapath, args.clean, get_tid, get_attr_adult, get_value_adult, na_values='?')
elif "food" in args.dataname.lower():
    report = hc.evaluate(args.datapath, args.clean, get_tupleid1, get_attr_hospital, get_value_food)
elif "physician" in args.dataname.lower():
    report = hc.evaluate(args.datapath, args.clean, get_tupleid, get_attr_hospital, get_value_physician)
else:
    raise Exception("customized function for data is not defined")

# record runtime
runtime = time.time() - start

# write result to csv file under data directory
result_path = os.path.join(args.outpath,"hc_eval_{}_{}.csv".format(args.dataname,args.msg))
exists = os.path.isfile(result_path)
result = open(result_path,"a+")

if not exists:
    result.write("data,dc,prunning_k,weight_decay,normalize,bias,featurizer," +
                 "notes,precision,recall,repairing_recall,F1,repairing_F1," +
                 "detected_errors,total_errors,correct_repairs," +
                 "total_repairs,total_repairs(Grdth_present),runtime,location\n")
report_str = ["%.4f"%i for i in report]
report_str = ",".join(report_str)
result.write("{data},{dc},{k},{w},{normalize},{bias},{f},{notes},{stat},{runtime},{location}\n".format(
    data=args.dataname,dc=args.dc,k=args.k,location=args.dcpath,notes=args.notes,
    stat=report_str,runtime=runtime,normalize=str(args.normalize),w=args.w,
    f="-".join( list( all_featurizer.keys() ) ), bias=str(args.bias) ) )
result.close()

if args.wlog:
    weight_log = open("{}_{}_omit{}_weights.csv".format(args.dataname, args.notes, args.omit),'w+')
    omit_str = "|".join(args.omit)
    weight_log.write("%s," % omit_str + featurizer_weights)


