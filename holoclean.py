import time
import logging
import os
import random

import torch
import numpy as np
import pandas as pd

from dataset import Dataset, Table, Source, AuxTables
from dcparser import Parser
from domain import DomainEngine
from detect import DetectEngine
from repair import RepairEngine
from evaluate import EvalEngine
from dataset.quantization import quantize_km
from utils import NULL_REPR


logging.basicConfig(format="%(asctime)s - [%(levelname)5s] - %(message)s", datefmt='%H:%M:%S')
root_logger = logging.getLogger()
gensim_logger = logging.getLogger('gensim')
root_logger.setLevel(logging.INFO)
gensim_logger.setLevel(logging.WARNING)


# Arguments for HoloClean
arguments = [
    (('-u', '--db_user'),
        {'metavar': 'DB_USER',
         'dest': 'db_user',
         'default': 'holocleanuser',
         'type': str,
         'help': 'User for DB used to persist state.'}),
    (('-p', '--db-pwd', '--pass'),
        {'metavar': 'DB_PWD',
         'dest': 'db_pwd',
         'default': 'abcd1234',
         'type': str,
         'help': 'Password for DB used to persist state.'}),
    (('-h', '--db-host'),
        {'metavar': 'DB_HOST',
         'dest': 'db_host',
         'default': 'localhost',
         'type': str,
         'help': 'Host for DB used to persist state.'}),
    (('-d', '--db_name'),
        {'metavar': 'DB_NAME',
         'dest': 'db_name',
         'default': 'holo',
         'type': str,
         'help': 'Name of DB used to persist state.'}),
    (('-t', '--threads'),
     {'metavar': 'THREADS',
      'dest': 'threads',
      'default': 20,
      'type': int,
      'help': 'How many threads to use for parallel execution. If <= 1, then no pool workers are used.'}),
    (('-dbt', '--timeout'),
     {'metavar': 'TIMEOUT',
      'dest': 'timeout',
      'default': 60000,
      'type': int,
      'help': 'Timeout for expensive featurization queries.'}),
    (('-s', '--seed'),
     {'metavar': 'SEED',
      'dest': 'seed',
      'default': 45,
      'type': int,
      'help': 'The seed to be used for torch.'}),
    (('-ls', '--layer_sizes'),
     {'metavar': 'LAYER_SIZES',
      'dest':  'layer_sizes',
      'default': [1],
      'type': list,
      'help': 'List of layer sizes of the final FC layers. Last layer must have output size of 1. For example for a hidden layer of size 200 one can specify [200,1].'}),
    (('-l', '--learning-rate'),
     {'metavar': 'LEARNING_RATE',
      'dest': 'learning_rate',
      'default': 0.001,
      'type': float,
      'help': 'The learning rate used during training.'}),
    (('-o', '--optimizer'),
     {'metavar': 'OPTIMIZER',
      'dest': 'optimizer',
      'default': 'adam',
      'type': str,
      'help': 'Optimizer used for learning.'}),
    (('-e', '--epochs'),
     {'metavar': 'LEARNING_EPOCHS',
      'dest': 'epochs',
      'default': 20,
      'type': float,
      'help': 'Number of epochs used for training.'}),
    (('-w', '--weight_decay'),
     {'metavar': 'WEIGHT_DECAY',
      'dest':  'weight_decay',
      'default': 0.01,
      'type': float,
      'help': 'Weight decay across iterations.'}),
    (('-m', '--momentum'),
     {'metavar': 'MOMENTUM',
      'dest': 'momentum',
      'default': 0.0,
      'type': float,
      'help': 'Momentum for SGD.'}),
    (('-b', '--batch-size'),
     {'metavar': 'BATCH_SIZE',
      'dest': 'batch_size',
      'default': 1,
      'type': int,
      'help': 'The batch size during training.'}),
    (('-wlt', '--weak-label-thresh'),
     {'metavar': 'WEAK_LABEL_THRESH',
      'dest': 'weak_label_thresh',
      'default': 0.90,
      'type': float,
      'help': 'Threshold of posterior probability to assign weak labels.'}),
    (('-dt1', '--domain_thresh_1'),
     {'metavar': 'DOMAIN_THRESH_1',
      'dest': 'domain_thresh_1',
      'default': 0.1,
      'type': float,
      'help': 'Minimum co-occurrence probability threshold required for domain values in the first domain pruning stage. Between 0 and 1.'}),
    (('-dt2', '--domain-thresh-2'),
     {'metavar': 'DOMAIN_THRESH_2',
      'dest': 'domain_thresh_2',
      'default': 0,
      'type': float,
      'help': 'Threshold of posterior probability required for values to be included in the final domain in the second domain pruning stage. Between 0 and 1.'}),
    (('-md', '--max-domain'),
     {'metavar': 'MAX_DOMAIN',
      'dest': 'max_domain',
      'default': 1000000,
      'type': int,
      'help': 'Maximum number of values to include in the domain for a given cell.'}),
    (('-cs', '--cor-strength'),
     {'metavar': 'COR_STRENGTH',
      'dest': 'cor_strength',
      'default': 0.05,
      'type': float,
      'help': 'Correlation threshold (absolute) when selecting correlated attributes for domain pruning.'}),
    (('-cs', '--nb-cor-strength'),
     {'metavar': 'NB_COR_STRENGTH',
      'dest': 'nb_cor_strength',
      'default': 0.3,
      'type': float,
      'help': 'Correlation threshold for correlated attributes when using NaiveBayes estimator.'}),
    (('-fn', '--feature-norm'),
     {'metavar': 'FEATURE_NORM',
      'dest': 'feature_norm',
      'default': False,
      'type': bool,
      'help': 'Normalize the features before training.'}),
    (('-wn', '--weight_norm'),
     {'metavar': 'WEIGHT_NORM',
      'dest': 'weight_norm',
      'default': False,
      'type': bool,
      'help': 'Normalize the weights after every forward pass during training.'}),
    (('-et', '--estimator_type'),
     {'metavar': 'ESTIMATOR_TYPE',
      'dest': 'estimator_type',
      'default': 'NaiveBayes',
      'type': str,
      'help': 'Which weak labelling and domain generation estimator to use. One of {NaiveBayes, Logistic, TupleEmbedding}.'}),
    (('-ee', '--estimator_epochs'),
     {'metavar': 'ESTIMATOR_EPOCHS',
      'dest': 'estimator_epochs',
      'default': 10,
      'type': int,
      'help': 'Number of epochs to run the weak labelling and domain generation estimator.'}),
    (('-ebs', '--estimator_batch_size'),
     {'metavar': 'ESTIMATOR_BATCH_SIZE',
      'dest': 'estimator_batch_size',
      'default': 32,
      'type': int,
      'help': 'Size of batch used in SGD in the weak labelling and domain generation estimator.'}),
    (('-ees', '--estimator_embedding_size'),
     {'metavar': 'ESTIMATOR_EMBEDDING_SIZE',
      'dest': 'estimator_embedding_size',
      'default': 10,
      'type': int,
      'help': 'If embeding_type = TupleEmbedding, uses this for the embedding size of the learned embedding vectors.'}),
    (('-ta', '--train_attrs'),
     {'metavar': 'TRAIN_ATTRS',
      'dest': 'train_attrs',
      'default': None,
      'type': list,
      'help': 'List of attributes to train and infer on. If None, train and infer on all columns. For example passing a list of one column allows one to train HoloClean on one column.'}),
    (('-im', '--infer_mode'),
     {'metavar': 'INFER_MODE',
      'dest': 'infer_mode',
      'default': 'dk',
      'type': str,
      'help': 'Infer on only possibly erroneous (DK) cells or all cells. One of {dk, all}.'}),
]

# Flags for Holoclean mode
flags = [
    (tuple(['--verbose']),
        {'default': False,
         'dest': 'verbose',
         'action': 'store_true',
         'help': 'verbose'}),
    (tuple(['--bias']),
        {'default': False,
         'dest': 'bias',
         'action': 'store_true',
         'help': 'Use bias term'}),
    (tuple(['--printfw']),
        {'default': False,
         'dest': 'print_fw',
         'action': 'store_true',
         'help': 'print the weights of featurizers'}),
    (tuple(['--debug-mode']),
        {'default': False,
         'dest': 'debug_mode',
         'action': 'store_true',
         'help': 'dump a bunch of debug information to debug\/'}),
]


class HoloClean:
    """
    Main entry point for HoloClean.
    It creates a HoloClean Data Engine
    """

    def __init__(self, **kwargs):
        """
        Constructor for Holoclean
        :param kwargs: arguments for HoloClean
        """

        # Initialize default execution arguments
        arg_defaults = {}
        for arg, opts in arguments:
            if 'directory' in arg[0]:
                arg_defaults['directory'] = opts['default']
            else:
                arg_defaults[opts['dest']] = opts['default']

        # Initialize default execution flags
        for arg, opts in flags:
            arg_defaults[opts['dest']] = opts['default']

        # check env vars
        for arg, opts in arguments:
            # if env var is set use that
            if opts["metavar"] and opts["metavar"] in os.environ.keys():
                logging.debug(
                    "Overriding {} with env varible {} set to {}".format(
                        opts['dest'],
                        opts["metavar"],
                        os.environ[opts["metavar"]])
                )
                arg_defaults[opts['dest']] = os.environ[opts["metavar"]]

        # Override defaults with manual flags
        for key in kwargs:
            arg_defaults[key] = kwargs[key]

        # Initialize additional arguments
        for (arg, default) in arg_defaults.items():
            setattr(self, arg, kwargs.get(arg, default))

        # Init empty session collection
        self.session = Session(arg_defaults)


class Session:
    """
    Session class controls the entire pipeline of HC
    """

    def __init__(self, env, name="session"):
        """
        Constructor for Holoclean session
        :param env: Holoclean environment
        :param name: Name for the Holoclean session
        """
        # use DEBUG logging level if verbose enabled
        if env['verbose']:
            root_logger.setLevel(logging.DEBUG)
            gensim_logger.setLevel(logging.DEBUG)

        logging.debug('initiating session with parameters: %s', env)

        # Initialize random seeds.
        random.seed(env['seed'])
        torch.manual_seed(env['seed'])
        np.random.seed(seed=env['seed'])

        # Initialize members
        self.name = name
        self.env = env
        self.ds = Dataset(name, env)
        self.dc_parser = Parser(env, self.ds)
        self.domain_engine = DomainEngine(env, self.ds)
        self.detect_engine = DetectEngine(env, self.ds)
        self.repair_engine = RepairEngine(env, self.ds)
        self.eval_engine = EvalEngine(env, self.ds)

    def load_data(self, name, fpath, na_values=None, entity_col=None, src_col=None,
                  exclude_attr_cols=None, numerical_attrs=None):
        """
        load_data takes the filepath to a CSV file to load as the initial dataset.

        :param name: (str) name to initialize dataset with.
        :param fpath: (str) filepath to CSV file.
        :param na_values: (str) value that identifies a NULL value
        :param entity_col: (st) column containing the unique
            identifier/ID of an entity.  For fusion tasks, rows with
            the same ID will be fused together in the output.
            If None, assumes every row is a unique entity.
        :param src_col: (str) if not None, for fusion tasks
            specifies the column containing the source for each "mention" of an
            entity.
        :param exclude_attr_cols: (str list)
        :param numerical_attrs: (str list)
        """
        status, load_time = self.ds.load_data(name,
                                              fpath,
                                              na_values=na_values,
                                              entity_col=entity_col,
                                              src_col=src_col,
                                              exclude_attr_cols=exclude_attr_cols,
                                              numerical_attrs=numerical_attrs)
        logging.info(status)
        logging.debug('Time to load dataset: %.2f secs', load_time)

    def load_dcs(self, fpath):
        """
        load_dcs ingests the Denial Constraints for initialized dataset.

        :param fpath: filepath to TXT file where each line contains one denial constraint.
        """
        status, load_time = self.dc_parser.load_denial_constraints(fpath)
        logging.info(status)
        logging.debug('Time to load dirty data: %.2f secs', load_time)

    def get_dcs(self):
        return self.dc_parser.get_dcs()

    def detect_errors(self, detect_list):
        status, detect_time = self.detect_engine.detect_errors(detect_list)
        logging.info(status)
        logging.debug('Time to detect errors: %.2f secs', detect_time)

    def disable_quantize(self):
        self.do_quantization = False
        self.ds.do_quantization = False
        self.domain_engine.do_quantization = False

    def quantize_numericals(self, num_attr_groups_bins):
        """
        :param num_attr_groups_bins: list[tuple] where each tuple consists of
        (# of bins, list[str]) where the list[str] is a group of attribues to be
        treated as numerical.
        """
        self.do_quantization = True
        self.ds.do_quantization = True
        self.domain_engine.do_quantization = True

        status, quantize_time, quantized_data = \
            quantize_km(self.env, self.ds.get_raw_data(), num_attr_groups_bins)

        logging.info(status)
        logging.debug('Time to quantize the dataset: %.2f secs' % quantize_time)

        self.load_quantized_data(quantized_data)


        return quantized_data

    def load_quantized_data(self, df):
        tic = time.time()
        name = self.ds.raw_data.name + '_quantized'
        self.ds.quantized_data = Table(name, Source.DF, df=df)

        # Re-store to DB, ensuring numerical values are stored as floats.
        df_correct_type = df.copy()
        for attr in self.ds.numerical_attrs:
            df_correct_type.loc[df_correct_type[attr] == NULL_REPR, attr] = np.nan
            df_correct_type[attr] = df_correct_type[attr].astype(float)
        df_correct_type.to_sql(name, self.ds.engine.engine, if_exists='replace', index=False,
                               index_label=None)

        for attr in self.ds.quantized_data.get_attributes():
            self.ds.quantized_data.create_db_index(self.ds.engine, [attr])
        logging.debug('Time to load quantized dataset: %.2f secs' % (time.time() - tic))



    def generate_domain(self):
        status, domain_time = self.domain_engine.setup()
        logging.info(status)
        logging.debug('Time to generate the domain: %.2f secs', domain_time)

    def run_estimator(self):
        """
        Uses estimator to weak label and prune domain.
        """
        self.domain_engine.run_estimator()

    def repair_errors(self, featurizers):
        return self._repair_errors(featurizers)

    def repair_validate_errors(self, featurizers, fpath, tid_col, attr_col,
            val_col, validate_period, na_values=None):
        return self._repair_errors(featurizers, fpath, tid_col, attr_col,
                val_col, na_values, validate_period)

    def _repair_errors(self, featurizers, fpath=None,
            tid_col=None, attr_col=None, val_col=None, na_values=None,
            validate_period=None):
        """
        Repair errors and optionally runs validation set per epoch.

        Must specify the following parameters if validation required:

        :param fpath: (str) filepath to test set (ground truth) CSV file.
        :param tid_col: (str) column in CSV that corresponds to the TID.
        :param attr_col: (str) column in CSV that corresponds to the attribute.
        :param val_col: (str) column in CSV that corresponds to correct value
            for the current TID and attribute (i.e. cell).
        :param na_values: (Any) how na_values are represented in the data.
        :param validate_period: (int) perform validation every nth epoch.
        """
        status, feat_time = self.repair_engine.setup_featurized_ds(featurizers)
        logging.info(status)
        logging.debug('Time to featurize data: %.2f secs', feat_time)
        status, setup_time = self.repair_engine.setup_repair_model()
        logging.info(status)
        logging.debug('Time to setup repair model: %.2f secs', feat_time)

        # If validation fpath provided, fit and validate
        if fpath is None:
            status, fit_time = self.repair_engine.fit_repair_model()
        else:
            # Set up validation set
            name = self.ds.raw_data.name + '_clean'
            status, load_time = self.eval_engine.load_data(name, fpath,
                    tid_col, attr_col, val_col, na_values=na_values)
            logging.info(status)
            logging.debug('Time to evaluate repairs: %.2f secs', load_time)

            status, fit_time = self.repair_engine.fit_validate_repair_model(self.eval_engine,
                    validate_period)

        logging.info(status)
        logging.debug('Time to fit repair model: %.2f secs', fit_time)
        status, infer_time = self.repair_engine.infer_repairs()
        logging.info(status)
        logging.debug('Time to infer correct cell values: %.2f secs', infer_time)
        status, time = self.ds.get_inferred_values()
        logging.info(status)
        logging.debug('Time to collect inferred values: %.2f secs', time)
        status, time = self.ds.get_repaired_dataset()
        logging.info(status)
        logging.debug('Time to store repaired dataset: %.2f secs', time)
        if self.env['print_fw']:
            status, time = self.repair_engine.get_featurizer_weights()
            logging.info(status)
            logging.debug('Time to store featurizer weights: %.2f secs', time)
            return status

    def evaluate(self, fpath, tid_col, attr_col, val_col, na_values=None):
        """
        evaluate generates an evaluation report with metrics (e.g. precision,
        recall) given a test set.

        :param fpath: (str) filepath to test set (ground truth) CSV file.
        :param tid_col: (str) column in CSV that corresponds to the TID.
        :param attr_col: (str) column in CSV that corresponds to the attribute.
        :param val_col: (str) column in CSV that corresponds to correct value
            for the current TID and attribute (i.e. cell).
        :param na_values: (Any) how na_values are represented in the data.

        Returns an EvalReport named tuple containing the experiment results.
        """
        name = self.ds.raw_data.name + '_clean'
        status, load_time = self.eval_engine.load_data(name, fpath, tid_col, attr_col, val_col, na_values=na_values)
        logging.info(status)
        logging.debug('Time to evaluate repairs: %.2f secs', load_time)
        status, report_time, eval_report = self.eval_engine.eval_report()
        logging.info(status)
        logging.debug('Time to generate report: %.2f secs', report_time)
        return eval_report

    def get_predictions(self):
        """
        Returns a dataframe with 3 columns:
            - tid, attribute, inferred_val, proba
        """

        query = """
        SELECT
            _tid_, attribute, inferred_val, prob
        FROM {dom}
        INNER JOIN {inf_vals} USING(_vid_)
        """.format(inf_vals=AuxTables.inf_values_idx.name,
                dom=AuxTables.cell_domain.name)
        res = self.ds.engine.execute_query(query)
        df_preds = pd.DataFrame(res,
                columns=['tid', 'attribute', 'inferred_val', 'proba'],
                dtype=str)
        return df_preds
