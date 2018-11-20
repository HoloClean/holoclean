import logging

from dataset import Dataset
from dcparser import Parser
from domain import DomainEngine
from detect import DetectEngine
from repair import RepairEngine
from evaluate import EvalEngine

# Arguments for HoloClean
arguments = [
    (('-u', '--db_user'),
        {'metavar': 'DB_USER',
         'dest': 'db_user',
         'default': 'holocleanuser',
         'type': str,
         'help': 'User for DB used to persist state.'}),
    (('-p', '--password', '--pass'),
        {'metavar': 'PASSWORD',
         'dest': 'db_pwd',
         'default': 'abcd1234',
         'type': str,
         'help': 'Password for DB used to persist state.'}),
    (('-h', '--host'),
        {'metavar': 'HOST',
         'dest': 'db_host',
         'default': 'localhost',
         'type': str,
         'help': 'Host for DB used to persist state.'}),
    (('-d', '--database'),
        {'metavar': 'DATABASE',
         'dest': 'db_name',
         'default': 'holo',
         'type': str,
         'help': 'Name of DB used to persist state.'}),
    (('-t', '--threads'),
     {'metavar': 'THREADS',
      'dest': 'threads',
      'default': 20,
      'type': int,
      'help': 'How many threads to use for parallel execution.'}),
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
    (('-l', '--learning-rate'),
     {'metavar': 'LEARNING_RATE',
      'dest': 'learning_rate',
      'default': 0.001,
      'type': float,
      'help': 'The learning rate used during training.'}),
    (('-k', '--pruning-topk'),
     {'metavar': 'PRUNING_TOPK',
      'dest': 'pruning_topk',
      'default': 10,
      'type': float,
      'help': 'Top-k used for domain pruning step.'}),
    (('-o', '--optimizer'),
     {'metavar': 'OPTIMIZER',
      'dest': 'optimizer',
      'default': 'adam',
      'type': str,
      'help': 'Optimizer used for learning.'}),
    (('-e', '--epochs'),
     {'metavar': 'LEARNING_EPOCHS',
      'dest': 'epochs',
      'default': 100,
      'type': float,
      'help': 'Number of epochs used for training.'}),
    (('-w', '--weight_decay'),
     {'metavar': 'WEIGHT_DECAY',
      'dest':  'weight_decay',
      'default': 0.0,
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
      'help': 'The batch size during training.'})
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
         'help': 'Use bias term'})
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

        # Initialize members
        self.name = name
        self.env = env
        self.ds = Dataset(name,env)
        self.dc_parser = Parser(env, self.ds)
        self.domain_engine = DomainEngine(env, self.ds)
        self.detect_engine = DetectEngine(env, self.ds)
        self.repair_engine = RepairEngine(env, self.ds)
        self.eval_engine = EvalEngine(env, self.ds)

        # use DEBUG logging level if verbose enabled
        logger = logging.getLogger()
        log_level = logging.INFO
        if self.env['verbose']:
            log_level = logging.DEBUG
        logger.setLevel(log_level)


    def load_data(self, name, f_path, f_name, na_values=None):
        status, load_time = self.ds.load_data(name, f_path,f_name, na_values=na_values)
        logging.info(status)
        logging.debug('Time to load dataset: %.2f secs'%load_time)

    def load_dcs(self, f_path, f_name):
        status, load_time = self.dc_parser.load_denial_constraints(f_path, f_name)
        logging.info(status)
        logging.debug('Time to load dirty data: %.2f secs'%load_time)

    def get_dcs(self):
        return self.dc_parser.get_dcs()

    def detect_errors(self, detect_list):
        status, detect_time = self.detect_engine.detect_errors(detect_list)
        logging.info(status)
        logging.debug('Time to detect errors: %.2f secs'%detect_time)

    def setup_domain(self):
        status, domain_time = self.domain_engine.setup()
        logging.info(status)
        logging.debug('Time to setup the domain: %.2f secs'%domain_time)

    def repair_errors(self, featurizers):
        status, feat_time = self.repair_engine.setup_featurized_ds(featurizers)
        logging.info(status)
        logging.debug('Time to featurize data: %.2f secs'%feat_time)
        status, setup_time = self.repair_engine.setup_repair_model()
        logging.info(status)
        logging.debug('Time to setup repair model: %.2f secs' % feat_time)
        status, fit_time = self.repair_engine.fit_repair_model()
        logging.info(status)
        logging.debug('Time to fit repair model: %.2f secs'%fit_time)
        status, infer_time = self.repair_engine.infer_repairs()
        logging.info(status)
        logging.debug('Time to infer correct cell values: %.2f secs'%infer_time)
        status, time = self.ds.get_inferred_values()
        logging.info(status)
        logging.debug('Time to collect inferred values: %.2f secs' % time)
        status, time = self.ds.get_repaired_dataset()
        logging.info(status)
        logging.debug('Time to store repaired dataset: %.2f secs' % time)

    def evaluate(self, f_path, f_name, get_tid, get_attr, get_value, na_values=None):
        name = self.ds.raw_data.name + '_clean'
        status, load_time = self.eval_engine.load_data(name, f_path, f_name, get_tid, get_attr, get_value, na_values=na_values)
        logging.info(status)
        logging.debug('Time to evaluate repairs: %.2f secs'%load_time)
        status, report_time = self.eval_engine.eval_report()
        logging.info(status)
        logging.debug('Time to generate report: %.2f secs' % report_time)
