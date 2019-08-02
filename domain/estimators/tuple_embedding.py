from ..estimator import Estimator
from domain.estimators.embedding.dataset import LookupDataset
from domain.estimators.embedding.trainer import Trainer
from domain.estimators.embedding.model import EmbeddingModel
from domain.estimators.embedding.predictor import Predictor
from torch.nn import CrossEntropyLoss, Softmax, MSELoss
from torch.optim import Adam
from dataset import AuxTables
import logging

def verify_numerical_attr_groups(dataset, numerical_attr_groups):
    """
    Verify numerical attribute groups are disjoint and exist

    Returns a list of the individual attributes.
    """
    numerical_attrs = None
    # Check if numerical attributes exist and are disjoint
    if numerical_attr_groups is not None:
        numerical_attrs = [attr for group in numerical_attr_groups for attr in group]

        if not all(attr in dataset.get_attributes() for attr in numerical_attrs):
            logging.error('all numerical attributes specified %s must exist in dataset: %s',
                    numerical_attr_groups,
                    dataset.get_attributes())
            raise Exception()

        if len(set(numerical_attrs)) < len(numerical_attrs):
            logging.error('all attribute groups specified %s must be disjoint in dataset',
                    numerical_attr_groups)
            raise Exception()

    return numerical_attrs

def input_sanity_check(train_attrs, attrs, numerical_attr_groups, embed_size):
    # Check if train attributes exist
    if train_attrs is not None:
        if not all(attr in attrs for attr in train_attrs):
            logging.error('%s: all attributes specified to use for training %s must exist in dataset: %s',
                    type(self).__name__,
                    train_attrs,
                    attrs)
            raise Exception()

    # Verify numerical dimensions are not bigger than the embedding size
    if  max(list(map(len, numerical_attr_groups)) or [0]) > embed_size:
        logging.error("%s: maximum numeric value dimension %d must be <= embedding size %d",
                type(self).__name__,
                max(list(map(len, numerical_attr_groups)) or [0]),
                embed_size)
        raise Exception()

class TupleEmbedding(Estimator):
    WEIGHT_DECAY = 0.

    def __init__(self, env, dataset, domain_df,
            numerical_attr_groups=None,
            memoize=False,
            neg_sample=True,
            dropout_pct=0.,
            learning_rate=0.05,
            validate_fpath=None, validate_tid_col=None, validate_attr_col=None,
            validate_val_col=None, validate_epoch=None):

        """
        :param dataset: (Dataset) original dataset
        :param domain_df: (DataFrame) dataframe containing domain values
        :param numerical_attr_groups: (list[list[str]]) attributes/columns to treat as numerical.
            A list of groups of column names. Each group consists of d attributes
            to be treated as d-dimensional numerical attribute.
            For example one can pass in [['lat', 'lon'],...] to treat both columns as
            a 2-d numerical attribute.

            The groups must be disjoint.

            Everything else will be treated as categorical.

            If None, treats everything as categorical.
        :param neg_sample: (bool) add negative examples for clean cells during training
        :param validate_fpath: (string) filepath to validation CSV
        :param validate_tid_col: (string) column containing TID
        :param validate_attr_col: (string) column containing attribute
        :param validate_val_col: (string) column containing correct value
        """

        Estimator.__init__(self, env, dataset, domain_df)
        numerical_attr_groups = numerical_attr_groups or []
        self.train_attrs = self.env['train_attrs']
        self.embed_size = self.env['estimator_embedding_size']
        self._numerical_attrs = verify_numerical_attr_groups(self.ds, numerical_attr_groups) or []

        input_sanity_check(self.train_attrs, self.attrs, numerical_attr_groups, self.embed_size)
        self.domain_recs = self._domain_df_process()

        # initialize lookup dataset for embedding model
        self.dataset = LookupDataset(env, self.ds, self.domain_df, numerical_attr_groups, neg_sample, memoize)

        # initialize embedding model
        self.model = EmbeddingModel(self.train_attrs, self.embed_size, self.dataset)

        # initialize predictor
        self.predictor = Predictor(self.model, self.dataset, self.domain_df, self.domain_recs)

        # initialize loss function for training categorical and numerical attribute
        num_loss =  MSELoss(reduction='mean')
        cat_loss = CrossEntropyLoss()

        # initialize optimizer

        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.WEIGHT_DECAY)

        # initialize trainer
        self.trainer = Trainer(self.model, num_loss, cat_loss, optimizer, self.dataset, self.predictor, self.domain_df)

    def train(self, num_epochs=10, batch_size=32, weight_entropy_lambda=0.,shuffle=True, train_only_clean=False):
        return self.trainer.train(num_epochs, batch_size, weight_entropy_lambda, shuffle, train_only_clean)
    
    def predict_pp_batch(self):
        return self.predictor.predict_pp_batch()

    def _domain_df_process(self):
        # Remove domain for numerical attributes.
        fil_numattr = self.domain_df['attribute'].isin(self._numerical_attrs)
        self.domain_df.loc[fil_numattr, 'domain'] = ''
        self.domain_df.loc[fil_numattr, 'domain_size'] = 0

        # Remove categorical domain/training cells without a domain
        filter_empty_domain = (self.domain_df['domain_size'] == 0) & ~fil_numattr
        if filter_empty_domain.sum():
            logging.warning('%s: removing %d categorical cells with empty domains',
                type(self).__name__,
                filter_empty_domain.sum())
            self.domain_df = self.domain_df[~filter_empty_domain]
        # Pre-split domain.
        self.domain_df['domain'] = self.domain_df['domain'].str.split('\|\|\|')

        # Add DK information to domain dataframe
        if self.ds.aux_table[AuxTables.dk_cells] is not None:
            df_dk = self.ds.aux_table[AuxTables.dk_cells].df
            self.domain_df = self.domain_df.merge(df_dk,
                    on=['_tid_', 'attribute'], how='left', suffixes=('', '_dk'))
            self.domain_df['is_clean'] = self.domain_df['_cid__dk'].isnull()
        else:
            self.domain_df['is_clean'] = True
            self.domain_df = self.domain_df[self.domain_df['attribute'].isin(self.train_attrs)]

        return self.domain_df.to_records()
    









