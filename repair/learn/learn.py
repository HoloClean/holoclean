import logging

import torch
from torch import optim
from torch.nn import Parameter, ParameterList, ReLU
from torch.nn.functional import softmax
from tqdm import tqdm
import numpy as np


class TiedLinear(torch.nn.Module):
    """
    TiedLinear is a linear layer with shared parameters for features between
    (output) classes that takes as input a tensor X with dimensions
        (batch size) X (max_domain) X (total # of features)
        where:
            max_domain is the desired output dimension/# of classes
    """

    def __init__(self, env, feat_info, max_domain, layer_sizes, bias=False):
        """
        feat_info (list[FeatInfo]): list of FeatInfo namedtuples for each
            featurizer
        max_domain (int): number of domain values (e.g. max domain)
        bias (bool): use bias on the first layer per feature.
        layer_sizes (list[int]): Output size of each linear layer. Last layer
            should have an output size of 1. E.g. [200, 1].
        """
        assert layer_sizes and layer_sizes[-1] == 1

        super(TiedLinear, self).__init__()
        self.env = env
        self.act = ReLU()

        self.max_domain = max_domain
        self.bias_flag = bias

        # Create first layer: this layer is special since some weights
        # cannot be learned.
        self.first_layer_weights = ParameterList()
        for feat in feat_info:
            weight = Parameter(feat.init_weight*torch.ones(feat.size,
                                                           layer_sizes[0]),
                               requires_grad=feat.learnable)
            if feat.learnable:
                torch.nn.init.xavier_uniform_(weight)
            self.first_layer_weights.append(weight)

        if self.bias_flag:
            self.first_layer_bias = Parameter(torch.zeros(1, sum(f.size for f in feat_info)))
            torch.nn.init.xavier_uniform_(self.first_layer_bias)

        # Create subsequent layers.
        self.other_weights = ParameterList()
        self.other_biases = ParameterList()
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            weight = Parameter(torch.zeros(in_dim, out_dim))
            # (max_domain, out_dim)
            bias = Parameter(torch.zeros(1, out_dim))

            # Randomly initialize weights.
            torch.nn.init.xavier_uniform_(weight)
            torch.nn.init.xavier_uniform_(bias)

            self.other_weights.append(weight)
            self.other_biases.append(bias)

        logging.debug("training model with first layer size: %s",
                      list(map(str, [w.shape for w in self.first_layer_weights])))
        if len(self.other_weights):
            logging.debug("training model with additional hidden layers of size: %s",
                          list(map(str, [w.shape for w in self.other_weights])))

    def forward(self, X, index, mask):
        """
        Performs one forward pass and outputs the logits of size
        (batch, max_domain)

        X: (batch, # of classes, total # of features)
        index: (batch)
        mask: (batch, # of classes)
        """
        if X.shape[0] == 0:
            logging.warning("performing forward pass with no samples")
            return torch.zeros(0, X.shape[1])

        # Multiply through the first layer.
        # (batch, # of classes, layers_size[0])
        output = X.matmul(torch.cat([t for t in self.first_layer_weights],
                                    dim=0))
        if self.bias_flag:
            output.add_(self.first_layer_bias.expand(self.max_domain, -1))

        for idx, (weight, bias) in enumerate(zip(self.other_weights,
                                                 self.other_biases)):
            # Apply activation on all but last layer.
            output = self.act(output)
            # (batch, # of classes, in_dim) --> (batch, # of classes, out_dim)
            output = output.matmul(weight) + bias.expand(self.max_domain, -1)
        # output should now be (batch, # of classes, 1)

        # (batch, # of classes)
        output = output.squeeze(-1)

        # Add our mask so that invalid domain classes for a given variable/VID
        # has a large negative value, resulting in a softmax probability
        # of de facto 0.
        # (batch, # of classes)
        output.index_add_(0, index, mask)
        return output


class RepairModel:

    def __init__(self, env, feat_info, max_domain, bias=False, layer_sizes=[1]):
        """
        feat_info (list[FeatInfo]): featurizer information
        max_domain (int): maximum domain size i.e. output dimension
        """
        self.env = env
        self.feat_info = feat_info
        self.max_domain = max_domain
        self.model = TiedLinear(self.env, feat_info, max_domain,
                                bias=bias, layer_sizes=layer_sizes)
        self.featurizer_weights = {}

        self.loss = torch.nn.CrossEntropyLoss()
        trainable_parameters = filter(lambda p: p.requires_grad,
                                      self.model.parameters())
        if self.env['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(trainable_parameters,
                                  lr=self.env['learning_rate'],
                                  momentum=self.env['momentum'],
                                  weight_decay=self.env['weight_decay'])
        else:
            self.optimizer = optim.Adam(trainable_parameters,
                                   lr=self.env['learning_rate'],
                                   weight_decay=self.env['weight_decay'])

    def fit_model(self, X_train, Y_train, mask_train, epochs):
        """
        X_train: (batch, # of classes (domain size), total # of features)
        Y_train: (batch, 1)
        mask_train: (batch, # of classes)
        """
        batch_size = self.env['batch_size']
        for epoch_idx in tqdm(range(1, epochs + 1)):
            cost = 0.
            num_batches = (X_train.shape[0] + batch_size - 1) // batch_size
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                cost += self.__train__(X_train[start:end],
                                       Y_train[start:end],
                                       mask_train[start:end])

            if self.env['verbose']:
                # Compute and print accuracy at the end of epoch
                grdt = Y_train.numpy().flatten()
                Y_pred = self.__predict__(X_train, mask_train)
                Y_assign = Y_pred.data.numpy().argmax(axis=1)
                logging.debug("Epoch %d, cost = %f, acc = %.2f%%",
                              epoch_idx, cost / max(num_batches, 1),
                              100. * np.mean(Y_assign == grdt))

    def infer_values(self, X_pred, mask_pred):
        logging.info('inferring on %d examples (cells)', X_pred.shape[0])
        output = self.__predict__(X_pred, mask_pred)
        return output

    def __train__(self, X_train, Y_train, mask_train):
        """
        X_train: (batch, # of classes (domain size), total # of features)
        Y_train: (batch, 1)
        mask_train: (batch, # of classes)
        """
        index = torch.LongTensor(range(X_train.size()[0]))

        # Fully-connected layer with shared parameters between output classes
        # for linear combination of input features.
        # Mask makes invalid output classes have a large negative value so
        # to zero out softmax probability.
        fx = self.model.forward(X_train, index, mask_train)
        # loss is CrossEntropyLoss: combines log softmax + Negative log
        # likelihood loss.
        # Y_Var is just a single 1D tensor with value (0 - 'class' - 1) i.e.
        # index of the correct class ('class' = max domain)
        # fx is a tensor of length 'class' the linear activation going in the
        # softmax.
        output = self.loss.forward(fx, Y_train.squeeze(1))

        self.optimizer.zero_grad()
        output.backward()
        self.optimizer.step()

        cost = output.item()
        return cost

    def __predict__(self, X_pred, mask_pred):
        """
        X_pred: (batch, # of classes (domain size), total # of features)
        Y_pred: (batch, 1)
        """
        index = torch.LongTensor(range(X_pred.size()[0]))
        fx = self.model.forward(X_pred, index, mask_pred)
        output = softmax(fx, 1)
        return output

    def get_featurizer_weights(self, feat_info):
        report = ""
        for i, f in enumerate(feat_info):
            # TODO: fix this since we now have 200 weights (second layer size)
            # per feature
            this_weight = self.model.first_layer_weights[i].data\
                .numpy().mean(axis=1)
            weight_str = "\n".join("{name} {weight}".format(name=name,
                                                            weight=weight)
                                   for name, weight in
                                   zip(f.feature_names,
                                       map(str, np.around(this_weight, 3))))
            feat_name = f.name
            feat_size = f.size
            max_w = max(this_weight)
            min_w = min(this_weight)
            mean_w = float(np.mean(this_weight))
            abs_mean_w = float(np.mean(np.absolute(this_weight)))
            # Create report
            report += "featurizer %s,size %d,max %.4f,min %.4f,avg %.4f,abs_avg %.4f,weights:\n%s\n" % (
                feat_name, feat_size, max_w, min_w, mean_w, abs_mean_w, weight_str
            )
            # Wrap in a dictionary.
            self.featurizer_weights[feat_name] = {
                'max': max_w,
                'min': min_w,
                'avg': mean_w,
                'abs_avg': abs_mean_w,
                'weights': this_weight,
                'size': feat_size
            }
        return report
