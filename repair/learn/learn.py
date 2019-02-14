import logging
import math

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import Parameter, ParameterList
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np


class TiedLinear(torch.nn.Module):
    """
    TiedLinear is a linear layer with shared parameters for features between
    (output) classes that takes as input a tensor X with dimensions
        (batch size) X (output_dim) X (in_features)
        where:
            output_dim is the desired output dimension/# of classes
            in_features are the features with shared weights across the classes
    """

    def __init__(self, env, feat_info, output_dim, bias=False):
        super(TiedLinear, self).__init__()
        self.env = env
        # Init parameters
        self.in_features = 0.0
        self.weight_list = ParameterList()
        if bias:
             self.bias_list = ParameterList()
        else:
             self.register_parameter('bias', None)
        self.output_dim = output_dim
        self.bias_flag = bias
        # Iterate over featurizer info list
        for feat_entry in feat_info:
            learnable = feat_entry.learnable
            feat_size = feat_entry.size
            init_weight = feat_entry.init_weight
            self.in_features += feat_size
            feat_weight = Parameter(init_weight*torch.ones(1, feat_size), requires_grad=learnable)
            if learnable:
                self.reset_parameters(feat_weight)
            self.weight_list.append(feat_weight)
            if bias:
                feat_bias = Parameter(torch.zeros(1, feat_size), requires_grad=learnable)
                if learnable:
                    self.reset_parameters(feat_bias)
                self.bias_list.append(feat_bias)

    def reset_parameters(self, tensor):
        stdv = 1. / math.sqrt(tensor.size(0))
        tensor.data.uniform_(-stdv, stdv)

    def concat_weights(self):
        self.W = torch.cat([t for t in self.weight_list],-1)
        # Normalize weights.
        if self.env['weight_norm']:
            self.W = self.W.div(self.W.norm(p=2))
        # expand so we can do matrix multiplication with each cell and their max # of domain values
        self.W = self.W.expand(self.output_dim, -1)
        if self.bias_flag:
            self.B = torch.cat([t.expand(self.output_dim, -1) for t in self.bias_list],-1)

    def forward(self, X, index, mask):
        # Concatenates different featurizer weights - need to call during every pass.
        self.concat_weights()
        output = X.mul(self.W)
        if self.bias_flag:
            output += self.B
        output = output.sum(2)
        # Add our mask so that invalid domain classes for a given variable/VID
        # has a large negative value, resulting in a softmax probability
        # of de facto 0.
        output.index_add_(0, index, mask)
        return output


class RepairModel:

    def __init__(self, env, feat_info, output_dim, bias=False):
        self.env = env
        # A list of tuples (name, is_featurizer_learnable, featurizer_output_size, init_weight, feature_names (list))
        self.feat_info = feat_info
        self.output_dim = output_dim
        self.model = TiedLinear(self.env, feat_info, output_dim, bias)
        self.featurizer_weights = {}

    def fit_model(self, X_train, Y_train, mask_train):
        n_examples, n_classes, n_features = X_train.shape

        loss = torch.nn.CrossEntropyLoss()
        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.env['optimizer'] == 'sgd':
            optimizer = optim.SGD(trainable_parameters, lr=self.env['learning_rate'], momentum=self.env['momentum'],
                                  weight_decay=self.env['weight_decay'])
        else:
            optimizer = optim.Adam(trainable_parameters, lr=self.env['learning_rate'], weight_decay=self.env['weight_decay'])

        # lr_sched = ReduceLROnPlateau(optimizer, 'min', verbose=True, eps=1e-5, patience=5)

        batch_size = self.env['batch_size']
        epochs = self.env['epochs']
        for i in tqdm(range(epochs)):
            cost = 0.
            num_batches = n_examples // batch_size
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                cost += self.__train__(loss, optimizer, X_train[start:end], Y_train[start:end], mask_train[start:end])

            # Y_pred = self.__predict__(X_train, mask_train)
            # train_loss = loss.forward(Y_pred, Variable(Y_train, requires_grad=False).squeeze(1))
            # logging.debug('overall training loss: %f', train_loss)
            # lr_sched.step(train_loss)

            if self.env['verbose']:
                # Compute and print accuracy at the end of epoch
                grdt = Y_train.numpy().flatten()
                Y_pred = self.__predict__(X_train, mask_train)
                Y_assign = Y_pred.data.numpy().argmax(axis=1)
                logging.debug("Epoch %d, cost = %f, acc = %.2f%%",
                        i + 1, cost / num_batches,
                        100. * np.mean(Y_assign == grdt))


    def infer_values(self, X_pred, mask_pred):
        logging.info('inferring on %d examples (cells)', X_pred.shape[0])
        output = self.__predict__(X_pred, mask_pred)
        return output

    def __train__(self, loss, optimizer, X_train, Y_train, mask_train):
        X_var = Variable(X_train, requires_grad=False)
        Y_var = Variable(Y_train, requires_grad=False)
        mask_var = Variable(mask_train, requires_grad=False)

        index = torch.LongTensor(range(X_var.size()[0]))
        index_var = Variable(index, requires_grad=False)

        optimizer.zero_grad()
        # Fully-connected layer with shared parameters between output classes
        # for linear combination of input features.
        # Mask makes invalid output classes have a large negative value so
        # to zero out softmax probability.
        fx = self.model.forward(X_var, index_var, mask_var)
        # loss is CrossEntropyLoss: combines log softmax + Negative log likelihood loss.
        # Y_Var is just a single 1D tensor with value (0 - 'class' - 1) i.e.
        # index of the correct class ('class' = max domain)
        # fx is a tensor of length 'class' the linear activation going in the softmax.
        output = loss.forward(fx, Y_var.squeeze(1))
        output.backward()
        optimizer.step()
        cost = output.item()
        return cost

    def __predict__(self, X_pred, mask_pred):
        X_var = Variable(X_pred, requires_grad=False)
        index = torch.LongTensor(range(X_var.size()[0]))
        index_var = Variable(index, requires_grad=False)
        mask_var = Variable(mask_pred, requires_grad=False)
        fx = self.model.forward(X_var, index_var, mask_var)
        output = softmax(fx, 1)
        return output

    def get_featurizer_weights(self, feat_info):
        report = ""
        for i, f in enumerate(feat_info):
            this_weight = self.model.weight_list[i].data.numpy()[0]
            weight_str = "\n".join("{name} {weight}".format(name=name, weight=weight)
                                   for name, weight in
                                   zip(f.feature_names, map(str, np.around(this_weight, 3))))
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
