import math
import torch
from torch.nn import Parameter
from torch.autograd import Variable
from torch import optim
from torch.nn.functional import softmax
from tqdm import tqdm
import numpy as np


class TiedLinear(torch.nn.Module):
    """
    TiedLinear is a linear layer with shared parameters for features between
    (output) classes that takes as input a tensor X with dimensions
        (batch size) X (output_dim) X (in_features)
        where:
            output_dim is the disired output dimension/# of classes
            in_features are the features with shared weights across the classes
    """

    def __init__(self, in_features, output_dim, bias=False):
        super(TiedLinear, self).__init__()
        self.in_features = in_features
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(1,in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1,in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # Broadcast parameters to the correct matrix dimensions for matrix
        # multiplication: this does NOT create new parameters: i.e. each
        # row of in_features of parameters are connected and will adjust
        # to the same values.
        self.W = self.weight.expand(output_dim, -1)
        if self.bias is not None:
            self.B = self.bias.expand(output_dim, -1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, index, mask):
        output = X.mul(self.W)
        if self.bias is not None:
            output += self.B
        output = output.sum(2)
        # Add our mask so that invalid domain classes for a given variable/VID
        # has a large negative value, resulting in a softmax probability
        # of de facto 0.
        output.index_add_(0, index, mask)
        return output


class RepairModel:

    def __init__(self, env, in_features, output_dim, bias=False):
        self.env = env
        torch.manual_seed(self.env['seed'])
        self.in_features = in_features
        self.output_dim = output_dim
        self.model = TiedLinear(in_features, output_dim, bias)

    def fit_model(self, X_train, Y_train, mask_train):
        n_examples, n_classes, n_features = X_train.shape
        loss = torch.nn.CrossEntropyLoss()
        if self.env['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.env['learning_rate'], momentum=self.env['momentum'],
                                  weight_decay=self.env['weight_decay'])
        else:
            optimizer = optim.Adam(self.model.parameters(), weight_decay=self.env['weight_decay'])
        batch_size = self.env['batch_size']
        epochs = self.env['epochs']
        for i in tqdm(range(epochs)):
            cost = 0.
            num_batches = n_examples // batch_size
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                cost += self.__train__(loss, optimizer, X_train[start:end], Y_train[start:end],
                                   mask_train[start:end])
            if self.env['verbose']:
                # Compute and print accuracy at the end of epoch
                grdt = Y_train.numpy().flatten()
                Y_pred = self.__predict__(X_train, mask_train)
                Y_assign = Y_pred.data.numpy().argmax(axis=1)
                print("Epoch %d, cost = %f, acc = %.2f%%" %
                      (i + 1, cost / num_batches,
                       100. * np.mean(Y_assign == grdt)))

    def infer_values(self, X_pred, mask_pred):
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

