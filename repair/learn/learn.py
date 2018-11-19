import math
import torch
from torch.nn import Parameter
from torch.autograd import Variable
from torch import optim
from torch.nn.functional import softmax
from tqdm import tqdm
import numpy as np


class TiedLinear(torch.nn.Module):

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
        output.index_add_(0, index, mask)
        return output


class RepairModel:

    def __init__(self, env, in_features, output_dim, bias=False):
        self.env = env
        torch.manual_seed(self.env['seed'])
        self.in_features = in_features
        self.output_dim = output_dim
        self.model = TiedLinear(in_features, output_dim, bias)
        self.featurizer_weights = {}

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
        fx = self.model.forward(X_var, index_var, mask_var)
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
        weight = self.model.state_dict()['weight'].cpu().numpy()[0]
        weight = map(lambda x: round(x, 4), weight)
        begin = 0
        report = ""
        for f in feat_info:
            this_weight = weight[begin:begin+f[1]]
            weight_str = " | ".join(map(str, this_weight))
            feat_name = f[0].split('.')[-1].split("'>")[0]
            max_w = max(this_weight)
            min_w = min(this_weight)
            mean_w = float(np.mean(this_weight))
            abs_mean_w = float(np.mean(np.absolute(this_weight)))
            # create report
            report += "featurizer %s,size %d,max %.4f,min %.4f,avg %.4f,abs_avg %.4f,weight %s\n" % (
                feat_name, int(f[1]), max_w, min_w, mean_w, abs_mean_w, weight_str
            )
            # create dictionary
            self.featurizer_weights[feat_name] = {
                'max': max_w,
                'min': min_w,
                'avg': mean_w,
                'abs_avg': abs_mean_w,
                'weights': this_weight,
                'size': f[1]
            }
            begin = begin+f[1]
        return report

