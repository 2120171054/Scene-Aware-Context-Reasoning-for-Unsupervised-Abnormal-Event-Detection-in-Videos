import  torch
from    torch import nn
from    torch.nn import functional as F
import math


class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 bias=True,
                 activation = None,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        # self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.featureless = featureless
        self.dropout_layer = nn.Dropout(p=dropout)
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, inputs):
        # print('inputs:', inputs)
        x=inputs[0]
        support=inputs[1]
        # print(x.size())
        # print(support.size())
        # if self.training :
            # x = F.dropout(x, self.dropout)
        x=self.dropout_layer(x)
        # convolve
        if not self.featureless: # if it has features x
            xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.spmm(support, xw)
        # print(out.size())
        if self.bias is not None:
            out += self.bias
        if self.activation is not None:
            return self.activation(out),support
        else:
            return out, support