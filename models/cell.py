import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from configs.genotypes import Genotype
from models.operations import FIRST_OPS, MIDDLE_OPS, LAST_OPS, MIXED_OPS

class MixedOp(nn.Module):
    def __init__(self, feature_dim, operations):
        super(MixedOp, self).__init__()
        self._feature_dim = feature_dim
        self._operations = operations
        self._args = {'feature_dim': self._feature_dim}
        self._ops = nn.ModuleList([nn.ModuleList([MIXED_OPS[op_name](self._args),
                                                  nn.Linear(self._feature_dim, self._feature_dim, bias = True),
                                                  nn.BatchNorm1d(self._feature_dim),
                                                  nn.ReLU()]) 
                                   for op_name in self._operations]) 

    def forward(self, weights, g, h, h_in):
        output = sum( w * self.op_forward(op, g, h, h_in) for w, op in zip(weights, self._ops) )
        return output

    def op_forward(self, op, g, h, h_in):
        nh = op[0](g, h, h_in)
        for i in range(1, len(op)):
            nh = op[i](nh)
        return nh

class Cell_First(nn.Module):

    def __init__(self, nodes, feature_dim):
        super(Cell_First, self).__init__()
        self._nodes = nodes
        self._feature_dim = feature_dim
        self._ops = nn.ModuleList()
        for i in range(nodes):
            for j in range(i + 1):
                mixed_op = MixedOp(feature_dim, operations = FIRST_OPS)
                self._ops.append(mixed_op)

    def forward(self, g, h, h_in, weights):
        states = [h]
        offset = 0
        for i in range(self._nodes):
            s = sum(self._ops[offset + j](weights[offset + j], g, h, h_in) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return states[1:]

class Cell_Middle(nn.Module):

    def __init__(self, nodes, feature_dim):
        super(Cell_Middle, self).__init__()
        self._nodes = nodes
        self._feature_dim = feature_dim
        self._ops = nn.ModuleList()
        for i in range(nodes):
            self._ops.append(MixedOp(feature_dim, operations = MIDDLE_OPS))

    def forward(self, g, states, h_in, weights):
        '''
        states  len : nb_nodes
        weights len : nb_nodes * nb_operations
        '''
        output = [self._ops[i](weights[i], g, states[i], h_in) for i in range(self._nodes)]
        return output

class Cell_Last(nn.Module):

    def __init__(self, in_nodes, nodes, feature_dim):
        super(Cell_Last, self).__init__()
        self._in_nodes = in_nodes
        self._nodes = nodes
        self._feature_dim = feature_dim
        self._ops = nn.ModuleList()
        for i in range(nodes):
            for j in range(i + in_nodes):
                mixed_op = MixedOp(feature_dim, operations = LAST_OPS)
                self._ops.append(mixed_op)

    def forward(self, g, states, h_in, weights):
        '''
        states len : nb_in_nodes
        '''
        offset = 0
        for i in range(self._nodes):
            s = sum(self._ops[offset + j](weights[offset + j], g, h, h_in) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return states

class Cell(nn.Module):

    def __init__(self, nb_first_nodes, nb_last_nodes, feature_dim, dropout = 0.0):
        super(Cell, self).__init__()
        self._nb_first_nodes = nb_first_nodes
        self._nb_last_nodes = nb_last_nodes
        self._feature_dim = feature_dim
        self._dropout = dropout
        self.cell_first  = Cell_First(nb_first_nodes, feature_dim)
        self.cell_middle = Cell_Middle(nb_first_nodes, feature_dim)
        self.cell_last   = Cell_Last(nb_first_nodes, nb_last_nodes, feature_dim)

        self.concat_weights = nn.Linear((nb_first_nodes + nb_last_nodes) * feature_dim, feature_dim)
        self.batchnorm_h = nn.BatchNorm1d(feature_dim)
        self.activate = nn.ReLU()

    def forward(self, g, h, weights_first, weights_middle, weights_last):
        h_in = h

        states = self.cell_first(g, h, h_in, weights_first)
        states = self.cell_middle(g, states, h_in, weights_middle)
        states = self.cell_last(g, states, h_in, weights_last)

        h = self.concat_weights(torch.cat(states, dim=1))
        h = self.batchnorm_h(h)
        h = self.activate(h)
        h = F.dropout(h, self._dropout, training = self.training)
        h = h_in + h
        return h