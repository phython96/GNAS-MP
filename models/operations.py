import torch
import dgl.function as fn
import torch.nn as nn
import numpy as np
# from models.networks import *


OPS = {
    'V_None'  : lambda args: V_None(args),
    'V_I'     : lambda args: V_I(args),
    'V_Max'   : lambda args: V_Max(args),
    'V_Mean'  : lambda args: V_Mean(args),
    'V_Min'   : lambda args: V_Min(args),
    'V_Sum'   : lambda args: V_Sum(args),
    'V_Sparse': lambda args: V_Sparse(args),
    'V_Dense' : lambda args: V_Dense(args),
}


First_Stage  = ['V_None', 'V_I', 'V_Sparse', 'V_Dense']
Second_Stage = ['V_I', 'V_Mean', 'V_Sum', 'V_Max']
Third_Stage  = ['V_None', 'V_I', 'V_Sparse', 'V_Dense']


class V_Package(nn.Module):

    def __init__(self, args, operation):
        
        super().__init__()
        self.args      = args
        self.operation = operation
        if type(operation) in [V_None, V_I]:
            self.seq = None
        else:
            self.seq = nn.Sequential()
            self.seq.add_module('fc_bn', nn.Linear(args.node_dim, args.node_dim, bias = True))
            if args.batchnorm_op:
                self.seq.add_module('bn', nn.BatchNorm1d(self.args.node_dim))
            self.seq.add_module('act', nn.ReLU())


    def forward(self, input):
        V = self.operation(input)
        if self.seq:
            V = self.seq(V)
        return V 


class NodePooling(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.A        = nn.Linear(args.node_dim, args.node_dim)
        # self.B        = nn.Linear(args.node_dim, args.node_dim)
        self.activate = nn.ReLU()

    def forward(self, V):
        V = self.A(V)
        V = self.activate(V)
        # V = self.B(V)
        return V


class V_None(nn.Module):

    def __init__(self, args):
        super().__init__()
    
    def forward(self, input):
        V = input['V']
        return 0. * V


class V_I(nn.Module):
    
    def __init__(self, args):
        super().__init__()
    
    def forward(self, input):
        V = input['V']
        return V


class V_Max(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pooling = NodePooling(args)

    def forward(self, input):
        G, V = input['G'], input['V']
        # G.ndata['V'] = V
        G.ndata['V'] = self.pooling(V)
        G.update_all(fn.copy_u('V', 'M'), fn.max('M', 'V'))
        return G.ndata['V']


class V_Mean(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pooling = NodePooling(args)
        
    def forward(self, input):
        G, V = input['G'], input['V']
        # G.ndata['V'] = V
        G.ndata['V'] = self.pooling(V)
        G.update_all(fn.copy_u('V', 'M'), fn.mean('M', 'V'))
        return G.ndata['V']


class V_Sum(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.pooling = NodePooling(args)
        
    def forward(self, input):
        G, V = input['G'], input['V']
        # G.ndata['V'] = self.pooling(V)
        G.ndata['V'] = V
        G.update_all(fn.copy_u('V', 'M'), fn.sum('M', 'V'))
        return G.ndata['V']


class V_Min(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.pooling = NodePooling(args)
        
    def forward(self, input):
        G, V = input['G'], input['V']
        G.ndata['V'] = self.pooling(V)
        G.update_all(fn.copy_u('V', 'M'), fn.min('M', 'V'))
        return G.ndata['V']


class V_Dense(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.W = nn.Linear(args.node_dim*2, args.node_dim, bias = True)

    def forward(self, input):
        V, V_in = input['V'], input['V_in']
        gates = torch.cat([V, V_in], dim = 1)
        gates = self.W(gates)
        return torch.sigmoid(gates) * V


class V_Sparse(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.W = nn.Linear(args.node_dim*2, args.node_dim, bias = True)
        self.a = nn.Linear(args.node_dim, 1, bias = False)

    def forward(self, input):
        V, V_in = input['V'], input['V_in']
        gates = torch.cat([V, V_in], dim = 1)
        # gates = self.W(gates)
        gates = torch.relu(self.W(gates))
        gates = self.a(gates)
        return torch.sigmoid(gates) * V


if __name__ == '__main__':
    print("test")