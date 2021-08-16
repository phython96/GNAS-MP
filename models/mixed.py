import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import V_Package, OPS


class Mixed(nn.Module):
    
    def __init__(self, args, ops):
        super().__init__()
        self.args       = args
        self.type       = type
        self.OPS        = ops
        self.candidates = nn.ModuleDict({
            name: V_Package(args, OPS[name](args))
            for name in self.OPS
        })
    

    def forward(self, input, weight):
        '''
        weights 是以操作名为key, 操作权重为val的字典
        operation name : operation weight
        '''
        weight = weight.softmax(0)
        output = sum( weight[i] * self.candidates[name](input) for i, name in enumerate(self.OPS) )
        # residual = input[1] if self.type == 'V' else input[2]
        return output # + residual * DecayScheduler().decay_rate

