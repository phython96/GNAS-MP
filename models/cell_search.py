import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import OPS
from models.mixed import Mixed

'''
cell_arch : 
    topology: list
        (src, dst, weights, ops)
'''

class Cell(nn.Module):

    def __init__(self, args, cell_arch):
        super().__init__()
        self.args           = args
        self.nb_nodes       = args.nb_nodes*3 #! warning 
        self.cell_arch      = cell_arch
        self.trans_concat_V = nn.Linear(self.nb_nodes*args.node_dim, args.node_dim, bias = True)
        self.batchnorm_V    = nn.BatchNorm1d(args.node_dim)
        self.activate       = nn.LeakyReLU(args.leaky_slope)
        self.load_arch()


    def load_arch(self):
        link_para = {}
        link_dict = {}
        for src, dst, w, ops in self.cell_arch:
            if dst not in link_dict:
                link_dict[dst] = []
            link_dict[dst].append((src, w))
            link_para[str((src, dst))] = Mixed(self.args, ops)
        
        self.link_dict = link_dict
        self.link_para = nn.ModuleDict(link_para)


    def forward(self, input, weight):
        G, V_in = input['G'], input['V']
        link_para = self.link_para
        link_dict = self.link_dict
        states = [V_in]
        for dst in range(1, self.nb_nodes + 1):
            tmp_states = []
            for src, w in link_dict[dst]:
                sub_input = {'G': G, 'V': states[src], 'V_in': V_in}
                tmp_states.append(link_para[str((src, dst))](sub_input, weight[w]))
            states.append(sum(tmp_states))
            
        V = self.trans_concat_V(torch.cat(states[1:], dim = 1))

        if self.batchnorm_V:
            V = self.batchnorm_V(V)
            
        V = self.activate(V)
        V = F.dropout(V, self.args.dropout, training = self.training)
        V = V + V_in
        return {'G': G, 'V': V}

