import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import V_Package, OPS


class Cell(nn.Module):

    def __init__(self, args, genotype):

        super().__init__()
        self.args           = args
        self.nb_nodes       = args.nb_nodes
        self.genotype       = genotype
        self.trans_concat_V = nn.Linear(args.nb_nodes * args.node_dim, args.node_dim, bias = True)
        self.batchnorm_V    = nn.BatchNorm1d(args.node_dim)
        self.activate       = nn.LeakyReLU(args.leaky_slope)
        self.load_genotype()
    

    def load_genotype(self):
        geno        = self.genotype
        link_dict   = {}
        module_dict = {}
        for edge in geno['topology']:
            src, dst, ops = edge['src'], edge['dst'], edge['ops']
            dst = f'{dst}'

            if dst not in link_dict:
                link_dict[dst] = []
            link_dict[dst].append(src)

            if dst not in module_dict:
                module_dict[dst] = nn.ModuleList([])
            module_dict[dst].append(V_Package(self.args, OPS[ops](self.args))) 

        self.link_dict   = link_dict
        self.module_dict = nn.ModuleDict(module_dict)
    

    def forward(self, input):

        G, V_in = input['G'], input['V']
        states = [V_in]
        for dst in range(1, self.nb_nodes + 1):
            dst = f'{dst}'
            agg = []
            for i, src in enumerate(self.link_dict[dst]):
                sub_input = {'G': G, 'V': states[src], 'V_in': V_in}
                agg.append(self.module_dict[dst][i](sub_input))
            states.append(sum(agg))

        V = self.trans_concat_V(torch.cat(states[1:], dim = 1))

        if self.batchnorm_V:
            V = self.batchnorm_V(V)

        V = self.activate(V)
        V = F.dropout(V, self.args.dropout, training = self.training)
        V = V + V_in
        return { 'G' : G, 'V' : V }


if __name__ == '__main__':
    import yaml 
    from easydict import EasyDict as edict
    geno = yaml.load(open('example_geno.yaml', 'r'))
    geno = geno['Genotype'][0]
    args = edict({
        'nb_nodes': 4,
        'node_dim': 50,
        'leaky_slope': 0.2, 
        'batchnorm_op': True, 
    })
    cell = Cell(args, geno)
    