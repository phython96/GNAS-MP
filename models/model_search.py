import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.cell_search import Cell
from models.operations import OPS, First_Stage, Second_Stage, Third_Stage
from models.networks import MLP
from data import TransInput, TransOutput, get_trans_input


class Model_Search(nn.Module):

    def __init__(self, args, trans_input_fn, loss_fn):
        super().__init__()
        self.args           = args
        self.nb_layers      = args.nb_layers
        self.cell_arch_topo = self.load_cell_arch()      # obtain architecture topology
        self.cell_arch_para = self.init_cell_arch_para() # register architecture topology parameters
        self.cells          = nn.ModuleList([Cell(args, self.cell_arch_topo[i]) for i in range(self.nb_layers)])
        self.loss_fn        = loss_fn
        self.trans_input_fn = trans_input_fn
        self.trans_input    = TransInput(trans_input_fn)
        self.trans_output   = TransOutput(args)
        if args.pos_encode > 0:
            self.position_encoding = nn.Linear(args.pos_encode, args.node_dim)


    def forward(self, input):
        arch_para_dict = self.group_arch_parameters()
        input = self.trans_input(input)
        G, V = input['G'], input['V']
        if self.args.pos_encode > 0:
            V = V + self.position_encoding(G.ndata['pos_enc'].float().cuda())
        output = {'G': G, 'V': V}
        for i, cell in enumerate(self.cells):
            output = cell(output, arch_para_dict[i])
        output = self.trans_output(output)
        return output


    def load_cell_arch(self):
        cell_arch_topo = []
        for _ in range(self.nb_layers):
            arch_topo = self.load_cell_arch_by_layer()
            cell_arch_topo.append(arch_topo)
        return cell_arch_topo


    def load_cell_arch_by_layer(self):
        arch_topo = []
        w = 0
        for dst in range(1, self.args.nb_nodes+1):
            for src in range(dst):
                arch_topo.append((src, dst, w, First_Stage))
                w += 1
        for dst in range(self.args.nb_nodes+1, 2*self.args.nb_nodes+1):
            src = dst - self.args.nb_nodes
            arch_topo.append((src, dst, w, Second_Stage))
            w += 1
        for dst in range(2*self.args.nb_nodes+1, 3*self.args.nb_nodes+1):
            for src in range(self.args.nb_nodes+1, 2*self.args.nb_nodes+1):
                arch_topo.append((src, dst, w, Third_Stage))
                w += 1
        return arch_topo 


    def init_cell_arch_para(self):
        cell_arch_para = []
        for i_layer in range(self.nb_layers):
            arch_para  = self.init_arch_para(self.cell_arch_topo[i_layer])
            cell_arch_para.extend(arch_para)
            self.nb_cell_topo = len(arch_para)
        return cell_arch_para


    def init_arch_para(self, arch_topo):
        arch_para = []
        for src, dst, w, ops in arch_topo:
            arch_para.append(Variable(1e-3 * torch.rand(len(ops)).cuda(), requires_grad = True))
        return arch_para


    def group_arch_parameters(self): 
        group = []
        start = 0
        for _ in range(self.nb_layers):
            group.append(self.arch_parameters()[start: start + self.nb_cell_topo])
            start += self.nb_cell_topo
        return group


    # def load_cell_arch_by_layer(self): 
    #     arch_type_dict = []
    #     w = 0
    #     for dst in range(1, self.args.nb_nodes + 1):
    #         for src in range(dst):
    #             arch_type_dict.append((src, dst, w))
    #             w += 1
    #     return arch_type_dict


    # def init_cell_arch_para(self):
    #     cell_arch_para = []
    #     for i_layer in range(self.nb_layers):
    #         cell_arch_para.append(self.init_arch_para(self.cell_arch_topo[i_layer]))
    #     return cell_arch_para


    # def init_arch_para(self, arch_topo):
    #     arch_para = Variable(1e-3 * torch.rand(len(arch_topo), len(OPS)).cuda(), requires_grad = True)
    #     return arch_para


    def new(self):
        model_new = Model_Search(self.args, get_trans_input(self.args), self.loss_fn).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    
    def load_alpha(self, alphas):
        for x, y in zip(self.arch_parameters(), alphas):
            x.data.copy_(y.data)

    def arch_parameters(self):
        return self.cell_arch_para

    def _loss(self, input, targets):
        scores = self.forward(input)
        return self.loss_fn(scores, targets)