import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.cell_search import Cell
from models.operations import OPS
from models.networks import MLP
from data import TransInput, TransOutput, get_trans_input


class Model_Search(nn.Module):

    def __init__(self, args, trans_input_fn, loss_fn):
        super().__init__()
        self.args           = args
        self.nb_layers      = args.nb_layers
        self.cell_arch_topo = self.load_cell_arch()      # 获取结构拓扑
        self.cell_arch_para = self.init_cell_arch_para() # 注册结构参数
        self.cells          = nn.ModuleList([Cell(args, self.cell_arch_topo[i]) for i in range(self.nb_layers)])
        self.loss_fn        = loss_fn
        self.trans_input_fn = trans_input_fn
        self.trans_input    = TransInput(trans_input_fn)
        self.trans_output   = TransOutput(args)
        if args.pos_encode > 0:
            self.position_encoding = nn.Linear(args.pos_encode, args.node_dim)


    def forward(self, input):
        arch_para_dict = self.get_cell_arch_para()
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
            arch = self.load_cell_arch_by_layer()
            cell_arch_topo.append(arch)
        return cell_arch_topo


    def load_cell_arch_by_layer(self): 
        arch_type_dict = []
        w = 0
        for dst in range(1, self.args.nb_nodes + 1):
            for src in range(dst):
                arch_type_dict.append((src, dst, w))
                w += 1
        return arch_type_dict


    def init_cell_arch_para(self):
        #! 根据拓扑结构拓扑初始化结构参数
        cell_arch_para = []
        for i_layer in range(self.nb_layers):
            cell_arch_para.append(self.init_arch_para(self.cell_arch_topo[i_layer]))
        return cell_arch_para


    def init_arch_para(self, arch_topo):
        #! 初始化具体的结构参数
        arch_para = Variable(1e-3 * torch.rand(len(arch_topo), len(OPS)).cuda(), requires_grad = True)
        return arch_para


    def get_cell_arch_para(self):
        return self.cell_arch_para
    
    # def arch_para_dict(self):
    #     #! 将参数返回成字典的形式
    #     i_variable = 0
    #     arch_para_dict = []
    #     for _ in range(self.nb_layers):
    #         arch_para_dict.append({'V' : self.cell_arch_para[i_variable],
    #                                'E' : self.cell_arch_para[i_variable + 1]})
    #         i_variable += 2
    #     return arch_para_dict
    #     arch_para_dict = []
    #     for i in range(self.nb_layers):
    #         arch_para_dict.append(self.cell_arch_para[i])
    #     return arch_para_dict


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