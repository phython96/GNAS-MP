import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from configs.genotypes import Genotype
from models.operations import FIRST_OPS, MIDDLE_OPS, LAST_OPS
from models.cell import Cell

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L = nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class Network(nn.Module):

    def __init__(self, layers, nodes, in_dim, feature_dim, num_classes,
                 criterion, data_type = 'gc', readout = 'mean', dropout = 0.0):
        super(Network, self).__init__()
        self._layers = layers
        self._in_dim = in_dim
        self._feature_dim = feature_dim
        self._num_classes = num_classes
        self._criterion = criterion
        self._nb_first_nodes = nodes
        self._nb_last_nodes = nodes
        self._data_type = data_type
        self._readout = readout
        self._dropout = dropout

        self._nb_first_edges = sum(1 + i for i in range(self._nb_first_nodes))
        self._nb_middle_edges = self._nb_first_nodes
        self._nb_last_edges = sum(self._nb_first_nodes + i for i in range(self._nb_last_nodes))

        if data_type in ['nc', 'rg']:
            self.embedding_h = nn.Embedding(self._in_dim, self._feature_dim)  # node feat is an integer
        else:
            self.embedding_h = nn.Linear(self._in_dim, self._feature_dim)

        self.cells = nn.ModuleList([Cell(self._nb_first_nodes, self._nb_last_nodes, self._feature_dim, self._dropout)
                                    for i in range(self._layers)])
        self._initialize_alphas()
        outdim = self._feature_dim if self._data_type not in ['ec'] else 2 * self._feature_dim
        self.classifier = MLPReadout(outdim, self._num_classes)

    def new(self):
        model_new = Network(self._layers, self._nb_first_nodes, self._in_dim, self._feature_dim, self._num_classes, self. _criterion,
                            self._data_type, self._readout, self._dropout).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    
    def load_alpha(self, alphas):
        for x, y in zip(self.arch_parameters(), alphas):
            x.data.copy_(y.data)
            
    def arch_parameters(self):
        return self._arch_parameters

    def _initialize_alphas(self):
        nb_layers = self._layers
        nb_first_edges = self._nb_first_edges
        nb_middle_edges = self._nb_middle_edges
        nb_last_edges = self._nb_last_edges
        nb_first_ops = len(FIRST_OPS)
        nb_middle_ops = len(MIDDLE_OPS)
        nb_last_ops = len(LAST_OPS)

        self.alphas_first_cell = Variable(1e-3 * torch.randn(nb_first_edges * nb_layers, nb_first_ops).cuda(),
                                    requires_grad=True)
        self.alphas_middle_cell = Variable(1e-3 * torch.randn(nb_middle_edges * nb_layers, nb_middle_ops).cuda(),
                                    requires_grad=True)
        self.alphas_last_cell = Variable(1e-3 * torch.randn(nb_last_edges * nb_layers, nb_last_ops).cuda(),
                                    requires_grad=True)
        self._arch_parameters = [
            self.alphas_first_cell,
            self.alphas_middle_cell,
            self.alphas_last_cell,
        ]

    def _forward(self, g, h):
        h = self.embedding_h(h)
        for i, cell in enumerate(self.cells):
            W_first, W_middle, W_last = self.show_weights(i)
            h = cell(g, h, W_first, W_middle, W_last)
        return h

    def forward(self, g, h):
        h = self._forward(g, h)
        if self._data_type in ['nc']:
            h = self.classifier(h)
            return h
        elif self._data_type in ['gc', 'rg']:
            g.ndata['h'] = h
            if self._readout == "sum":
                hg = dgl.sum_nodes(g, 'h')
            elif self._readout == "max":
                hg = dgl.max_nodes(g, 'h')
            elif self._readout == "mean":
                hg = dgl.mean_nodes(g, 'h')
            else:
                hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            hg = self.classifier(hg)
            return hg
        elif self._data_type in ['ec']:
            def _edge_feat(edges):
                e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
                e = self.classifier(e)
                return {'e': e}
            g.ndata['h'] = h
            g.apply_edges(_edge_feat)
            return g.edata['e']

    def _loss(self, g, h, batch_targets):
        batch_scores = self.forward(g, h)
        return self._criterion(batch_scores, batch_targets)

    def normalize_weights(self, W_first, W_middle, W_last):
        W_first  = F.softmax(W_first, dim = 1)
        W_middle = F.softmax(W_middle, dim = 1)
        W_last   = F.softmax(W_last, dim = 1)
        return W_first, W_middle, W_last

    def show_weights(self, nb_layer):
        nb_first_edges = self._nb_first_edges
        nb_middle_edges = self._nb_middle_edges
        nb_last_edges = self._nb_last_edges
        return self.normalize_weights(self.alphas_first_cell[nb_layer * nb_first_edges   : (nb_layer+1) * nb_first_edges],
                                      self.alphas_middle_cell[nb_layer * nb_middle_edges : (nb_layer+1) * nb_middle_edges],
                                      self.alphas_last_cell[nb_layer * nb_last_edges     : (nb_layer+1) * nb_last_edges])

    def show_genotype(self, nb_layer):
        outdegree = {}
        gene = []
        nb_first_nodes = self._nb_first_nodes
        nb_last_nodes  = self._nb_last_nodes
        W_first, W_middle, W_last = self.show_weights(nb_layer)
        # edges in first cell
        start = 0
        for n in range(1, nb_first_nodes + 1):
            end = start + n
            W = W_first[start : end]
            j = sorted(range(n), key = lambda x : -max(W[x][k] for k in range(len(FIRST_OPS))
                                                       if k != FIRST_OPS.index('f_zero')))[0]
            k_best = None
            for k in range(len(FIRST_OPS)):
                if k == FIRST_OPS.index('f_zero'): continue
                if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
            gene.append((FIRST_OPS[k_best], n, j))
            outdegree[j] = outdegree.get(j, 0) + 1
            start = end
        # edges in middle cell
        middle_nodes = list(range(1, 1 + nb_first_nodes))
        for n in range(0, nb_first_nodes):
            k = torch.argmax(W_middle[n]).cpu().item()
            if k != MIDDLE_OPS.index('f_identity'):
                new_node = max(middle_nodes) + 1
                pre_node = middle_nodes[n]
                gene.append((MIDDLE_OPS[k], new_node, pre_node))
                outdegree[pre_node] = outdegree.get(pre_node, 0) + 1
                middle_nodes[n] = new_node

        #edges in last cell
        start = 0
        for n in range(nb_last_nodes):
            node_id = n + max(middle_nodes) + 1
            end = start + nb_first_nodes + n
            W = W_last[start : end]
            j = sorted(range(nb_first_nodes + n), key=lambda x: -max(W[x][k] for k in range(len(FIRST_OPS))
                                                    if k != FIRST_OPS.index('f_zero')))[0]
            k_best = None
            for k in range(len(FIRST_OPS)):
                if k == FIRST_OPS.index('f_zero'): continue
                if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
            pre_node_id = middle_nodes[j] if j < nb_first_nodes else j - nb_first_nodes + max(middle_nodes) + 1
            gene.append((FIRST_OPS[k_best], node_id, pre_node_id))
            outdegree[pre_node_id] = outdegree.get(pre_node_id, 0) + 1
            start = end

        # big = max(middle_nodes) +  nb_last_nodes
        _genotype = Genotype(alpha_cell = gene,
                             #concat_node = [i for i in range(1, big + 1) if outdegree.get(i, 0) == 0])
                             concat_node=None)
        return _genotype

    def show_genotypes(self):
        return [self.show_genotype(i) for i in range(self._layers)]

