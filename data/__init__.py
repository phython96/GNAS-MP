import dgl
import torch
import torch.nn as nn
from data.molecules import MoleculeDataset
from data.QM9 import QM9Dataset
from data.SBMs import SBMsDataset
from data.TSP import TSPDataset
from data.superpixels import SuperPixDataset
from data.cora import CoraDataset
from models.networks import *
from utils.utils import *


class TransInput(nn.Module):

    def __init__(self, trans_fn):
        super().__init__()
        self.trans_V = trans_fn[0]
        self.trans_E = trans_fn[1]
    
    def forward(self, input):
        G, V, E = input
        if self.trans_V:
            V = self.trans_V(V)
        if self.trans_E:
            E = self.trans_E(E)
        return (G, V, E)


class TransOutput(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.task == 'node_level': 
            channel_sequence = (args.node_dim, ) * args.nb_mlp_layer + (args.nb_classes, )
            self.trans = MLP(channel_sequence)
        elif args.task == 'link_level':
            if self.args.edge_feature:
                channel_sequence = (args.edge_dim, ) * args.nb_mlp_layer + (args.nb_classes, )
            else:
                channel_sequence = (args.node_dim * 2, ) * args.nb_mlp_layer + (args.nb_classes, )
            self.trans = MLP(channel_sequence)
        elif args.task == 'graph_level':
            if args.edge_feature:
                channel_sequence = (args.node_dim + args.edge_dim, ) * args.nb_mlp_layer + (args.nb_classes, )
            else:
                channel_sequence = (args.node_dim, ) * args.nb_mlp_layer + (args.nb_classes, )
            self.trans = MLP(channel_sequence)
        else:
            raise Exception('Unknown task!')
            

    def forward(self, input):
        G, V, E = input
        if self.args.task == 'node_level':
            output = self.trans(V)
        elif self.args.task == 'link_level':
            if self.args.edge_feature:
                output = self.trans(E)
            else: 
                def _edge_feat(edges):
                    e = torch.cat([edges.src['V'], edges.dst['V']], dim=1)
                    return {'e': e}
                G.ndata['V'] = V
                G.apply_edges(_edge_feat)
                output = self.trans(G.edata['e'])
        elif self.args.task == 'graph_level':
            G.ndata['V'] = V
            G.edata['E'] = E
            if self.args.edge_feature:
                readout = torch.cat([dgl.mean_nodes(G, 'V'), dgl.mean_edges(G, 'E')], dim = -1)
            else:
                readout = dgl.mean_nodes(G, 'V')
            output = self.trans(readout)
        else:
            raise Exception('Unknown task!')
        return output


def get_trans_input(args):
    if args.data in ['ZINC']:
        trans_input_V = nn.Embedding(args.in_dim_V, args.node_dim) 
        trans_input_E = nn.Embedding(args.in_dim_E, args.edge_dim)
    elif args.data in ['TSP']:
        trans_input_V = nn.Linear(args.in_dim_V, args.node_dim)
        trans_input_E = nn.Linear(args.in_dim_E, args.edge_dim)
    elif args.data in ['SBM_CLUSTER', 'SBM_PATTERN']:
        trans_input_V = nn.Embedding(args.in_dim_V, args.node_dim)
        trans_input_E = nn.Linear(args.in_dim_E, args.edge_dim)
    elif args.data in ['CIFAR10', 'MNIST', 'Cora']:
        trans_input_V = nn.Linear(args.in_dim_V, args.node_dim)
        trans_input_E = nn.Linear(args.in_dim_E, args.edge_dim)
    elif args.data in ['QM9']:
        trans_input_V = nn.Linear(args.in_dim_V, args.node_dim)
        trans_input_E = nn.Linear(args.in_dim_E, args.edge_dim)
    else:
        raise Exception('Unknown dataset!')
    return (trans_input_V, trans_input_E)


def get_loss_fn(args):
    if args.data in ['ZINC', 'QM9']:
        loss_fn = MoleculesCriterion()
    elif args.data in ['TSP']:
        loss_fn = TSPCriterion()
    elif args.data in ['SBM_CLUSTER', 'SBM_PATTERN']:
        loss_fn = SBMsCriterion(args.nb_classes)
    elif args.data in ['CIFAR10', 'MNIST']:
        loss_fn = SuperPixCriterion()
    elif args.data in ['Cora']:
        loss_fn = CiteCriterion()
    else:
        raise Exception('Unknown dataset!')
    return loss_fn


def load_data(args):
    if args.data in ['ZINC']:
        return MoleculeDataset(args.data)
    elif args.data in ['QM9']:
        return QM9Dataset(args.data, args.extra)
    elif args.data in ['TSP']:
        return TSPDataset(args.data)
    elif args.data in ['MNIST', 'CIFAR10']:
        return SuperPixDataset(args.data) 
    elif args.data in ['SBM_CLUSTER', 'SBM_PATTERN']: 
        return SBMsDataset(args.data)
    elif args.data in ['Cora']:
        return CoraDataset(args.data)
    else:
        raise Exception('Unknown dataset!')


def load_metric(args):
    if args.data in ['ZINC', 'QM9']:
        return MAE
    elif args.data in ['TSP']:
        return binary_f1_score
    elif args.data in ['MNIST', 'CIFAR10']:
        return accuracy_MNIST_CIFAR
    elif args.data in ['SBM_CLUSTER', 'SBM_PATTERN']:
        return accuracy_SBM
    elif args.data in ['Cora']:
        return CoraAccuracy
    else:
        raise Exception('Unknown dataset!')
