import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

# configs of three-level search space # 
MIXED_OPS = {
    'f_zero': lambda args: f_zero_op(),
    'f_identity': lambda args: f_identity_op(),
    'f_dense': lambda args: f_dense_op(args),
    'f_sparse': lambda args: f_sparse_op(args),
    'a_max': lambda args: a_max_op(args),
    'a_mean': lambda args: a_mean_op(args),
    'a_sum': lambda args: a_sum_op(args),
    'a_std': lambda args: a_std_op(args),
}

FIRST_OPS = ['f_zero', 'f_identity', 'f_dense', 'f_sparse']
MIDDLE_OPS = ['f_identity', 'a_max', 'a_sum', 'a_mean']
LAST_OPS = ['f_zero', 'f_identity', 'f_dense', 'f_sparse']

# global variables & funcations # 
msg = fn.copy_src(src='h', out='m')
EPS = 1e-5

# identity feature filter # 
class f_identity_op(nn.Module):

    def __init__(self):
        super(f_identity_op, self).__init__()

    def forward(self, g, h, h_in):
        return h

# zero feature filter #
class f_zero_op(nn.Module):

    def __init__(self):
        super(f_zero_op, self).__init__()

    def forward(self, g, h, h_in):
        return 0 * h

# max - aggregator #
def reduce_max(nodes):
    accum = torch.max(nodes.mailbox['m'], 1)[0]
    return {'h' : accum}

class a_max_op(nn.Module):

    def __init__(self, args):
        super(a_max_op, self).__init__()
        feature_dim = args.get('feature_dim', 100)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, g, h, h_in):
        h = F.relu(self.linear(h))
        g.ndata['h'] = h
        g.update_all(msg, reduce_max)
        h = g.ndata['h']
        return h

# mean - aggregator # 
def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h' : accum}

class a_mean_op(nn.Module):

    def __init__(self, args):
        super(a_mean_op, self).__init__()
        feature_dim = args.get('feature_dim', 100)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, g, h, h_in):
        h = F.relu(self.linear(h))
        g.ndata['h'] = h
        g.update_all(msg, reduce_mean)
        h = g.ndata['h']
        return h

# sum - aggregator # 
def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}

class a_sum_op(nn.Module):

    def __init__(self, args):
        super(a_sum_op, self).__init__()

    def forward(self, g, h, h_in):
        g.ndata['h'] = h
        g.update_all(msg, reduce_sum)
        h = g.ndata['h']
        return h


# std - aggregator # 
def reduce_var(h):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var

def reduce_std(nodes):
    h = nodes.mailbox['m']
    return {'h' : torch.sqrt(reduce_var(h) + EPS)}

class a_std_op(nn.Module):

    def __init__(self, args):
        super(a_std_op, self).__init__()

    def forward(self, g, h, h_in):
        g.ndata['h'] = h
        g.update_all(msg, reduce_std)
        h = g.ndata['h']
        return h


# dense - feature filter #
class f_dense_op(nn.Module):
    def __init__(self, args):
        super(f_dense_op, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(2 * self._feature_dim,  self._feature_dim, bias = True)

    def forward(self, g, h, h_in):
        gates = torch.cat([h, h_in], dim = 1)
        gates = self.W(gates)
        return torch.sigmoid(gates) * h

# sparse - feature filter #
class f_sparse_op(nn.Module):
    def __init__(self, args):
        super(f_sparse_op, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(2 * self._feature_dim, self._feature_dim, bias = True)
        self.a = nn.Linear(self._feature_dim, 1, bias = False)

    def forward(self, g, h, h_in):
        gates = torch.cat([h, h_in], dim = 1)
        gates = self.W(gates)
        gates = self.a(gates)
        return torch.sigmoid(gates) * h

