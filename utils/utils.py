import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from models.operations import V_OPS, E_OPS, get_OPS

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def load_alpha(genotype):
    alpha_cell = torch.Tensor(genotype.alpha_cell)
    alpha_edge = torch.Tensor(genotype.alpha_edge)
    return [alpha_cell, alpha_edge]

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask      = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

# ----------------------------------------------------------------
# metrics
class mask: 

    def __init__(self, data_type, mask):

        self.data_type = data_type
        self.mask      = mask

    def __call__(self, func):
        
        def wrapped_func(*args, **kwargs):
            
            if self.mask:
                args  = list(args)
                graph = kwargs['graph']
                stage = kwargs['stage']

                graph_data = graph.ndata if self.data_type == 'V' else graph.edata

                if f'{stage}_mask' in graph_data:
                    mask = graph_data[f'{stage}_mask']
                    args[-1] = args[-1][mask]
                    args[-2] = args[-2][mask]
            return func(*args)

        return wrapped_func

def accuracy(output, target, topk=(1,)):
    maxk       = max(topk)
    batch_size = target.size(0)

    _, pred    = output.topk(maxk, 1, True, True)
    pred       = pred.t()
    correct    = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

@mask('E', False)
def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels.
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')

@mask('V', False)
def accuracy_SBM(scores, targets):

    S                    = targets.cpu().numpy()
    C                    = np.argmax(torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy(), axis=1)
    CM                   = confusion_matrix(S, C).astype(np.float32)
    nb_classes           = CM.shape[0]
    targets              = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes           = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100. * np.sum(pr_classes) / float(nb_classes)
    return acc

@mask('G', False)
def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc    = (scores == targets).float().mean().item()
    return acc

@mask('G', False)
def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE

@mask('V', True)
def CoraAccuracy(scores, targets):
    return (scores.argmax(1) == targets).float().mean().item()

# ----------------------------------------------------------------
# loss functions

class MoleculesCriterion(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()
    
    @mask('G', False)
    def forward(self, pred, label):
        return self.loss_fn(pred, label)

class TSPCriterion(nn.Module):

    def __init__(self): 
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=None)
    
    @mask('E', False)
    def forward(self, pred, label):
        return self.loss_fn(pred, label)

class SuperPixCriterion(nn.Module): 

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=None)
    
    @mask('G', False)
    def forward(self, pred, label):
        return self.loss_fn(pred, label)

class SBMsCriterion(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.n_classes = num_classes

    @mask('V', False)
    def forward(self, pred, label):
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().cuda()
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)
        return loss

class CiteCriterion(nn.Module): 

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=None)
    
    @mask('V', True)
    def forward(self, pred, label):
        return self.loss_fn(pred, label)


# ----------------------------------------------------------------
from collections import namedtuple
Genotype = namedtuple('Genotype', 'V E')

def genotype_type(args, arch_para_type, arch_topo_type, type):
    result         = []
    OPS            = get_OPS(type)
    arch_para_type = arch_para_type.softmax(dim = 1)
    link           = [ [] for i in range(args.nb_nodes + 1) ]
    for Si, Vj, Ek, L in arch_topo_type:
        link[Si].append((Vj, Ek, arch_para_type[L]))
    for i_node in range(1, args.nb_nodes + 1):
        nb_link   = len(link[i_node])
        best_links = sorted(
            range(nb_link), 
            key = lambda lk : -max(link[i_node][lk][-1][j] for j in range(len(OPS)) if j != OPS.index(f'{type}_None'))
        )[:2] #! 截取的操作数量
        for blink in best_links:
            blink   = link[i_node][blink]
            best_op = torch.argmax(blink[-1][1:]).item() + 1
            result.append((i_node, blink[0], blink[1], OPS[best_op]))
    return result
    

def genotype(args, arch_para, arch_topo):
    #! 根据结构参数离散化得到基因型
    genotype = Genotype(V = genotype_type(args, arch_para['V'], arch_topo['V'], 'V'),
                        E = genotype_type(args, arch_para['E'], arch_topo['E'], 'E'))
    return genotype

def genotypes(args, cell_arch_para, cell_arch_topo):
    genotypes = []
    for layer_para, layer_topo in zip(cell_arch_para, cell_arch_topo):
        genotypes.append(genotype(args, layer_para, layer_topo))
    return genotypes

# ----------------------------------------------------------------
import math
def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton

@Singleton
class DecayScheduler(object):
    def __init__(self, base_lr=1.0, last_iter=-1, T_max=50, T_start=0, T_stop=50, decay_type='cosine'):
        self.base_lr    = base_lr
        self.T_max      = T_max
        self.T_start    = T_start
        self.T_stop     = T_stop
        self.cnt        = 0
        self.decay_type = decay_type
        self.decay_rate = 1.0

    def step(self, epoch):
        if epoch >= self.T_start:
          if self.decay_type == "cosine":
              self.decay_rate = self.base_lr * (1 + math.cos(math.pi * epoch / (self.T_max - self.T_start))) / 2.0 if epoch <= self.T_stop else self.decay_rate
          elif self.decay_type == "slow_cosine":
              self.decay_rate = self.base_lr * math.cos((math.pi/2) * epoch / (self.T_max - self.T_start)) if epoch <= self.T_stop else self.decay_rate
          elif self.decay_type == "linear":
              self.decay_rate = self.base_lr * (self.T_max - epoch) / (self.T_max - self.T_start) if epoch <= self.T_stop else self.decay_rate
          else:
              self.decay_rate = self.base_lr
        else:
            self.decay_rate = self.base_lr


def annouce(content, color = None):
    if type(content) != str:
        print(content)
    else:
        if color:
            bg = f'\033[1;{color};40m'
            ed = '\033[0m'
        else:
            bg, ed = '', ''
        print(bg + content + ed)
