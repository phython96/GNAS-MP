import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


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
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
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


def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels.

    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')


def accuracy_SBM(scores, targets):

    S = targets.cpu().numpy()
    C = np.argmax(torch.nn.Softmax(dim=1)(
        scores).cpu().detach().numpy(), axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
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


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    return acc


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE
