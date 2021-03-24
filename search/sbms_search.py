import sys
import os
import dgl
import time
import torch
import pickle
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.model_search import Network

from utils.utils import *
from torch.autograd import Variable
from data.data import LoadData
from models.architect import Architect
from tqdm import tqdm

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

class MyCriterion(nn.Module):
    def __init__(self, num_classes):
        super(MyCriterion, self).__init__()
        self.n_classes = num_classes
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

def start(args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("args = %s", args)

    dataset = LoadData(args.data_name)
    if args.data_name == 'SBM_PATTERN':
        in_dim = 3
        num_classes = 2
    elif args.data_name == 'SBM_CLUSTER':
        in_dim = 7
        num_classes = 6
    print(f"input dimension: {in_dim}, number classes: {num_classes}")

    criterion = MyCriterion(num_classes)
    criterion = criterion.cuda()

    model = Network(args.layers, args.nodes, in_dim, args.feature_dim, num_classes, criterion, args.data_type, args.readout)
    model = model.cuda()
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    train_data, val_data, test_data = dataset.train, dataset.val, dataset.test

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    print(f"train set full size : {num_train}; split train set size : {split}")
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size = args.batch_size,
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory = True,
        num_workers=args.workers,
        collate_fn = dataset.collate)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size = args.batch_size,
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory = True,
        num_workers=args.workers,
        collate_fn = dataset.collate)

    true_valid_queue = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=dataset.collate)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=dataset.collate)

    optimizer = torch.optim.SGD(model.parameters(),args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    architect = Architect(model, args)

    # viz = Visdom(env = '{} {}'.format(args.data_name,  time.asctime(time.localtime(time.time()))  ))
    viz = None
    save_file = open(args.save_result, "w")
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('[LR]\t%f', lr)

        if epoch % args.save_freq == 0:
            print(model.show_genotypes())
            save_file.write(f"Epoch : {epoch}\n{model.show_genotypes()}\n")
            for i in range(args.layers):
                logging.info('layer = %d', i)
                genotype = model.show_genotype(i)
                logging.info('genotype = %s', genotype)
            '''
            w1, w2, w3 = model.show_weights(0)
            print('[1] weights in first cell\n',w1)
            print('[2] weights in middle cell\n', w2)
            print('[3] weights in last cell\n', w3)
            '''
        # training
        macro_acc, micro_acc, loss = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, viz)
        # true validation
        macro_acc, micro_acc, loss = infer(true_valid_queue, model, criterion, stage = 'validating')
        # testing
        macro_acc, micro_acc, loss = infer(test_queue, model, criterion, stage = ' testing  ')

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, viz = None):
    model.train()
    top1 = AvgrageMeter()
    epoch_loss = 0
    macro_acc = 0
    desc = '=> searching'

    with tqdm(train_queue, desc=desc) as t:
        alpha = 0
        architect.loss = 0
        for step, (batch_graphs, batch_targets) in enumerate(t):
            start = time.time()
            n = batch_targets.size(0)
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            batch_targets = batch_targets.cuda()

            # get a random minibatch from the search queue with replacement
            batch_graphs_search, batch_targets_search = next(iter(valid_queue))
            batch_x_search = batch_graphs_search.ndata['feat'].cuda()  # num x feat
            batch_targets_search = batch_targets_search.cuda()

            architect.step(batch_graphs, batch_x, batch_targets,
                           batch_graphs_search, batch_x_search, batch_targets_search,
                           lr, optimizer, unrolled=args.unrolled)

            alpha += model._arch_parameters[1].softmax(dim=1)

            optimizer.zero_grad()
            batch_scores = model(batch_graphs, batch_x)
            loss = criterion(batch_scores, batch_targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            macro_acc += accuracy_SBM(batch_scores, batch_targets)
            prec1 = accuracy(batch_scores, batch_targets, topk=(1, ))[0]
            top1.update(prec1.item(), n)
            t.set_postfix(time=time.time()-start, loss=epoch_loss/(step+1),
                          MACRO_ACC=macro_acc/(step+1), MICRO_ACC=top1.avg)
    
    epoch_loss /= (step + 1)
    macro_acc /= (step + 1)
    micro_acc  = top1.avg
    '''
    chosen = torch.argmax(alpha, dim=1)
    ans = [0] * 4
    for x in chosen:
        ans[x] += 1
    if viz is not None:
        viz.line(np.array([[ans[i]] for i in range(4)]),
                 np.array([epoch + 1]),
                 win='operations', update='append', opts=dict(legend=["Identity", "Max", "Sum", "Mean"]))

        viz.line(np.array([[alpha.mean(dim=0)[i].item() / (step + 1)] for i in range(4)]),
                 np.array([epoch + 1]),
                 win='alphas', update='append', opts=dict(legend=["Identity", "Max", "Sum", "Mean"]))

        viz.line(np.array([[grad[i].item()] for i in range(4)] + [[grad_abs[i].item()] for i in range(4)]),
                 np.array([epoch + 1]),
                 win='grad', update='append', opts=dict(legend=["Identity_grad", "Max_grad", "Sum_grad", "Mean_grad"]
                                                               + ["Identity_grad_abs", "Max_grad_abs", "Sum_grad_abs",
                                                                  "Mean_grad_abs"]))

        viz.line(np.array([[architect.loss.item() / (step + 1)], [epoch_loss]]),
                 np.array([epoch + 1]),
                 win='loss', update='append', opts=dict(legend=["alpha_loss", "weight_loss"]))
    '''
    return macro_acc, micro_acc, epoch_loss

def infer(valid_queue, model, criterion, stage):
    model.eval()
    top1 = AvgrageMeter()
    epoch_loss = 0
    macro_acc = 0

    desc = f'=> {stage}'
    with tqdm(valid_queue, desc=desc) as t:
        for step, (batch_graphs, batch_targets) in enumerate(t):
            start = time.time()
            n = batch_targets.size(0)
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            batch_targets = batch_targets.cuda()

            batch_scores = model(batch_graphs, batch_x)
            loss = criterion(batch_scores, batch_targets)
            prec1  = accuracy(batch_scores, batch_targets, topk=(1, ))[0]
            top1.update(prec1.item(), n)
            epoch_loss += loss.detach().item()
            macro_acc += accuracy_SBM(batch_scores, batch_targets)
            t.set_postfix(time=time.time()-start, loss=epoch_loss/(step+1),
                          MACRO_ACC=macro_acc/(step+1), MICRO_ACC=top1.avg)

    epoch_loss /= (step + 1)
    macro_acc /= (step + 1)
    micro_acc  = top1.avg

    return macro_acc, micro_acc, epoch_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser("GNAS")
    parser.add_argument('--readout', type=str, default='mean', help='graph read out')
    parser.add_argument('--data_type', type=str, default='nc', help='data type')
    parser.add_argument('--data_name', type=str, default='SBM_PATTERN', help='data name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--layers', type=int, default=4, help='total number of layers')
    parser.add_argument('--feature_dim', type=int, default=96, help='number of features')
    parser.add_argument('--nodes', type=int, default=3, help='total number of nodes')
    parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--save_result', type=str, default="./save/SBMs_search.txt")

    args = parser.parse_args()
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    start(args)