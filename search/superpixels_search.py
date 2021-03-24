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

def start(args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    dataset = LoadData(args.data_name)
    in_dim = dataset.train[0][0].ndata['feat'][0].size(0)
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    print(in_dim, num_classes)

    model = Network(args.layers, args.nodes, in_dim, args.feature_dim, num_classes, criterion, args.data_type, args.readout, args.dropout)
    model = model.cuda()
    logging.info("param size = %fMB", count_parameters_in_MB(model))
    
    train_data, val_data, test_data = dataset.train, dataset.val, dataset.test
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    print(f"[dataset] num classes : {num_train}, split : {split}")

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

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    # viz = Visdom(env = '{} {}'.format(args.data_name,  time.asctime(time.localtime(time.time()))  ))
    viz = None
    save_file = open(args.save_result, "w")
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('[epoch] %d\tlr: %e', epoch, lr)

        #genotype = model.show_genotype(0)
        #logging.info('genotype = %s', genotype)
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
        train_acc, train_loss = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, viz)
        # true validation
        true_valid_acc, true_valid_loss = infer(true_valid_queue, model, criterion, stage = 'validating')
        # testing
        test_acc, test_loss = infer(test_queue, model, criterion, stage = ' testing  ')


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, viz = None):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    nb_data = 0
    desc = '=> searching'
    with tqdm(train_queue, desc=desc) as t:
        alpha = 0
        architect.loss = 0
        for step, (batch_graphs, batch_targets) in enumerate(t):
            start = time.time()
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            #batch_e = batch_graphs.edata['feat'].cuda()
            batch_targets = batch_targets.cuda()

            # get a random minibatch from the search queue with replacement
            batch_graphs_search, batch_targets_search = next(iter(valid_queue))
            batch_x_search = batch_graphs_search.ndata['feat'].cuda()  # num x feat
            #batch_e_search = batch_graphs_search.edata['feat'].cuda()
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

            epoch_train_acc += accuracy_MNIST_CIFAR(batch_scores, batch_targets)
            epoch_loss += loss.detach().item()
            nb_data += batch_targets.size(0)

            t.set_postfix(time=time.time()-start, loss=epoch_loss/(step+1),
                          MICRO_ACC=epoch_train_acc / nb_data)

    epoch_loss /= (step + 1)
    epoch_train_acc /=  nb_data
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

        viz.line(np.array([[architect.loss.item() / (step + 1)], [epoch_loss]]),
                 np.array([epoch + 1]),
                 win='loss', update='append', opts=dict(legend=["alpha_loss", "weight_loss"]))
    '''

    return epoch_train_acc, epoch_loss

def infer(valid_queue, model, criterion, stage = 'valid'):
    model.eval()
    epoch_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    desc = f'=> {stage}'
    with tqdm(valid_queue, desc=desc) as t:
        for step, (batch_graphs, batch_targets) in enumerate(t):
            start = time.time()
            n = batch_targets.size(0)
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            # batch_e = batch_graphs.edata['feat'].cuda()
            batch_targets = batch_targets.cuda()

            batch_scores = model(batch_graphs, batch_x)
            loss = criterion(batch_scores, batch_targets)

            epoch_test_acc += accuracy_MNIST_CIFAR(batch_scores, batch_targets)
            epoch_loss += loss.detach().item()
            nb_data += batch_targets.size(0)
            t.set_postfix(time=time.time()-start, loss=epoch_loss/(step+1),
                          MICRO_ACC=epoch_test_acc / nb_data)

    epoch_test_acc /= nb_data
    epoch_loss /= (step + 1)
    return epoch_test_acc, epoch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser("GNAS")
    parser.add_argument('--readout', type=str, default='mean', help='graph read out')
    parser.add_argument('--data_type', type=str, default='gc', help='data type')
    parser.add_argument('--data_name', type=str, default='CIFAR10', help='data name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--layers', type=int, default=4, help='total number of layers')
    parser.add_argument('--feature_dim', type=int, default=90, help='number of features')
    parser.add_argument('--nodes', type=int, default=3, help='total number of nodes')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--save_result', type=str, default="./save/superpixels_search.txt")
    args = parser.parse_args()
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    start(args)