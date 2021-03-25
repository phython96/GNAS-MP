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

from utils.utils import * 
from tqdm import tqdm
from torch.autograd import Variable
from models.model import Network
from data.data import LoadData
from configs.genotypes import *
import torch.backends.cudnn as cudnn

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
    cudnn.enabled = True
    cudnn.benchmark = True
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    dataset = LoadData(args.data_name)
    in_dim = dataset.train[0][0].ndata['feat'][0].size(0)
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    print(f"=> input dimension: {in_dim}, number classes: {num_classes}")
    
    if args.data_name == 'MNIST':
        genotype = MNIST_Net
    elif args.data_name == 'CIFAR10':
        genotype = CIFAR10_Net
    else:
        print("Unknown dataset.")
        exit()

    print('=> loading from genotype: \n', genotype)
    model = Network(args, genotype, num_classes, in_dim, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    train_data, val_data, test_data = dataset.train, dataset.val, dataset.test

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size = args.batch_size,
        pin_memory = True,
        num_workers=args.workers,
        collate_fn = dataset.collate,
        shuffle = True) #新增shuffle

    valid_queue = torch.utils.data.DataLoader(
        val_data, batch_size = args.batch_size,
        pin_memory = True,
        num_workers=args.workers,
        collate_fn = dataset.collate,
        shuffle = False)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=dataset.collate,
        shuffle = False)
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=3e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5,
                                                               patience=5,
                                                               verbose=True)
        
    for epoch in range(args.epochs):
        logging.info('[EPOCH]\t%d', epoch)
        if args.optimizer == 'SGD':
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('[LR]\t%f', lr)

        # training
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, stage = 'validating')
        # testing
        test_acc, test_obj = infer(test_queue, model, criterion, stage = 'testing   ')
        desc = '[train] acc: {:.3f}, loss: {:.3f}\t[validate] acc:{:.3f}, loss: {:.3f}\t[test] acc: {:.3f}, loss: {:.3f}'.format(
            train_acc, train_obj, valid_acc, valid_obj, test_acc, test_obj
        )
        logging.info(desc)

        if args.optimizer == 'ADAM':
            scheduler.step(valid_obj)
            if optimizer.param_groups[0]['lr'] < 1e-5:
                print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                break

def train(train_queue, model, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    desc = '=> training  '
    with tqdm(train_queue, desc=desc) as t:
        for step, (batch_graphs, batch_targets) in enumerate(t):
            start = time.time()
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            #batch_e = batch_graphs.edata['feat'].cuda()
            batch_targets = batch_targets.cuda()

            optimizer.zero_grad()
            batch_scores = model(batch_graphs, batch_x)
            loss = criterion(batch_scores, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_train_acc += accuracy_MNIST_CIFAR(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            t.set_postfix(time=time.time()-start, loss=epoch_loss/(step+1),
                          MICRO_ACC=epoch_train_acc / nb_data)
    epoch_loss /= (step + 1)
    epoch_train_acc /= nb_data

    return epoch_train_acc * 100, epoch_loss

def infer(valid_queue, model, criterion, stage):
    epoch_loss = 0
    epoch_valid_acc = 0
    nb_data = 0
    model.eval()
    desc = f'=> {stage}'
    with tqdm(valid_queue, desc=desc) as t:
        for step, (batch_graphs, batch_targets) in enumerate(t):
            start = time.time()
            n = batch_targets.size(0)
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            #batch_e = batch_graphs.edata['feat'].cuda()
            batch_targets = batch_targets.cuda()

            batch_scores = model(batch_graphs, batch_x)
            loss = criterion(batch_scores, batch_targets)

            epoch_loss += loss.detach().item()
            epoch_valid_acc += accuracy_MNIST_CIFAR(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            t.set_postfix(time=time.time()-start, loss=epoch_loss/(step+1),
                          MICRO_ACC=epoch_valid_acc / nb_data)
    epoch_loss /= (step + 1)
    epoch_valid_acc /= nb_data
    return epoch_valid_acc * 100, epoch_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser("GNAS")
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    # data
    parser.add_argument('--data_type', type=str, default='gc', help='data type')
    parser.add_argument('--data_name', type=str, default='CIFAR10', help='data name')
    # models
    parser.add_argument('--readout', type=str, default='mean', help='graph read out')
    parser.add_argument('--layers', type=int, default=2, help='total number of layers')
    parser.add_argument('--nodes', type=int, default=4, help='total number of nodes')
    parser.add_argument('--feature_dim', type=int, default=70, help='number of features')
    parser.add_argument('--op_norm', action='store_true', default=False)
    # train
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', type = str, default='SGD', help='optimizer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    # report and save
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')

    args = parser.parse_args()
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    start(args)