import sys
sys.path.append("/home/shaofei_cai/open-source/GNAS")

import os
import dgl
import time
import torch
import pickle
import logging
import argparse
import numpy as np
import torch.nn as nn
import utils.utils as utils
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
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    
    dataset = LoadData(args.data_name)
    in_dim = dataset.num_atom_type
    num_classes = 1
    criterion = nn.L1Loss()
    criterion = criterion.cuda()
    print(f"=> input dimension: {in_dim}, number classes: {num_classes}")
    
    genotype = ZINC_Net
    print('=> loading from genotype: \n', genotype)
    model = Network(genotype, args.layers, in_dim, args.feature_dim, num_classes, criterion, args.data_type, args.readout, args.dropout)
    model = model.cuda()
    #logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("=> param size = %f", utils.count_parameters_in_MB(model) * 1e6)

    train_data, val_data, test_data = dataset.train, dataset.val, dataset.test

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size = args.batch_size,
        pin_memory = True,
        num_workers=args.workers,
        collate_fn = dataset.collate,
        shuffle = True)

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
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5,
                                                     patience=10,
                                                     verbose=True)
    
    for epoch in range(args.epochs):
        logging.info('[EPOCH]\t%d', epoch)
        if args.optimizer == 'SGD':
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('[LR]\t%f', lr)

        #genotype = model.genotype()
        #logging.info('genotype = %s', genotype)

        # training
        train_mae, train_obj = train(train_queue, model, criterion, optimizer)

        # validation
        valid_mae, valid_obj = infer(valid_queue, model, criterion, stage = 'validating')

        # testing
        test_mae,  test_obj = infer(test_queue, model, criterion, stage = 'testing   ')
        desc = '[train] mae: {:.3f}, loss: {:.3f}\t[validate] mae:{:.3f}, loss: {:.3f}\t[test] mae: {:.3f}, loss: {:.3f}'.format(
            train_mae, train_obj, valid_mae, valid_obj, test_mae, test_obj
        )
        logging.info(desc)
        
        if args.optimizer == 'ADAM':
            scheduler.step(valid_obj)
            if optimizer.param_groups[0]['lr'] < 1e-5:
                print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                break
        
        utils.save(model, os.path.join(args.save, 'weights.pt'))

def train(train_queue, model, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    desc = '=> training  '
    with tqdm(train_queue, desc=desc) as t:
        for step, (batch_graphs, batch_targets) in enumerate(t):
            start = time.time()
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            batch_e = batch_graphs.edata['feat'].cuda()
            batch_targets = batch_targets.cuda()

            optimizer.zero_grad()
            batch_scores = model(batch_graphs, batch_x)
            loss = criterion(batch_scores, batch_targets)
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_train_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            #if step % args.report_freq == 0 and step > 0:
            #    logging.info('[training] 03d\tloss: %f\tmae: %f', step, epoch_loss / (step + 1), epoch_train_mae / (step + 1))
                #print(F.softmax(model.alphas_cell, dim=1))
            t.set_postfix(time=time.time()-start, loss=epoch_loss/(step+1),
                          MAE=epoch_train_mae/(step+1))

    epoch_loss /= (step + 1)
    epoch_train_mae /= (step + 1)
    return epoch_train_mae, epoch_loss

def infer(valid_queue, model, criterion, stage):
    epoch_loss = 0
    epoch_mae = 0
    nb_data = 0
    model.eval()
    desc = f'=> {stage}'
    with tqdm(valid_queue, desc=desc) as t:
        for step, (batch_graphs, batch_targets) in enumerate(t):
            start = time.time()
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            batch_e = batch_graphs.edata['feat'].cuda()
            batch_targets = batch_targets.cuda()

            batch_scores = model(batch_graphs, batch_x)
            loss = criterion(batch_scores, batch_targets)
            epoch_loss += loss.detach().item()
            epoch_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            #if step % args.report_freq == 0 and step > 0:
            #    logging.info('[%s] %03d\tloss: %f\tmae: %f', stage, step, epoch_loss / (step + 1), epoch_mae / (step + 1))
            t.set_postfix(time=time.time()-start, loss=epoch_loss/(step+1),
                          MAE=epoch_mae/(step+1))

    epoch_loss /= (step + 1)
    epoch_mae /= (step + 1)
    
    return epoch_mae, epoch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser("GNAS")
    parser.add_argument('--readout', type=str, default='mean', help='graph read out')
    parser.add_argument('--data_type', type=str, default='rg', help='data type')
    parser.add_argument('--data_name', type=str, default='ZINC', help='data name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--layers', type=int, default=2, help='total number of layers')
    parser.add_argument('--feature_dim', type=int, default=70, help='number of features')
    parser.add_argument('--nodes', type=int, default=3, help='total number of nodes')
    parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', type = str, default='SGD', help='optimizer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    args = parser.parse_args()
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    start(args)