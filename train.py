import os
import sys
import dgl
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from data import *
from models.model_train import *
from utils.utils import *
from tensorboardX import SummaryWriter
from utils.record_utils import record_run

class Trainer(object):

    def __init__(self, args):
        
        self.args = args

        annouce('=> [0] Initial TensorboardX')
        self.writer = SummaryWriter(comment = f'Task: {args.task}, Data: {args.data}, Geno: {args.load_genotypes}')

        annouce('=> [1] Initial Settings')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled   = True

        annouce('=> [2] Initial Models')
        if not os.path.isfile(args.load_genotypes):
            raise Exception('Genotype file not found!')
        else:
            with open(args.load_genotypes) as f:
                genotypes      = eval(f.read())
                args.nb_layers = len(genotypes)
                args.nb_nodes  = len({ x for x, a, b, c in genotypes[0].V})
        self.metric    = load_metric(args)
        self.loss_fn   = get_loss_fn(args).cuda()
        trans_input_fn = get_trans_input(args)
        self.model     = Model_Train(args, genotypes, trans_input_fn, self.loss_fn).to("cuda")
        annouce(f'=> Subnet Parameters: {count_parameters_in_MB(self.model)}', 33)

        annouce(f'=> [3] Preparing Dataset')
        self.dataset    = load_data(args)
        if args.pos_encode > 0:
            #! 加载 - 位置编码
            annouce(f'==> [3.1] Adding positional encodings')
            self.dataset._add_positional_encodings(args.pos_encode)
        self.train_data = self.dataset.train
        self.val_data   = self.dataset.val
        self.test_data  = self.dataset.test
        self.load_dataloader()

        annouce(f'=> [4] Initial Optimizers')
        if args.optimizer == 'SGD':
            self.optimizer   = torch.optim.SGD(
                params       = self.model.parameters(),
                lr           = args.lr,
                momentum     = args.momentum,
                weight_decay = args.weight_decay,
            )
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
                optimizer  = self.optimizer,
                T_max      = float(args.epochs),
                eta_min    = args.lr_min
            )

        elif args.optimizer == 'ADAM':
            self.optimizer   = torch.optim.Adam(
                params       = self.model.parameters(),
                lr           = args.lr,
                weight_decay = args.weight_decay,
            )

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = self.optimizer,
                mode      = 'min',
                factor    = 0.5,
                patience  = args.patience,
                verbose   = True
            )
        else:
            raise Exception('Unknown optimizer!')


    def load_dataloader(self):
        
        num_train = int(len(self.train_data) * self.args.data_clip)
        indices   = list(range(num_train))

        self.train_queue = torch.utils.data.DataLoader(
            dataset     = self.train_data,
            batch_size  = self.args.batch,
            pin_memory  = True,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate,
        )

        num_valid = int(len(self.val_data) * self.args.data_clip)
        indices   = list(range(num_valid))

        self.val_queue  = torch.utils.data.DataLoader(
            dataset     = self.val_data,
            batch_size  = self.args.batch,
            pin_memory  = True,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate,
            shuffle     = False
        )

        num_test  = int(len(self.test_data) * self.args.data_clip)
        indices   = list(range(num_test))

        self.test_queue = torch.utils.data.DataLoader(
            dataset     = self.test_data,
            batch_size  = self.args.batch,
            pin_memory  = True,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate,
            shuffle     = False,
        )
    

    def scheduler_step(self, valid_loss):

        if self.args.optimizer == 'SGD':
            self.scheduler.step()
            lr = scheduler.get_lr()[0]
        elif self.args.optimizer == 'ADAM':
            self.scheduler.step(valid_loss)
            lr = self.optimizer.param_groups[0]['lr']
            if lr < 1e-5:
                annouce('=> !! learning rate is smaller than threshold !!')
        return lr
    

    def debug(self):
        #! test ntk.py 
        from nastools.ntk import get_ntk_n
        result = get_ntk_n(self.train_queue, [self.model, self.model], num_batch=1)
        import ipdb; ipdb.set_trace()
        from nastools.linear_region_counter import Linear_Region_Collector
        lrc_model = Linear_Region_Collector([self.model, self.model], batch_size=16, sample_batch=10, dataloader=self.train_queue)
        result = lrc_model.forward_batch_sample()
        import ipdb; ipdb.set_trace()



    def run(self):
        
        self.debug()

        annouce(f'=> [5] Train Genotypes')
        self.lr = self.args.lr
        for i_epoch in range(self.args.epochs):
            #! 训练
            train_result = self.train(i_epoch, 'train')
            annouce(f"=> train result [{i_epoch}] - loss: {train_result['loss']:.4f} - metric : {train_result['metric']:.4f}")
            with torch.no_grad():
                #! 验证
                val_result   = self.infer(i_epoch, self.val_queue, 'val')
                annouce(f"=> valid result [{i_epoch}] - loss: {val_result['loss']:.4f} - metric : {val_result['metric']:.4f}")
                #! 测试
                test_result  = self.infer(i_epoch, self.test_queue, 'test')
                annouce(f"=> test  result [{i_epoch}] - loss: {test_result['loss']:.4f} - metric : {test_result['metric']:.4f}", 31)

                self.lr = self.scheduler_step(val_result['loss'])
        
        annouce(f'=> Finished! Genotype = {args.load_genotypes}')
    

    @record_run('train')
    def train(self, i_epoch, stage = 'train'):

        self.model.train()
        epoch_loss   = 0
        epoch_metric = 0
        desc         = '=> training'
        device       = torch.device('cuda')

        with tqdm(self.train_queue, desc = desc, ascii = True) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                #! 1. 准备训练集数据
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)
                E = batch_graphs.edata['feat'].to(device)
                batch_targets = batch_targets.to(device)
                # plot_graphs_threshold(self.args, G, [E, E, E, E])
                #! 2. 优化模型参数
                self.optimizer.zero_grad()
                batch_scores = self.model((G, V, E))
                loss         = self.loss_fn(batch_scores, batch_targets, graph = batch_graphs, stage = stage)
                loss.backward()
                self.optimizer.step()

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets, graph = batch_graphs, stage = stage)

                loss_avg   = epoch_loss / (i_step + 1)
                metric_avg = epoch_metric / (i_step + 1)

                result = {'loss' : loss_avg, 'metric' : metric_avg}
                t.set_postfix(lr = self.lr, **result)
                
        return result


    @record_run('infer')
    def infer(self, i_epoch, dataloader, stage = 'infer'):

        self.model.eval()
        epoch_loss   = 0
        epoch_metric = 0
        desc         = '=> inferring'
        device       = torch.device('cuda')

        with tqdm(dataloader, desc = desc, ascii = True) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)
                E = batch_graphs.edata['feat'].to(device)
                batch_targets = batch_targets.to(device)
                batch_scores  = self.model((G, V, E))
                loss          = self.loss_fn(batch_scores, batch_targets, graph = batch_graphs, stage = stage)

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets, graph = batch_graphs, stage = stage)

                loss_avg   = epoch_loss / (i_step + 1)
                metric_avg = epoch_metric / (i_step + 1)

                result = {'loss' : epoch_loss / (i_step + 1), 'metric' : metric_avg}
                t.set_postfix(**result)

        return result
    

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')

    annouce('==================TRAIN====================', 31)
    annouce('== Graph Neural Architecture Search V2.0 ==', 31)
    annouce('===========================================', 31)

    parser = argparse.ArgumentParser('Graph Neural Architecture Search V2.0')
    parser.add_argument('--task', type = str, default = 'graph_level')
    parser.add_argument('--data', type = str, default = 'ZINC')
    parser.add_argument('--extra', type = str, default = '')
    parser.add_argument('--in_dim_V', type = int, default = 28)
    parser.add_argument('--in_dim_E', type = int, default = 4)
    parser.add_argument('--node_dim', type = int, default = 70)
    parser.add_argument('--edge_dim', type = int, default = 70)
    parser.add_argument('--nb_layers', type = int, default = 4)
    parser.add_argument('--nb_nodes', type = int, default = 3)
    parser.add_argument('--nb_classes', type = int, default = 1)
    parser.add_argument('--leaky_slope', type = float, default = 1e-2)
    parser.add_argument('--batchnorm_op', action = 'store_true', default = False)
    parser.add_argument('--edge_feature', action = 'store_true', default = False)
    parser.add_argument('--nb_mlp_layer', type = int, default = 4)
    parser.add_argument('--dropout', type = float, default = 0.0)
    parser.add_argument('--pos_encode', type = int, default = 0)

    parser.add_argument('--data_clip', type = float, default = 1.0)
    parser.add_argument('--nb_workers', type = int, default = 0)
    parser.add_argument('--seed', type = int, default = 41)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch', type = int, default = 64)
    parser.add_argument('--lr', type = float, default = 0.025)
    parser.add_argument('--lr_min', type = float, default = 0.001)
    parser.add_argument('--momentum', type = float, default = 0.9)
    parser.add_argument('--weight_decay', type = float, default = 3e-4)
    parser.add_argument('--optimizer', type = str, default = 'ADAM')
    parser.add_argument('--patience', type = int, default = 10)
    parser.add_argument('--load_genotypes', type = str, required = True)

    args = parser.parse_args()
    annouce(args)
    Trainer(args).run()
    # - end - #
