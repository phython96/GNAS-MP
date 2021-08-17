import os
import sys
import dgl
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from data import *
from models.model_search import *
from utils.utils import *
from models.architect import Architect
# from utils.visualize import *
from utils.utils import DecayScheduler


class Searcher(object):
    
    def __init__(self, args):

        self.args = args
        self.console = Console()

        self.console.print('=> [1] Initial settings')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled   = True

        self.console.print('=> [2] Initial models')
        self.metric    = load_metric(args)
        self.loss_fn   = get_loss_fn(args).cuda()
        self.model     = Model_Search(args, get_trans_input(args), self.loss_fn).cuda()
        self.console.print(f'[red]=> Supernet Parameters: {count_parameters_in_MB(self.model)}')

        self.console.print(f'=> [3] Preparing dataset')
        self.dataset     = load_data(args)
        if args.pos_encode > 0:
            #! add positional encoding
            self.console.print(f'==> [3.1] Adding positional encodings')
            self.dataset._add_positional_encodings(args.pos_encode)
        self.search_data = self.dataset.train
        self.val_data    = self.dataset.val
        self.test_data   = self.dataset.test
        self.load_dataloader()

        self.console.print(f'=> [4] Initial optimizer')
        self.optimizer   = torch.optim.SGD(
            params       = self.model.parameters(),
            lr           = args.lr,
            momentum     = args.momentum,
            weight_decay = args.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
            optimizer  = self.optimizer,
            T_max      = float(args.epochs),
            eta_min    = args.lr_min
        )

        self.architect = Architect(self.model, self.args)


    def load_dataloader(self):

        num_search  = int(len(self.search_data) * self.args.data_clip)
        indices     = list(range(num_search))
        split       = int(np.floor(self.args.portion * num_search))
        self.console.print(f'=> Para set size: {split}, Arch set size: {num_search - split}')
        
        self.para_queue = torch.utils.data.DataLoader(
            dataset     = self.search_data,
            batch_size  = self.args.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory  = True,
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate
        )

        self.arch_queue = torch.utils.data.DataLoader(
            dataset     = self.search_data,
            batch_size  = self.args.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
            pin_memory  = True,
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate
        )

        num_valid = int(len(self.val_data) * self.args.data_clip)
        indices   = list(range(num_valid))

        self.val_queue  = torch.utils.data.DataLoader(
            dataset     = self.val_data,
            batch_size  = self.args.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            pin_memory  = True,
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate
        )

        num_test = int(len(self.test_data) * self.args.data_clip)
        indices  = list(range(num_test))

        self.test_queue = torch.utils.data.DataLoader(
            dataset     = self.test_data,
            batch_size  = self.args.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            pin_memory  = True,
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate
        )


    def run(self):

        self.console.print(f'=> [4] Search & Train')
        for i_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.lr = self.scheduler.get_lr()[0]
            # if i_epoch % self.args.report_freq == 0:
            #     # todo report genotype
            #     geno = genotypes(
            #         args           = self.args,
            #         cell_arch_para = self.model.arch_para_dict(),
            #         cell_arch_topo = self.model.cell_arch_topo
            #     )
            #     plot_genotypes(args, i_epoch, geno)
            #     print( geno )
            #     # for i in range(self.args.nb_layers):
            #     #     print( self.model.arch_para_dict()[i]['V'].softmax(1).detach().cpu().numpy() )

            search_result = self.search()
            self.console.print(f"=> [{i_epoch}] search result - loss: {search_result['loss']:.4f} - metric : {search_result['metric']:.4f}")
            DecayScheduler().step(i_epoch)

            with torch.no_grad():
                val_result  = self.infer(self.val_queue)
                self.console.print(f"[yellow]=> [{i_epoch}] valid result  - loss: {val_result['loss']:.4f} - metric : {val_result['metric']:.4f}")

                test_result = self.infer(self.test_queue)
                self.console.print(f"[red]=> [{i_epoch}] test  result  - loss: {test_result['loss']:.4f} - metric : {test_result['metric']:.4f}")


    def search(self):
        
        self.model.train()
        epoch_loss   = 0
        epoch_metric = 0
        desc         = '=> searching'
        device       = torch.device('cuda')

        with tqdm(self.para_queue, desc = desc, ascii = True) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                #! 1. preparing training datasets
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)
                # E = batch_graphs.edata['feat'].to(device)
                batch_targets = batch_targets.to(device)
                #! 2. preparing validating datasets
                batch_graphs_search, batch_targets_search = next(iter(self.arch_queue))
                GS = batch_graphs_search.to(device)
                VS = batch_graphs_search.ndata['feat'].to(device)
                # ES = batch_graphs_search.edata['feat'].to(device)
                batch_targets_search = batch_targets_search.to(device)
                #! 3. optimizing architecture topology parameters
                self.architect.step(
                    input_train       = {'G': G, 'V': V},
                    target_train      = batch_targets,
                    input_valid       = {'G': GS, 'V': VS},
                    target_valid      = batch_targets_search,
                    eta               = self.lr,
                    network_optimizer = self.optimizer,
                    unrolled          = self.args.unrolled
                )
                #! 4. optimizing model parameters
                self.optimizer.zero_grad()
                batch_scores  = self.model({'G': G, 'V': V})
                loss          = self.loss_fn(batch_scores, batch_targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets)
                t.set_postfix(lr         = self.lr,
                              skip_decay = DecayScheduler().decay_rate, 
                              loss       = epoch_loss / (i_step + 1),
                              metric     = epoch_metric / (i_step + 1))

        return {'loss'   : epoch_loss / (i_step + 1), 
                'metric' : epoch_metric / (i_step + 1)}


    def infer(self, dataloader):

        self.model.eval()
        epoch_loss   = 0
        epoch_metric = 0
        desc         = '=> inferring'
        device       = torch.device('cuda')

        with tqdm(dataloader, desc = desc, ascii = True) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)
                # E = batch_graphs.edata['feat'].to(device)
                batch_targets = batch_targets.to(device)
                batch_scores  = self.model({'G': G, 'V': V})
                loss          = self.loss_fn(batch_scores, batch_targets)

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets)
                t.set_postfix(loss   = epoch_loss / (i_step + 1), 
                              metric = epoch_metric / (i_step + 1))

        return {'loss'   : epoch_loss / (i_step + 1), 
                'metric' : epoch_metric / (i_step + 1)}


if __name__ == '__main__':

    import warnings
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser('Rethinking Graph Neural Architecture Search From Message Passing')
    parser.add_argument('--task',               type = str,             default = 'graph_level')
    parser.add_argument('--data',               type = str,             default = 'ZINC')
    parser.add_argument('--in_dim_V',           type = int,             default = 28)
    parser.add_argument('--node_dim',           type = int,             default = 70)
    parser.add_argument('--nb_classes',         type = int,             default = 1)
    parser.add_argument('--nb_layers',          type = int,             default = 4)
    parser.add_argument('--nb_nodes',           type = int,             default = 3)
    parser.add_argument('--leaky_slope',        type = float,           default = 0.1)
    parser.add_argument('--batchnorm_op',       action = 'store_true',  default = False)
    parser.add_argument('--nb_mlp_layer',       type = int,             default = 4)
    parser.add_argument('--dropout',            type = float,           default = 0.0)
    parser.add_argument('--pos_encode',         type = int,             default = 0)

    parser.add_argument('--portion',            type = float,           default = 0.5)
    parser.add_argument('--data_clip',          type = float,           default = 1.0)
    parser.add_argument('--nb_workers',         type = int,             default = 0)
    parser.add_argument('--seed',               type = int,             default = 41)
    parser.add_argument('--epochs',             type = int,             default = 50)
    parser.add_argument('--batch',              type = int,             default = 64)
    parser.add_argument('--lr',                 type = float,           default = 0.025)
    parser.add_argument('--lr_min',             type = float,           default = 0.001)
    parser.add_argument('--momentum',           type = float,           default = 0.9)
    parser.add_argument('--weight_decay',       type = float,           default = 3e-4)
    parser.add_argument('--unrolled',           action = 'store_true',  default = False)
    parser.add_argument('--search_mode',        type = str,             default = 'train')
    parser.add_argument('--arch_lr',            type = float,           default = 3e-4)
    parser.add_argument('--arch_weight_decay',  type = float,           default =1e-3)
    parser.add_argument('--report_freq',        type = int,             default = 1)
    parser.add_argument('--arch_save',          type = str,             default = './save_arch')

    console = Console()
    args = parser.parse_args()
    title   = "Rethinking Graph Neural Architecture Search from Message Passing"
    richPanel = Panel.fit("[gray]" + str(vars(args)), title = title)
    console.print(richPanel)
    data_path = os.path.join(args.arch_save, args.data)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    with open(os.path.join(data_path, "configs.txt"), "w") as f:
        f.write(str(args))
    # DecayScheduler(T_max = args.epochs)
    Searcher(args).run()