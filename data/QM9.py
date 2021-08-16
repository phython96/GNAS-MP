import time
import torch
import dgl.data

class QM9DGL(torch.utils.data.Dataset):
    
    def __init__(self, graph_list):
        self.graph_list = graph_list
    
    def __getitem__(self, i):
        return self.graph_list[i]

    def __len__(self):
        return len(self.graph_list)

class QM9Dataset(torch.utils.data.Dataset):

    def __init__(self, name, target):
        """
            Loading QM9 Dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name   = name
        self.target = target
        self.data   = dgl.data.QM9EdgeDataset([target])
        graph_list  = []
        for i in range(len(self.data)):
            graph, label = self.data.__getitem__(i)
            #graph.ndata['feat'] = torch.cat([graph.ndata['pos'], graph.ndata['attr']], dim = -1)
            graph.ndata['feat'] = graph.ndata['attr']
            graph.edata['feat'] = graph.edata['edge_attr']
            graph_list.append((graph, label))

        self.train = QM9DGL(graph_list[:110000])
        self.val   = QM9DGL(graph_list[110000:120000])
        self.test  = QM9DGL(graph_list[120000:])

        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels        = torch.cat(labels, dim = 0)
        return batched_graph, labels


if __name__ == '__main__':
    dataset = QM9Dataset('QM9', 'mu')
    import ipdb; ipdb.set_trace()