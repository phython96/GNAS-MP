import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import MIXED_OPS

class MLPReadout(nn.Module):

	def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
		super().__init__()
		list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
		list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
		self.FC_layers = nn.ModuleList(list_FC_layers)
		self.L = L

	def forward(self, x):
		y = x
		for l in range(self.L):
			y = self.FC_layers[l](y)
			y = F.relu(y)
		y = self.FC_layers[self.L](y)
		return y

class OpModule(nn.Module):

	def __init__(self, args, operation_name):
		super(OpModule, self).__init__()
		self.args = args
		self._feature_dim = args.feature_dim
		self._dropout = args.dropout
		args = {'feature_dim' : self._feature_dim}
		self.op = MIXED_OPS[operation_name](args)
		self.linear = nn.Linear(self._feature_dim, self._feature_dim, bias = True)
		self.batchnorm_h = nn.BatchNorm1d(self._feature_dim)
		self.activate = nn.ReLU()

	def forward(self, g, h, h_in) :
		h = self.op(g, h, h_in)
		h = self.linear(h)
		if self.args.op_norm:
			h = self.batchnorm_h(h) 
		h = self.activate(h)
		#h = F.dropout(h, self._dropout, training=self.training)
		return h

class Cell(nn.Module):

	def __init__(self, args, genotype):
		super(Cell, self).__init__()
		self.args = args
		self._genotype = genotype
		self._nb_nodes = len(set([edge[1] for edge in genotype.alpha_cell]))
		self._feature_dim = args.feature_dim
		self._dropout = args.dropout
		self._concat_node = list(range(1, 1 + self._nb_nodes)) if genotype.concat_node is None else genotype.concat_node
		self.batchnorm_h = nn.BatchNorm1d(self._feature_dim)
		self.activate = nn.ReLU()
		self._compile()

	def _compile(self):
		nb_nodes = self._nb_nodes
		self._ops = nn.ModuleList([nn.ModuleList([nn.ModuleList() for i in range(n)]) for n in range(1, 1 + nb_nodes)])
		for (op_name, center_node, pre_node) in self._genotype.alpha_cell:
			center_node -= 1
			self._ops[center_node][pre_node].append(OpModule(self.args, op_name))
		self.concat = nn.Linear(len(self._concat_node) * self._feature_dim, self._feature_dim)

	def forward(self, g, h):
		h_in = h # for residual
		states = [h]
		for n in range(self._nb_nodes):
			hs = []
			for i in range(n + 1):
				if len(self._ops[n][i]) > 0:
					_h = self._ops[n][i][0](g, states[i], h_in)
					#_h = F.dropout(_h, self._dropout, training=self.training)
					hs.append(_h)
			states.append(sum(hs))

		states = [states[idx] for idx in self._concat_node]
		nh = self.concat(torch.cat(states, dim = 1))
		nh = self.batchnorm_h(nh)	# batchnorm before ReLU
		nh = self.activate(nh)		# ReLU
		nh = F.dropout(nh, self._dropout, training=self.training)
		nh = h_in + nh				# residual gor deep network
		return nh

class Network(nn.Module):
	
	#def __init__(self, genotype, layers, in_dim, feature_dim, num_classes, criterion, data_type='gc', readout='mean', dropout = 0.0):
	def __init__(self, args, genotype, num_classes, in_dim, criterion):
		super(Network, self).__init__()
		self.args = args
		self._genotype = genotype
		self._layers = args.layers
		self._in_dim = in_dim
		self._feature_dim = args.feature_dim
		self._num_classes = num_classes
		self._criterion = criterion
		self._data_type = args.data_type
		self._readout = args.readout
		self._dropout = args.dropout
		
		if self._data_type in ['nc', 'rg']:
			self.embedding_h = nn.Embedding(self._in_dim, self._feature_dim)  # node feat is an integer
		else:
			self.embedding_h = nn.Linear(self._in_dim, self._feature_dim)

		if type(self._genotype) == list:
			genotypes = self._genotype
		else:
			genotypes = [self._genotype for i in range(self._layers)]
		self.cells = nn.ModuleList([Cell(args, genotypes[i]) for i in range(self._layers)])
		# judge whether link prediction task
		outdim = self._feature_dim if self._data_type not in ['ec'] else 2 * self._feature_dim
		self.classifier = MLPReadout(outdim, self._num_classes)
	
	def _forward(self, g, h):
		h = self.embedding_h(h)
		for i, cell in enumerate(self.cells):
			h = cell(g, h)
		return h
	
	def forward(self, g, h):
		h = self._forward(g, h)
		if self._data_type == 'nc':
			h = self.classifier(h)
			return h
		elif self._data_type in ['gc', 'rg']:
			g.ndata['h'] = h
			if self._readout == "sum":
				hg = dgl.sum_nodes(g, 'h')
			elif self._readout == "max":
				hg = dgl.max_nodes(g, 'h')
			elif self._readout == "mean":
				hg = dgl.mean_nodes(g, 'h')
			else:
				hg = dgl.mean_nodes(g, 'h') 
			hg = self.classifier(hg)
			return hg
		elif self._data_type in ['ec']:
			def _edge_feat(edges):
				e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
				e = self.classifier(e)
				return {'e': e}
			g.ndata['h'] = h
			g.apply_edges(_edge_feat)
			return g.edata['e']
	
	def _loss(self, g, h, target):
		logits = self.forward(g, h)
		return self._criterion(logits, target)
	
		
