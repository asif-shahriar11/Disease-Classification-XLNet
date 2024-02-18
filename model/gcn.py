from dgl.nn.pytorch import GraphConv
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
import torch as th
import torch.nn as nn

class CustomGraphConv(GraphConv):

    def forward(self, graph, feat,  weight=None, edge_weights=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
        

class GCN(nn.Module):
    def __init__(self, input_features, n_hidden, n_classes, n_layers, activation, dropout, normalization = 'None'):
        """
        :param in_feats: number of features for each input node
        :param n_hidden: number of features for each hidden layer
        :param n_classes: number of output classes for the classification task
        :param n_layers: total number of layers in the network, including output layer but not input layer
        :param activation: activation function to be used between layers
        :param dropout: dropout probability
        :param normalization: normalization to be used after each layer, can be 'None', 'batch', 'layer', or 'both'
        """
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(CustomGraphConv(input_features, n_hidden, activation=activation, norm=normalization))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(CustomGraphConv(n_hidden, n_hidden, activation=activation, norm=normalization))
        # output layer
        self.layers.append(CustomGraphConv(n_hidden, n_classes, activation=None, norm=normalization))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features, edge_weights=None):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weights=edge_weights)
        return h