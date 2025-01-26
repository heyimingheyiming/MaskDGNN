#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dgl  # For graph neural network operations
import torch  # For tensor operations and deep learning
import torch.nn as nn  # PyTorch neural network module
import torch.nn.functional as F  # PyTorch functional module for common operations
import dgl.function as fn  # DGL module for message passing and aggregation functions
from dgl.utils import expand_as_pair  # Utility to expand features to source and destination node pairs
from model.config import cfg  # Configuration object with model settings
from dgl.nn.pytorch.conv import GraphConv  # Graph convolutional layer from DGL
from torch_scatter import scatter_add  # For aggregation of tensors by index
from copy import deepcopy  # For deep copying objects

class FilterLayer(nn.Module):
    def __init__(self, max_input_length, out_features: int):
        super(FilterLayer, self).__init__()

        self.max_input_length = max_input_length
        self.complex_weight = nn.Parameter(torch.randn(out_features, 2, dtype=torch.float32))
        self.Dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, g=None, fast_weights=None):
        max_input_length, hidden = x.shape

        y = x
        tmp = torch.fft.rfft(y, n=max_input_length, dim=0, norm='forward')
        weight1 = fast_weights
        weight = torch.view_as_complex(weight1)
        tmp = tmp * weight  # Apply weighting in the frequency domain
        sequence_emb_fft = torch.fft.irfft(tmp, n=max_input_length, dim=0, norm='forward')
        sequence_emb_fft = sequence_emb_fft[:, 0:hidden]
        y = self.Dropout(sequence_emb_fft)
        y = x + y  # Residual connection

        return y

class GCNLayer(nn.Module):
    def __init__(self, dim_in, dim_out, pos_isn):
        super(GCNLayer, self).__init__()
        self.pos_isn = pos_isn
        if cfg.gnn.skip_connection == 'affine':
            self.linear_skip_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
            self.linear_skip_bias = nn.Parameter(torch.ones(size=(dim_out, )))
        elif cfg.gnn.skip_connection == 'identity':
            assert dim_in == dim_out

        self.linear_msg_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
        self.linear_msg_bias = nn.Parameter(torch.ones(size=(dim_out, )))

        self.activate = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_skip_weight, gain=gain)
        nn.init.xavier_normal_(self.linear_msg_weight, gain=gain)
        nn.init.constant_(self.linear_skip_bias, 0)
        nn.init.constant_(self.linear_msg_bias, 0)

    def norm(self, graph):
        edge_index = graph.edges()
        row = edge_index[0]
        edge_weight = torch.ones((row.size(0),), device=row.device)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=graph.num_nodes())
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt

    def message_fun(self, edges):
        return {'m': edges.src['h'] * 0.1}  # Pass scaled source node features as messages

    def forward(self, g, feats, fast_weights=None):
        if fast_weights:
            linear_skip_weight = fast_weights[0]
            linear_skip_bias = fast_weights[1]
            linear_msg_weight = fast_weights[2]
            linear_msg_bias = fast_weights[3]
        else:
            linear_skip_weight = self.linear_skip_weight
            linear_skip_bias = self.linear_skip_bias
            linear_msg_weight = self.linear_msg_weight
            linear_msg_bias = self.linear_msg_bias

        feat_src, feat_dst = expand_as_pair(feats, g)
        norm_ = self.norm(g)
        feat_src = feat_src * norm_.view(-1, 1)  # Normalize source node features
        g.srcdata['h'] = feat_src
        aggregate_fn = fn.copy_u('h', 'm')

        g.update_all(message_func=aggregate_fn, reduce_func=fn.sum(msg='m', out='h'))
        rst = g.dstdata['h']
        rst = rst * norm_.view(-1, 1)  # Normalize aggregated results

        rst_ = F.linear(rst, linear_msg_weight, linear_msg_bias)

        if cfg.gnn.skip_connection == 'affine':
            skip_x = F.linear(feats, linear_skip_weight, linear_skip_bias)
        elif cfg.gnn.skip_connection == 'identity':
            skip_x = feats

        return rst_ + skip_x  # Combine aggregated result with skip connection

class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, num_layers, dropout, max_input_length):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_input_length = max_input_length

        for i in range(num_layers):
            d_in = in_features if i == 0 else out_features
            pos_isn = True if i == 0 else False
            layer = GCNLayer(d_in, out_features, pos_isn)
            self.add_module('layer{}'.format(i), layer)

        self.filter_layer = FilterLayer(max_input_length, out_features)

        self.weight1 = nn.Parameter(torch.ones(size=(hidden_dim, out_features)))
        self.weight2 = nn.Parameter(torch.ones(size=(1, hidden_dim)))

        if cfg.model.edge_decoding == 'dot':
            self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)  # Dot product decoding
        elif cfg.model.edge_decoding == 'cosine_similarity':
            self.decode_module = nn.CosineSimilarity(dim=-1)  # Cosine similarity decoding
        else:
            raise ValueError('Unknown edge decoding {}.'.format(cfg.model.edge_decoding))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight1, gain=gain)
        nn.init.xavier_normal_(self.weight2, gain=gain)

    def forward(self, g, x, fast_weights=None):
        total_layers = len(list(self.children()))
        for i, layer in enumerate(self.children()):
            if i == total_layers - 1:
                y = x
                x = self.filter_layer(y, g, fast_weights[2 + i * 4])  # Apply FilterLayer
                break
            x = layer(g, x, fast_weights[2 + i * 4: 2 + (i + 1) * 4])  # Forward through GCN layers

        if fast_weights:
            weight1 = fast_weights[0]
            weight2 = fast_weights[1]
        else:
            weight1 = self.weight1
            weight2 = self.weight2

        x = F.normalize(x)
        g.node_embedding = x

        pred = F.dropout(x, self.dropout)
        pred = F.relu(F.linear(pred, weight1))
        pred = F.dropout(pred, self.dropout)
        pred = torch.sigmoid(F.linear(pred, weight2))

        node_feat = pred[g.edge_label_index]
        nodes_first = node_feat[0]
        nodes_second = node_feat[1]

        pred = self.decode_module(nodes_first, nodes_second)

        return pred
