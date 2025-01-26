#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dgl
import math
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from model import MaskDGNN
from test_new import test
from train_new import train
from model.config import cfg
from deepsnap.graph import Graph
from model.Logger import getLogger
from dataset_prep import load, load_r
from model.utils import create_optimizer
from deepsnap.dataset import GraphDataset
from scipy.sparse import coo_matrix
import warnings

warnings.filterwarnings("ignore")
from collections import Counter


def mask_edge_new(edge_index: torch.Tensor, edge_feature: torch.Tensor, node_activity_score, p):
    """
    Calculate masking probabilities for edges based on node activity scores. More important edges are less likely to be masked.
    """
    if edge_index.size(1) <= 10:
        e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.full_like(e_ids, 0.0, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        remaining_edge_index = edge_index[:, ~mask]
        masked_edge_index = edge_index[:, mask]
    else:
        edges = edge_index.t()
        edge_scores = []
        for idx, edge in enumerate(edges):
            node1, node2 = edge.tolist()
            score1 = node_activity_score.get(node1, 0)
            score2 = node_activity_score.get(node2, 0)
            edge_score = (score1 + score2) / 2
            edge_scores.append((1 - edge_score, idx))

        edge_scores_sorted = sorted(edge_scores, key=lambda x: x[0], reverse=True)
        num_edges = edge_index.shape[1]
        mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)

        num_to_mask = int(num_edges * p)
        total_score = sum([score for score, _ in edge_scores_sorted])

        mask_probabilities = []
        for score, idx in edge_scores_sorted:
            adjusted_score = score * (num_to_mask / total_score)
            mask_probabilities.append(min(adjusted_score, 1))

        for mask_prob, idx in zip(mask_probabilities, edge_scores_sorted):
            if np.random.rand() < mask_prob:
                mask[idx[1]] = True

        remaining_edge_index = edge_index[:, ~mask]
        masked_edge_index = edge_index[:, mask]

    return remaining_edge_index, masked_edge_index, mask


def mask_edge_norate(edge_index: torch.Tensor, edge_feature: torch.Tensor, node_activity_score):
    """
    Mask edges without considering a specific masking rate.
    """
    if edge_index.size(1) <= 10:
        e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.full_like(e_ids, 0.0, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        remaining_edge_index = edge_index[:, ~mask]
        masked_edge_index = edge_index[:, mask]
    else:
        edges = edge_index.t()
        edge_scores = []
        for idx, edge in enumerate(edges):
            node1, node2 = edge.tolist()
            score1 = node_activity_score.get(node1, 0)
            score2 = node_activity_score.get(node2, 0)
            edge_score = (score1 + score2) / 2
            edge_scores.append((1 - edge_score, idx))

        edge_scores_sorted = sorted(edge_scores, key=lambda x: x[0], reverse=True)
        num_edges = edge_index.shape[1]
        mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)

        for mask_prob, idx in edge_scores_sorted:
            if np.random.rand() < mask_prob:
                mask[idx] = True

        remaining_edge_index = edge_index[:, ~mask]
        masked_edge_index = edge_index[:, mask]

    return remaining_edge_index, masked_edge_index, mask

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uci-msg', help='Dataset')
    parser.add_argument('--cuda_device', type=int, default=2, help='Cuda device')
    parser.add_argument('--seed', type=int, default=2023, help='Seed for reproducibility')
    parser.add_argument('--repeat', type=int, default=10, help='Number of repetitions')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--out_dim', type=int, default=64, help='Output dimension')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--maml_lr', type=float, default=0.008, help='Meta-learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--drop_rate', type=float, default=0.16, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--num_hidden', type=int, default=256, help='Number of hidden units')
    parser.add_argument('--window_num', type=int, default=8, help='Window size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--residual', type=bool, default=False, help='Residual connections')
    parser.add_argument('--beta', type=float, default=0.89, help='Beta for adaptive learning rate')
    parser.add_argument('--mask', type=float, default=0.2, help='Edge masking probability')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping rounds')

    args = parser.parse_args()
    logger = getLogger(cfg.log_path)

    # Load dataset
    if args.dataset in ["reddit_body", "reddit_title", "as_733", "uci-msg", "bitcoinotc", "bitcoinalpha",
                        'stackoverflow_M', "mooc", "uslegis", "untrade", "unvote", "canparl", "bitcoinotc-yuan",
                        "bitcoinalpha-yuan", "dblp", "wiki-talk-temporal"]:
        graphs, e_feat, e_time, n_feat, edge_index1, node_activity_score = load_r(args.dataset)
    else:
        raise ValueError

    n_dim = n_feat[0].shape[1]
    n_node = n_feat[0].shape[0]
    device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device >= 0 else 'cpu')

    all_auc_avg = 0.0
    best_auc = 0.0
    best_model = 0

    for rep in range(args.repeat):
        logger.info(
            'dataset:{}, epochs:{}, num_layers:{}, num_hidden:{}, lr:{}, maml_lr:{}, window_num:{}, drop_rate:{}, mask:{}'.format(
                args.dataset, args.epochs, args.num_layers, args.num_hidden, args.lr, args.maml_lr, args.window_num,
                args.drop_rate, args.mask))
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        graph_l = []
        graph_t = []
        n = math.ceil(len(graphs) * 0.7)

        for idx, graph in tqdm(enumerate(graphs)):
            graph_d = dgl.from_scipy(graph)
            graph_d.edge_feature = torch.Tensor(e_feat[idx])
            graph_d.edge_time = torch.Tensor(e_time[idx])

            if n_feat[idx].shape[0] != n_node or n_feat[idx].shape[1] != n_dim:
                graph_d.node_feature = torch.Tensor(graph_l[idx - 1].node_feature)
            else:
                graph_d.node_feature = torch.Tensor(n_feat[idx])

            graph_d = dgl.remove_self_loop(graph_d)
            graph_d = dgl.add_self_loop(graph_d)

            edges = graph_d.edges()
            row = edges[0].numpy()
            col = edges[1].numpy()
            row_tensor = torch.from_numpy(row)
            col_tensor = torch.from_numpy(col)
            row_tensor = row_tensor[:graph_d.num_edges() - graph_d.num_nodes()]
            col_tensor = col_tensor[:graph_d.num_edges() - graph_d.num_nodes()]

            y_pos = np.ones(shape=(len(row_tensor),))
            y_neg = np.zeros(shape=(len(row_tensor),))
            y = list(y_pos) + list(y_neg)

            edge_label_index = [row.tolist(), col.tolist()]

            graph_d.edge_label = torch.Tensor(y)
            graph_d.edge_label_index = torch.LongTensor(edge_label_index)

            graph_l.append(graph_d)

            remaining_edges, masked_edges, mask1 = mask_edge_new(
                torch.stack([row_tensor, col_tensor]), graph_d.edge_feature, node_activity_score[idx], args.mask)

            row1 = remaining_edges[0]
            col1 = remaining_edges[1]
            ts = [1] * len(row1)
            sub_g = coo_matrix((ts, (row1, col1)), shape=(graph_d.num_nodes(), graph_d.num_nodes()))

            graph_t.append(dgl.from_scipy(sub_g))
            graph_t[idx] = dgl.remove_self_loop(graph_t[idx])
            graph_t[idx] = dgl.add_self_loop(graph_t[idx])
            graph_t[idx].node_feature = graph_d.node_feature
            graph_t[idx].edge_label_index = graph_d.edge_label_index
            graph_t[idx].edge_label = graph_d.edge_label

        model = WinGNN.Model(n_dim, args.out_dim, args.num_hidden, args.num_layers, args.dropout, n_node)
        model.train()
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
        model = model.to(device)

        best_param = train(args, model, optimizer, device, graph_t, graph_l, logger, n)
        model.load_state_dict(best_param['best_state'])
        S_dw = best_param['best_s_dw']

        model.eval()
        avg_auc = test(graph_l, model, args, logger, n, S_dw, device)

        if avg_auc > best_auc:
            best_model = best_param['best_state']
        all_auc_avg += avg_auc

    torch.save(best_model, f'model_parameter/{args.dataset}.pkl')
    all_auc_avg = all_auc_avg / args.repeat
    print(all_auc_avg)
