#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :load datasets

import os
import copy
import math
import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import networkx as nx

def load(nodes_num):
    path = "dataset/dblp_timestamp/"

    train_e_feat_path = path + 'train_e_feat/' + type + '/'
    test_e_feat_path = path + 'test_e_feat/' + type + '/'

    train_n_feat_path = path + type + '/' + 'train_n_feat/'
    test_n_feat_path = path + type + '/' + 'test_n_feat/'

    path = path + type
    train_path = path + '/train/'
    test_path = path + '/test/'

    train_n_feat = read_e_feat(train_n_feat_path)
    test_n_feat = read_e_feat(test_n_feat_path)

    train_e_feat = read_e_feat(train_e_feat_path)
    test_e_feat = read_e_feat(test_e_feat_path)

    num = 0
    train_graph = read_graph(train_path, nodes_num, num)
    num = num + len(train_graph)
    test_graph = read_graph(test_path, nodes_num, num)
    return train_graph, train_e_feat, train_n_feat, test_graph, test_e_feat, test_n_feat

def load_r(name):
    path = "dataset/" + name
    path_ei = path + '/' + 'edge_index/'
    path_nf = path + '/' + 'node_feature/'
    path_ef = path + '/' + 'edge_feature/'
    path_et = path + '/' + 'edge_time/'

    edge_index = read_npz(path_ei)
    edge_feature = read_npz(path_ef)
    node_feature = read_npz(path_nf)
    edge_time = read_npz(path_et)

    nodes_num = node_feature[0].shape[0]


    sub_graph = []
    for e_i in edge_index:
        row = e_i[0]
        col = e_i[1]
        ts = [1] * len(row)
        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)

    nodes_num = node_feature[0].shape[0]

    pagerank_scores_all, connected_nodes_all = compute_all_pageranks(sub_graph, nodes_num)
    #degree_scores_all, connected_nodes_all = compute_all_degree_scores(sub_graph, nodes_num) if dataset=wiki-talk-temporal use degree_scores instead of pagerank_scores

    node_change_scores_all = compute_node_change_scores_noaverge(sub_graph, nodes_num)

    node_activity_score=combine_scores_with_weights(pagerank_scores_all, node_change_scores_all)
    #node_activity_score = combine_scores_with_weights(degree_scores_all, node_change_scores_all)
    return sub_graph, edge_feature, edge_time, node_feature, edge_index, node_activity_score  #return sub_graph, edge_feature, edge_time, node_feature, edge_index, node_activity_score

def read_npz(path):
    filesname = os.listdir(path)
    npz = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('.')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        npz.append(np.load(path + filename, allow_pickle=True))
    return npz

def read_e_feat(path):
    filesname = os.listdir(path)
    e_feat = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        e_feat.append(np.load(path+filename))
    return e_feat

def read_graph(path, nodes_num, num):
    filesname = os.listdir(path)
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id) - num
        file_s[id] = filename

    sub_graph = []
    for file in file_s:
        sub_ = pd.read_csv(path + file)

        row = sub_.src_l.values
        col = sub_.dst_l.values

        node_m = set(row).union(set(col))
        ts = [1] * len(row)

        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)

    return sub_graph


def combine_scores_with_weights(pagerank_scores_all, node_change_scores_all, weight_pr=0.3, weight_nc=0.7):
    combined_scores_all = []

    for snapshot_idx, (pagerank_scores, node_change_scores) in enumerate(
            zip(pagerank_scores_all, node_change_scores_all)):
        combined_scores = {}
        all_nodes = set(pagerank_scores.keys()).union(node_change_scores.keys())
        for node in all_nodes:
            pr_score = pagerank_scores.get(node, 0)
            nc_score = node_change_scores.get(node, 0)
            combined_scores[node] = weight_pr * pr_score + weight_nc * nc_score
        combined_scores_all.append(combined_scores)

    return combined_scores_all


def compute_degree_scores(sub_graph, nodes_num):
    a = 0
    b = 1
    lambda_param = 1
    max_exp_input = 200

    G = nx.from_scipy_sparse_matrix(sub_graph, create_using=nx.Graph())
    connected_nodes = {node for node in G.nodes() if len(list(G.neighbors(node))) > 0}
    subgraph = G.subgraph(connected_nodes)

    degree_scores = {node: subgraph.degree(node) for node in subgraph.nodes()}
    max_degree = max(degree_scores.values()) if degree_scores else 0
    min_degree = min(degree_scores.values()) if degree_scores else 0

    if not degree_scores:
        return {}, connected_nodes

    sorted_nodes = sorted(degree_scores.items(), key=lambda x: x[1])
    sorted_scores = [score for _, score in sorted_nodes]

    interval_size = (max_degree - min_degree) / 10 if max_degree != min_degree else 1
    intervals = [(min_degree + i * interval_size, min_degree + (i + 1) * interval_size) for i in range(10)]
    interval_counts = [0] * 10

    for score in sorted_scores:
        for i, (low, high) in enumerate(intervals):
            if low <= score < high:
                interval_counts[i] += 1
                break

    total_nodes = len(sorted_scores)
    interval_ratios = [count / total_nodes for count in interval_counts]

    accumulated_ratio = 0.0
    noise_intervals = set()
    for i in range(9, -1, -1):
        accumulated_ratio += interval_ratios[i]
        if accumulated_ratio > 0.05:
            break
        noise_intervals.add(i)

    if not noise_intervals:
        pass
    else:
        noise_score = min([intervals[i][0] for i in noise_intervals])
        for node, score in degree_scores.items():
            if score >= noise_score:
                degree_scores[node] = noise_score

    max_degree = max(degree_scores.values())
    min_degree = min(degree_scores.values())
    if max_degree == min_degree:
        normalized_degree_scores = {node: (a + b) / 2 for node in degree_scores}
    else:
        normalized_degree_scores = {
            node: a + (b - a) * (
                    (min(lambda_param * (deg), max_exp_input)) /
                    (min(lambda_param * (max_degree), max_exp_input))
            )
            for node, deg in degree_scores.items()
        }

    return normalized_degree_scores, connected_nodes


def compute_all_degree_scores(sub_graphs, nodes_num):
    all_degree_scores = []
    all_connected_nodes = []
    for idx, sg in enumerate(sub_graphs):
        degree_scores, connected = compute_degree_scores(sg, nodes_num)
        all_degree_scores.append(degree_scores)
        all_connected_nodes.append(connected)
        print(f"Snapshot {idx}: Computed degree scores for {len(connected)} connected nodes")
    return all_degree_scores, all_connected_nodes

def compute_pagerank(sub_graph, nodes_num, a=0, b=1.0, lambda_param=1.0):
    G = nx.from_scipy_sparse_matrix(sub_graph, create_using=nx.Graph())
    connected_nodes = {node for node in G.nodes() if len(list(G.neighbors(node))) > 0}
    subgraph = G.subgraph(connected_nodes)
    pagerank_scores = nx.pagerank(subgraph, alpha=0.85)

    if not pagerank_scores:
        return {}, connected_nodes

    sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1])
    sorted_scores = [score for _, score in sorted_nodes]
    min_pr = sorted_scores[0]
    max_pr = sorted_scores[-1]
    interval_size = (max_pr - min_pr) / 10
    intervals = [(min_pr + i * interval_size, min_pr + (i + 1) * interval_size) for i in range(10)]
    interval_counts = [0] * 10

    for score in sorted_scores:
        for i, (low, high) in enumerate(intervals):
            if low <= score < high:
                interval_counts[i] += 1
                break

    total_nodes = len(sorted_scores)
    interval_ratios = [count / total_nodes for count in interval_counts]
    accumulated_ratio = 0.0
    noise_intervals = set()
    for i in range(9, -1, -1):
        accumulated_ratio += interval_ratios[i]
        if accumulated_ratio > 0.05:
            break
        noise_intervals.add(i)

    if noise_intervals:
        noise_score = min([intervals[i][0] for i in noise_intervals])
        for node, score in pagerank_scores.items():
            if score >= noise_score:
                pagerank_scores[node] = noise_score

    max_pr = max(pagerank_scores.values())
    min_pr = min(pagerank_scores.values())

    if max_pr == min_pr:
        advanced_scaled_pagerank_scores = {node: (a + b) / 2 for node in pagerank_scores}
    else:
        advanced_scaled_pagerank_scores = {
            node: a + (b - a) * (
                    ((lambda_param * (pr))) /
                    ((lambda_param * (max_pr)))
            )
            for node, pr in pagerank_scores.items()
        }

    return advanced_scaled_pagerank_scores, connected_nodes

def compute_all_pageranks(sub_graphs, nodes_num):
    all_pageranks = []
    all_connected_nodes = []
    for idx, sg in enumerate(sub_graphs):
        pr, connected = compute_pagerank(sg, nodes_num)
        all_pageranks.append(pr)
        all_connected_nodes.append(connected)
        print(f"Snapshot {idx}: Computed PageRank for {len(connected)} connected nodes")
    return all_pageranks, all_connected_nodes

def pagerank_to_matrix(pagerank_scores_all, connected_nodes_all, nodes_num):
    matrices = []
    for snapshot_idx, pagerank_scores in enumerate(pagerank_scores_all):
        matrix = np.zeros((1, nodes_num))
        for node, score in pagerank_scores.items():
            matrix[0, node] = score
        matrices.append(matrix)
    return matrices

def compute_node_change_scores_noaverge(sub_graphs, nodes_num, gamma=1.0, lambda_param=1.0, a=0.0, b=1.0):
    node_change_scores_all = []
    first_graph = nx.from_scipy_sparse_matrix(sub_graphs[0], create_using=nx.Graph())
    connected_nodes_first = {node for node in first_graph.nodes() if len(list(first_graph.neighbors(node))) > 0}
    first_node_change_scores = {node: (a + b) / 2 for node in connected_nodes_first}
    node_change_scores_all.append(first_node_change_scores)
    print(f"Snapshot 0: Assigned default node change scores for {len(first_node_change_scores)} nodes")

    for t in range(1, len(sub_graphs)):
        current_graph = nx.from_scipy_sparse_matrix(sub_graphs[t], create_using=nx.Graph())
        prev_graph = nx.from_scipy_sparse_matrix(sub_graphs[t - 1], create_using=nx.Graph())

        connected_nodes_current = {node for node in current_graph.nodes() if len(list(current_graph.neighbors(node))) > 0}
        subgraph_current = current_graph.subgraph(connected_nodes_current)
        connected_nodes_prev = {node for node in prev_graph.nodes() if len(list(prev_graph.neighbors(node))) > 0}
        subgraph_prev = prev_graph.subgraph(connected_nodes_prev)

        prev_degrees = dict(subgraph_prev.degree())

        edge_changes = {}
        for node in set(subgraph_current.nodes()):
            current_neighbors = set(subgraph_current.neighbors(node)) if node in subgraph_current else set()
            prev_neighbors = set(subgraph_prev.neighbors(node)) if node in subgraph_prev else set()
            edge_changes[node] = len(current_neighbors.symmetric_difference(prev_neighbors))

        node_change_scores = {}
        max_exp_input = 200

        for node in edge_changes:
            prev_degree = prev_degrees.get(node, 0)
            delta_e = edge_changes[node]
            if prev_degree == 0 and delta_e == 0:
                score = 0
            elif prev_degree == 0 and delta_e != 0:
                score = gamma * (min(lambda_param * (delta_e / math.sqrt(delta_e - 0.5)), max_exp_input))
            else:
                score = gamma * (min(lambda_param * (delta_e / math.sqrt(prev_degree + 2)), max_exp_input))
            node_change_scores[node] = score

        sorted_nodes = sorted(node_change_scores.items(), key=lambda x: x[1])
        sorted_scores = [score for _, score in sorted_nodes]
        min_score = sorted_scores[0]
        max_score = sorted_scores[-1]
        interval_size = (max_score - min_score) / 10

        intervals = [(min_score + i * interval_size, min_score + (i + 1) * interval_size) for i in range(10)]
        interval_counts = [0] * 10
        for score in sorted_scores:
            for i, (low, high) in enumerate(intervals):
                if low <= score < high:
                    interval_counts[i] += 1
                    break

        total_nodes = len(sorted_scores)
        interval_ratios = [count / total_nodes for count in interval_counts]
        accumulated_ratio = 0.0
        noise_intervals = set()
        for i in range(9, -1, -1):
            accumulated_ratio += interval_ratios[i]
            if accumulated_ratio > 0.05:
                break
            noise_intervals.add(i)

        if noise_intervals:
            noise_score = min([intervals[i][0] for i in noise_intervals])
            for node, score in node_change_scores.items():
                if score >= noise_score:
                    node_change_scores[node] = noise_score

        if node_change_scores:
            max_score = max(node_change_scores.values())
            min_score = min(node_change_scores.values())
            if max_score == min_score:
                node_change_scores = {node: (a + b) / 2 for node in node_change_scores}
            else:
                node_change_scores = {
                    node: a + (b - a) * (score / max_score)
                    for node, score in node_change_scores.items()
                }

        node_change_scores_all.append(node_change_scores)
        print(f"Snapshot {t}: Computed and normalized node change scores for {len(node_change_scores)} nodes")

    return node_change_scores_all

def compute_node_change_scores(sub_graphs, nodes_num, gamma=1.0, lambda_param=1.0, a=0.4, b=1.0):
    node_change_scores_all = []
    first_graph = nx.from_scipy_sparse_matrix(sub_graphs[0], create_using=nx.Graph())
    connected_nodes_first = {node for node in first_graph.nodes() if len(list(first_graph.neighbors(node))) > 0}
    first_node_change_scores = {node: (a + b) / 2 for node in connected_nodes_first}
    node_change_scores_all.append(first_node_change_scores)
    print(f"Snapshot 0: Assigned default node change scores for {len(first_node_change_scores)} nodes")

    for t in range(1, len(sub_graphs)):
        current_graph = nx.from_scipy_sparse_matrix(sub_graphs[t], create_using=nx.Graph())
        prev_graph = nx.from_scipy_sparse_matrix(sub_graphs[t - 1], create_using=nx.Graph())

        connected_nodes_current = {node for node in current_graph.nodes() if len(list(current_graph.neighbors(node))) > 0}
        subgraph_current = current_graph.subgraph(connected_nodes_current)
        connected_nodes_prev = {node for node in prev_graph.nodes() if len(list(prev_graph.neighbors(node))) > 0}
        subgraph_prev = prev_graph.subgraph(connected_nodes_prev)

        prev_degrees = dict(subgraph_prev.degree())

        edge_changes = {}
        for node in set(subgraph_current.nodes()):
            current_neighbors = set(subgraph_current.neighbors(node)) if node in subgraph_current else set()
            prev_neighbors = set(subgraph_prev.neighbors(node)) if node in subgraph_prev else set()
            edge_changes[node] = len(current_neighbors.symmetric_difference(prev_neighbors))

        node_change_scores = {}
        max_exp_input = 200

        for node in edge_changes:
            prev_degree = prev_degrees.get(node, 0)
            delta_e = edge_changes[node]
            if prev_degree == 0 and delta_e == 0:
                score = 0
            elif prev_degree == 0 and delta_e != 0:
                score = gamma * (min(lambda_param * (delta_e / math.sqrt(delta_e - 0.5)), max_exp_input))
            else:
                score = gamma * (min(lambda_param * (delta_e / math.sqrt(prev_degree + 2)), max_exp_input))
            node_change_scores[node] = score

        if node_change_scores:
            unique_scores = sorted(set(node_change_scores.values()))
            num_unique_scores = len(unique_scores)
            score_to_normalized = {
                score: idx / (num_unique_scores - 1) if num_unique_scores > 1 else 0.5
                for idx, score in enumerate(unique_scores)
            }
            node_change_scores = {
                node: score_to_normalized[score] for node, score in node_change_scores.items()
            }
            node_change_scores = {
                node: a + (b - a) * normalized_score for node, normalized_score in node_change_scores.items()
            }

        node_change_scores_all.append(node_change_scores)
        print(f"Snapshot {t}: Computed and normalized node change scores for {len(node_change_scores)} nodes")

    return node_change_scores_all
