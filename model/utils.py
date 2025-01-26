#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import dgl  # 导入DGL库，用于图神经网络的操作
import numpy as np  # 导入NumPy库，用于数组操作
import torch  # 导入PyTorch库，用于张量操作和深度学习
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch import optim as optim  # 从PyTorch中导入优化器模块
from model.config import cfg  # 从配置文件中导入cfg对象，包含模型的配置信息
from torch_scatter import scatter_max, scatter_mean, scatter_min  # 从torch_scatter库中导入scatter_max, scatter_mean, scatter_min函数，用于按索引聚合张量
from model.loss import prediction, Link_loss_meta  # 从model.loss模块中导入prediction和Link_loss_meta函数，用于计算损失

def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):  # 定义create_optimizer函数，用于创建优化器
    opt_lower = opt.lower()  # 将优化器名称转换为小写

    parameters = model.parameters()  # 获取模型的参数
    opt_args = dict(lr=lr, weight_decay=weight_decay)  # 创建包含学习率和权重衰减的字典

    opt_split = opt_lower.split("_")  # 按下划线分割优化器名称
    opt_lower = opt_split[-1]  # 获取分割后的最后一个部分
    if opt_lower == "adam":  # 如果优化器是Adam
        optimizer = optim.Adam(parameters, **opt_args)  # 创建Adam优化器
    elif opt_lower == "adamw":  # 如果优化器是AdamW
        optimizer = optim.AdamW(parameters, **opt_args)  # 创建AdamW优化器
    elif opt_lower == "adadelta":  # 如果优化器是Adadelta
        optimizer = optim.Adadelta(parameters, **opt_args)  # 创建Adadelta优化器
    elif opt_lower == "radam":  # 如果优化器是RAdam
        optimizer = optim.RAdam(parameters, **opt_args)  # 创建RAdam优化器
    elif opt_lower == "sgd":  # 如果优化器是SGD
        opt_args["momentum"] = 0.9  # 设置动量参数为0.9
        return optim.SGD(parameters, **opt_args)  # 创建SGD优化器并返回
    else:  # 如果优化器名称不匹配
        assert False and "Invalid optimizer"  # 抛出异常，提示无效优化器

    return optimizer  # 返回创建的优化器

def create_activation(activation):  # 定义create_activation函数，用于创建激活函数
    if activation == "relu":  # 如果激活函数是ReLU
        return nn.ReLU()  # 返回ReLU激活函数
    elif activation == "tanh":  # 如果激活函数是Tanh
        return nn.Tanh()  # 返回Tanh激活函数
    elif activation == "sigmoid":  # 如果激活函数是Sigmoid
        return nn.Sigmoid()  # 返回Sigmoid激活函数
    elif activation == "leaky_relu":  # 如果激活函数是Leaky ReLU
        return nn.LeakyReLU()  # 返回Leaky ReLU激活函数
    elif activation == "elu":  # 如果激活函数是ELU
        return nn.ELU()  # 返回ELU激活函数
    elif activation == "softmax":  # 如果激活函数是Softmax
        return nn.Softmax()  # 返回Softmax激活函数

    raise RuntimeError("activation error, not {}".format(activation))  # 如果激活函数名称不匹配，抛出异常

def edge_index_difference(edge_all, edge_except, num_nodes):  # 定义edge_index_difference函数，用于计算边索引的差异
    idx_all = edge_all[0] * num_nodes + edge_all[1]  # 计算所有边的索引
    idx_except = edge_except[0] * num_nodes + edge_except[1]  # 计算要排除的边的索引
    # 过滤掉在idx_except中的边。
    mask = torch.from_numpy(np.isin(idx_all, idx_except)).to(torch.bool)  # 创建掩码，标记要排除的边
    # mask = torch.isin(idx_all, idx_except)
    idx_kept = idx_all[~mask]  # 保留不在掩码中的边
    i = idx_kept // num_nodes  # 计算保留边的源节点索引
    j = idx_kept % num_nodes  # 计算保留边的目标节点索引
    return torch.stack([i, j], dim=0).long()  # 返回保留边的索引

def gen_negative_edges(edge_index, num_neg_per_node, num_nodes):  # 定义gen_negative_edges函数，用于生成负边
    src_lst = torch.unique(edge_index[0])  # 获取唯一的源节点索引
    num_neg_per_node = int(1.5 * num_neg_per_node)  # 增加一些冗余，计算每个节点的负边数量
    i = src_lst.repeat_interleave(num_neg_per_node)  # 将源节点索引重复指定次数
    # nodes = torch.unique(edge_index.flatten())
    # nodes = nodes.cpu().numpy()
    j = torch.Tensor(np.random.choice(num_nodes, len(i), replace=True))  # 随机选择目标节点索引，生成负边候选
    # 候选负边，每个源节点有X个候选目标节点。
    candidates = torch.stack([i, j], dim=0).long()  # 将源节点和目标节点索引堆叠为边索引

    # 过滤掉候选中的正边。
    neg_edge_index = edge_index_difference(candidates, edge_index.to('cpu'), num_nodes)  # 过滤掉正边，保留负边
    return neg_edge_index  # 返回负边索引


@torch.no_grad()  # 使用装饰器，表示在此函数中不计算梯度
def fast_batch_mrr_and_recall(edge_label_index, edge_label, pred_score, num_neg_per_node, num_nodes):  # 定义函数，用于计算MRR和Recall

    src_lst = torch.unique(edge_label_index[0], sorted=True)  # 获取唯一的源节点索引，并排序
    num_users = len(src_lst)  # 计算用户数量

    edge_pos = edge_label_index[:, edge_label == 1]  # 获取正样本的边索引
    edge_neg = edge_label_index[:, edge_label == 0]  # 获取负样本的边索引

    # 通过构建，负边索引应该按源节点排序。
    assert torch.all(edge_neg[0].sort()[0] == edge_neg[0])  # 验证负边的源节点是排序的

    # 所有正样本和负样本的预测分数。
    p_pos = pred_score[edge_label == 1]  # 获取正样本的预测分数
    p_neg = pred_score[edge_label == 0]  # 获取负样本的预测分数

    # 对于每个源节点，计算所有正样本边中的最高分数
    # 我们想计算这条边的排名。
    # 构建模型性能的区间。
    if cfg.metric.mrr_method == 'mean':  # 如果MRR方法是mean
        best_p_pos = scatter_mean(src=p_pos, index=edge_pos[0], dim_size=num_nodes)  # 计算每个源节点的平均分数
    elif cfg.metric.mrr_method == 'min':  # 如果MRR方法是min
        best_p_pos, _ = scatter_min(src=p_pos, index=edge_pos[0], dim_size=num_nodes)  # 计算每个源节点的最小分数
    else:  # 默认设置，考虑最自信边的排名。
        best_p_pos, _ = scatter_max(src=p_pos, index=edge_pos[0], dim_size=num_nodes)  # 计算每个源节点的最大分数
    # best_p_pos的形状为(num_nodes)，对于不在src_lst中的节点，其值为0。
    # 取出了节点上的最大概率
    best_p_pos_by_user = best_p_pos[src_lst]  # 获取源节点列表中的最大分数

    # 合理性检查。
    # src_lst_2, inverse = torch.unique(edge_pos[0], return_inverse=True)
    # best_p_pos, _ = scatter_max(p_pos, inverse)
    # assert torch.all(best_p_pos_by_user == best_p_pos)

    uni, counts = torch.unique(edge_neg[0], sorted=True, return_counts=True)  # 获取唯一的负样本源节点及其计数
    # assert torch.all(counts >= num_neg_per_node)
    # assert torch.all(uni == src_lst)
    # 注意：edge_neg (src, dst)按src排序。
    # 查找每个源节点在edge_neg[0]中的第一次出现的索引。
    # 负边[0], [1,1,...1, 2, 2, ... 2, 3, ..]
    first_occ_idx = torch.cumsum(counts, dim=0) - counts  # 计算每个源节点第一次出现的索引
    add = torch.arange(num_neg_per_node, device=first_occ_idx.device)  # 创建一个从0到num_neg_per_node-1的张量

    # 从每个源节点中取出前100个负样本。
    score_idx = first_occ_idx.view(-1, 1) + add.view(1, -1)  # 计算前100个负样本的索引

    assert torch.all(edge_neg[0][score_idx].float().std(axis=1) == 0)  # 验证每个源节点的负样本索引是正确的
    # Z = edge_neg[0][first_occ_idx - 1]
    # A = edge_neg[0][first_occ_idx]
    # B = edge_neg[0][first_occ_idx + 1]
    # assert torch.all(Z != A)
    # assert torch.all(B == A)
    # 前100个负样本的预测分数
    p_neg_by_user = p_neg[score_idx]  # 获取前100个负样本的预测分数 (num_users, num_neg_per_node)
    # 比较100个负样本和正样本最大概率
    compare = (p_neg_by_user >= best_p_pos_by_user.view(num_users, 1)).float()  # 比较每个负样本和正样本的最大分数
    assert compare.shape == (num_users, num_neg_per_node)  # 验证compare的形状

    # 计算每个源节点的负样本中有多少个分数大于正样本的最大分数。
    # 如果没有这样的负样本，排名为1。
    # 所有源节点的负样本大于正样本的统计
    rank_by_user = compare.sum(axis=1) + 1  # 计算每个源节点的排名 (num_users,)
    assert rank_by_user.shape == (num_users,)  # 验证rank_by_user的形状

    mrr = float(torch.mean(1 / rank_by_user))  # 计算MRR
    # print(f'MRR={mrr}, time taken: {datetime.now() - start}')
    # 计算Recall@k
    recall_at = dict()  # 创建字典存储Recall@k
    for k in [1, 3, 10]:  # 遍历k值
        recall_at[k] = float((rank_by_user <= k).float().mean())  # 计算Recall@k

    return mrr, recall_at  # 返回MRR和Recall@k


@torch.no_grad()  # 使用装饰器，表示在此函数中不计算梯度
def report_rank_based_eval_meta(model, graph, x, fast_weights, num_neg_per_node: int = 1000):  # 定义函数，用于报告基于排名的评估结果num_neg_per_node: int = 1000
    if num_neg_per_node == -1:  # 如果负样本数量为-1
        # 不报告基于排名的指标，用于调试模式。
        return 0, 0, 0, 0  # 返回四个0
    # 获取正样本的边索引。
    edge_index = graph.edge_label_index[:, graph.edge_label == 1]  # 获取正样本的边索引

    neg_edge_index = gen_negative_edges(edge_index, num_neg_per_node, num_nodes=graph.num_nodes())  # 生成负样本的边索引

    new_edge_label_index = torch.cat((edge_index, neg_edge_index), dim=1)  # 将正样本和负样本的边索引拼接在一起
    new_edge_label = torch.cat((torch.ones(edge_index.shape[1]),  # 创建新的边标签，正样本为1，负样本为0
                                torch.zeros(neg_edge_index.shape[1])
                                ), dim=0)

    # 构建评估样本。
    graph.edge_label_index = new_edge_label_index.to('cpu').long()  # 将新的边索引转换为CPU上的长整型张量
    graph.edge_label = new_edge_label.to('cpu').long()  # 将新的边标签转换为CPU上的长整型张量

    # 将状态移动到GPU
    pred = model(graph, x, fast_weights)  # 使用模型进行预测   ???为什么在这里又要预测
    pred = pred.to('cpu')  # 将预测结果转换为CPU上的张量

    mrr, recall_at = fast_batch_mrr_and_recall(graph.edge_label_index, graph.edge_label,  # 计算MRR和Recall@k
                                               pred, num_neg_per_node, graph.num_nodes())
    #return 0,0,0,0
    return mrr, recall_at[1], recall_at[3], recall_at[10]  # 返回MRR和Recall@1, Recall@3, Recall@10


def rand_prop(graph):  # 定义函数，用于随机丢弃节点特征
    features = graph.node_feature  # 获取图的节点特征
    n = features.shape[0]  # 获取节点数量
    # 掩码
    drop_rate = cfg.dropnode_rate  # 获取节点丢弃率
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)  # 创建一个丢弃率的浮点型张量

    masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).to(features)  # 生成一个伯努利分布的掩码张量

    features = masks * features  # 将掩码应用到节点特征上

    return features  # 返回丢弃后的节点特征


def update_states(states, fast_weights):  # 定义函数，用于更新状态
    count = 0  # 初始化计数器
    for key in states.keys():  # 遍历状态字典的键
        assert isinstance(states[key], torch.Tensor)  # 确认状态值是张量
        states[key] = fast_weights[count]  # 更新状态值
        count += 1  # 增加计数器
    return states  # 返回更新后的状态


def paramters_(state):  # 定义函数，用于获取需要梯度的参数
    out = list()  # 初始化输出列表
    for key in state.keys():  # 遍历状态字典的键
        state[key].requires_grad = True  # 设置状态值需要梯度
        out.append(state[key])  # 将状态值添加到输出列表中
    return out  # 返回输出列表
