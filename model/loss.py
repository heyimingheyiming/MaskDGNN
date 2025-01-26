#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数接口
import numpy as np  # 导入NumPy库
import torchmetrics  # 导入PyTorch的评估指标库
from model.config import cfg  # 从配置文件中导入cfg
from sklearn.metrics import average_precision_score  # 从sklearn中导入平均精度评分函数
from sklearn.metrics import f1_score, accuracy_score  # 从sklearn中导入F1评分和准确率评分函数
from sklearn.metrics import roc_auc_score  # 从sklearn中导入ROC AUC评分函数

def prediction(pred_score, true_l):  # 定义函数，用于计算预测结果的各项指标
    # Acc_torch = torchmetrics.Accuracy(task='binary').to(pred_score)
    # Macro_Auc_torch = torchmetrics.AUROC(task='binary', average='macro').to(pred_score)
    # Micro_Auc_torch = torchmetrics.AUROC(task='binary', average='micro').to(pred_score)
    # Ap_torch = torchmetrics.AveragePrecision(task='binary').to(pred_score)
    # F1_torch = torchmetrics.F1Score(task='binary', average='macro').to(pred_score)
    #
    # acc_torch = Acc_torch(pred_score, true_l.to(pred_score))
    # macro_auc_torch = Macro_Auc_torch(pred_score, true_l.to(pred_score))
    # micro_auc_torch = Micro_Auc_torch(pred_score, true_l.to(pred_score))
    # ap_torch = Ap_torch(pred_score, true_l.to(pred_score))
    # f1_torch = F1_torch(pred_score, true_l.to(pred_score))

    pred = pred_score.clone()  # 克隆预测分数
    pred = torch.where(pred > 0.5, 1, 0)  # 将预测分数大于0.5的置为1，否则置为0
    pred = pred.detach().cpu().numpy()  # 将预测结果转换为NumPy数组
    pred_score = pred_score.detach().cpu().numpy()  # 将预测分数转换为NumPy数组

    # true = np.ones_like(pred)
    true = true_l  # 获取真实标签
    true = true.cpu().numpy()  # 将真实标签转换为NumPy数组
    acc = accuracy_score(true, pred)  # 计算准确率
    ap = average_precision_score(true, pred_score)  # 计算平均精度
    f1 = f1_score(true, pred, average='macro')  # 计算F1评分（宏平均）
    macro_auc = roc_auc_score(true, pred_score, average='macro')  # 计算ROC AUC评分（宏平均）
    micro_auc = roc_auc_score(true, pred_score, average='micro')  # 计算ROC AUC评分（微平均）

    # print(acc, ap, f1, macro_auc, micro_auc)
    # print(acc_torch, ap_torch, f1_torch, macro_auc_torch, micro_auc_torch)
    return acc, ap, f1, macro_auc, micro_auc  # 返回各项指标
    # return acc_torch, ap_torch, f1_torch, macro_auc_torch, micro_auc_torch


def Link_loss_meta(pred, y):  # 定义函数，用于计算链接预测的损失
    L = nn.BCELoss()  # 创建二元交叉熵损失函数
    pred = pred.float()  # 将预测结果转换为浮点型
    y = y.to(pred)  # 将真实标签转换为与预测结果相同的设备和数据类型
    loss = L(pred, y)  # 计算损失

    return loss  # 返回损失


