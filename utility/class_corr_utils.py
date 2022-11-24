# -*- coding: utf-8 -*-
# @Time    : 2022/10/17 15:35
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : class_corr_utils.py
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def label2matrix(label):
    n_samples = label.shape[0]
    unique_label = np.unique(label)
    n_classes = unique_label.shape[0]
    label_matrix = np.zeros((n_samples, n_classes))

    class_distribute = pd.DataFrame(label).value_counts()
    for i in range(n_classes):
        label_matrix[label == unique_label[i], i] = 1
    return label_matrix, class_distribute, n_classes


def cal_center_corr(x, label_matrix, class_distribute):
    # 计算类间关联矩阵C 热核和cos
    x_center = np.dot(np.dot(x.T, label_matrix), np.diag(1 / class_distribute))  # 计算类中心，注意标签顺序
    d = pairwise_distances(x_center.T)
    d **= 2
    t = np.mean(d)
    x_heat_kernel = np.exp(-d / t)
    return x_heat_kernel
