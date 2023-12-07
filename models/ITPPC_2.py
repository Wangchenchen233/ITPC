# -*- coding: utf-8 -*-
# @Time    : 2022/11/3 18:09
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : ITPPC_2.py
from sklearn.metrics import pairwise_distances

from models.ITPPC import ITPPC
from models.ITPPC_wo_R import ITPPC_wo_R
import numpy as np

from utility.class_corr_utils import label2matrix
from utility.subfunc import estimateReg
from utility.unsupervised_evaluation import cluster_evaluation_cluster, best_map
from sklearn.cluster import KMeans


def calculate_R(X, F, n_classes):
    x_center = np.dot(np.dot(X, F), np.diag(1 / F.sum(0)))
    Dist_c = pairwise_distances(x_center.T) ** 2
    _, C = estimateReg(Dist_c, int(n_classes - 2))
    C = (C + C.T) / 2
    Lc = np.diag(C.sum(0)) - C
    return C, Lc


def ITPPC_2(X, Ls0, F0, Local_reg, k, d, n_classes, para):
    """

    :param X:d,n
    :param Ls0:
    :param F0:
    :param Local_reg:
    :param k: 近邻个数
    :param d: 低维维度
    :param n_classes:类别个数
    :param para:
    :return:
    """
    lamb1, lamb2 = para
    Ls = Ls0
    # init
    Z, F, S = ITPPC_wo_R(X, Ls, F0, Local_reg, True, k, d, n_classes, lamb2)
    k_means = KMeans(n_clusters=n_classes, tol=0.0001)
    k_means.fit(F)
    y_predict = k_means.labels_ + 1
    label_matrix, _, _ = label2matrix(y_predict)
    C, Lc = calculate_R(X, label_matrix, n_classes)

    maxiter = 1
    for iter_ in range(maxiter):
        # calculate Y
        Z, F, S_l = ITPPC(X, Ls, Lc, F, Local_reg, True, k, d, n_classes, para)

    return F, S_l
