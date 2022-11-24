# -*- coding: utf-8 -*-
# @Time    : 2022/11/3 18:30
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : para_init.py
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from utility.subfunc import estimateReg, eig_lastk


def para_init(X, y, k, nclass):
    from simulation.simi_four_classes import label2matrix
    label_matrix, class_distribute, n_classes = label2matrix(y)

    # init S
    Dist_x = pairwise_distances(X) ** 2
    Local_reg, S = estimateReg(Dist_x, k)
    S = (S + S.T) / 2
    Ls = np.diag(S.sum(0)) - S

    # init C
    x_center = np.dot(np.dot(X.T, label_matrix), np.diag(1 / label_matrix.sum(0)))
    Dist_c = pairwise_distances(x_center.T) ** 2
    _, C = estimateReg(Dist_c, nclass - 2)
    C = (C + C.T) / 2
    Lc = np.diag(C.sum(0)) - C

    # init F
    F, E_val = eig_lastk(Ls, n_classes)
    if np.sum(E_val[:n_classes + 1]) < 10e-11:
        print("already c connected component")
    F = np.abs(F) + 0.001
    temp = np.diag(np.sqrt(np.diag(1 / (np.dot(F.transpose(), F) + 1e-16))))
    F = np.dot(F, temp)
    return S, Ls, C, Lc, F, Local_reg, n_classes

