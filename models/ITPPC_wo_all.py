# -*- coding: utf-8 -*-
# @Time    : 2022/11/3 18:26
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : ITPPC_wo_all.py

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from utility.subfunc import eig_lastk, local_structure_learning_ppc

eps = np.spacing(1)


def PPC_wo_RH(x, ls0, local_reg, islocal, k, d, c):
    """

    :param x:
    :param ls0: 初始的LS
    :param local_reg: 参数
    :param islocal:是否局部方法
    :param k: 近邻数
    :param d: 降维后维度
    :param c: 类别
    :return:
    """
    lambda_ = local_reg
    w, _ = eig_lastk(np.dot(np.dot(x.T, ls0), x), d)

    iter_max = 30
    evals_all = []
    for iter_ in range(iter_max):
        if iter_ == 0:
            ls = ls0
        else:
            # update s
            x2 = np.dot(x, w)
            dist_x = pairwise_distances(x2) ** 2
            f_old = f
            dist_f = pairwise_distances(f) ** 2
            s = local_structure_learning_ppc(k, lambda_, dist_x, dist_f, islocal)
            # calculate ls
            ls = np.diag(s.sum(0)) - s

        # update W
        xlx = np.dot(np.dot(x.T, ls), x)
        w, _ = eig_lastk(xlx, d)

        # update F
        f, e_val = eig_lastk(ls, c)
        evals_all.append(e_val)

        fn1 = np.sum(e_val[:c])
        fn2 = np.sum(e_val[:c + 1])
        if fn1 > 10e-10:
            lambda_ *= 2
            lambda_ = min(lambda_, 1e8)
        elif fn2 < 10e-10:
            lambda_ /= 2
            f = f_old
        else:
            break
    return w, f, s
