# -*- coding: utf-8 -*-
# PCAN
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from utility.subfunc import local_structure_learning, eig_lastk

eps = np.spacing(1)


def PCAN(x, ls0, local_reg, islocal, k, d, c):
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
    n_sample, n_feature = x.shape

    # init ls
    lambda_ = local_reg

    h = np.eye(n_sample) - np.ones((n_sample, n_sample)) / n_sample
    st = np.dot(np.dot(x.T, h), x)
    invst = np.linalg.inv(st)

    # init w
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
            dist_f = lambda_ * pairwise_distances(f) ** 2
            s = local_structure_learning(k, local_reg, dist_x, dist_f, islocal)
            # calculate ls
            ls = np.diag(s.sum(0)) - s

        # update W
        xlx = np.dot(np.dot(x.T, ls), x)
        sxlx = np.dot(invst, xlx)
        w, _ = eig_lastk(sxlx, d)

        # update F
        f, e_val = eig_lastk(ls, c)
        evals_all.append(e_val)

        fn1 = np.sum(e_val[:c])
        fn2 = np.sum(e_val[:c + 1])
        if fn1 > 10e-10:
            lambda_ = 2 * lambda_
            lambda_ = min(lambda_, 1e8)
        elif fn2 < 10e-10:
            lambda_ = lambda_ / 2
            f = f_old
        else:
            break
    return w
