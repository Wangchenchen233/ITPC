# -*- coding: utf-8 -*-
# @Time    : 2022/11/3 18:28
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : models.py
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from utility.subfunc import eig_lastk, local_structure_learning_ppc
from utility import QPSM

eps = np.spacing(1)


def ITPC(X, Ls0, Lc, F0, local_reg, islocal, k, r, para):
    """
    solving model in 2022.10.10
    # sum_ij S_ij||x_i -x_j|| + lamb1 sum_ij C_ij ||h_i-h_j||
    # +lamb2 sum_ij F_ij||x_i-h_j|| + lamb3 trace(F^TLsF)
    X: D,N feature matrix
    W: D,d projection matrix
    Z: d,N W^TX
    C: r,r class corr
    H: d,r latent represent
    F: N,r label matrix
    F0: init label matrix
    Ls: N,N sample similarity matrix
    local_reg, islocal, k: learning S
    d: low space dimension
    r: class dimension
    para: lamb1,lamb2
    gamma: constrain F^TF=I
    """
    nclass = F0.shape[1]
    lamb1, lamb2 = para
    lamb3 = local_reg
    gamma = 1e8

    # init
    F = F0
    F_old = np.zeros_like(F)
    f = np.zeros_like(F)
    S = np.zeros_like(Ls0)

    iter_max = 30
    obj_history = np.zeros(iter_max)
    for iter_ in range(iter_max):
        # update S
        if iter_ == 0:
            Ls = Ls0
        else:
            dist_x = pairwise_distances(X.T) ** 2
            dist_f = pairwise_distances(F) ** 2
            S = local_structure_learning_ppc(k, lamb3, dist_x, dist_f, islocal)
            # calculate ls
            Ls = np.diag(S.sum(0)) - S

        # update H
        h_l = np.dot(X, F)
        LambH = np.diag(F.sum(0))
        hinv = np.linalg.inv(2 * lamb1 / lamb2 * Lc + LambH + 1e-8 * np.eye(nclass))
        H = np.dot(h_l, hinv)  # d,r

        # update F
        F_old = F
        F_up = 2 * gamma * F
        Phi = pairwise_distances(X.T, H.T) ** 2  # N, r
        F_down = 4 * lamb3 * np.dot(Ls, F) + lamb2 * Phi + 2 * gamma * np.dot(np.dot(F, F.T), F)
        F = F * F_up / F_down

        temp = np.diag(np.sqrt(np.diag(1 / (np.dot(F.transpose(), F) + 1e-16))))
        F = np.dot(F, temp)

        f, e_val = eig_lastk(Ls, r)

        fn1 = np.sum(e_val[:r])
        fn2 = np.sum(e_val[:r + 1])
        if fn1 > 10e-10:
            lamb3 *= 2
            lamb3 = min(lamb3, 1e8)
        elif fn2 < 10e-10:
            lamb3 /= 2
            F = F_old
        else:
            break
    return f, S
