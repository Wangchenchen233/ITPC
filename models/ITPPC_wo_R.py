# -*- coding: utf-8 -*-
# @Time    : 2022/11/3 18:22
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : ITPPC_wo_R.py
from sklearn.metrics import pairwise_distances
from utility import QPSM
from utility.subfunc import eig_lastk, local_structure_learning_ppc
import numpy as np


def ITPPC_wo_R(X, Ls0, F0, local_reg, islocal, k, d, r, lamb2):
    """
    solving model in 2022.10.10
    # sum_ij S_ij||W^Tx_i -W^Tx_j||
    # +lamb2 sum_ij F_ij||W^Tx_i-h_j|| + lamb3 trace(F^TLsF)
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
    u: for solve W, in QPSM
    """
    nclass = F0.shape[1]
    lamb3 = local_reg
    gamma = 1e8

    W, _ = eig_lastk(np.dot(np.dot(X, Ls0), X.T), d)
    Z = np.dot(W.T, X)
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
            dist_x = pairwise_distances(Z.T) ** 2
            dist_f = pairwise_distances(F) ** 2
            S = local_structure_learning_ppc(k, lamb3, dist_x, dist_f, islocal)
            # calculate ls
            Ls = np.diag(S.sum(0)) - S

        # update H
        h_l = np.dot(Z, F)
        LambH = np.diag(F.sum(0))
        hinv = np.linalg.inv(LambH + 1e-8 * np.eye(nclass))
        H = np.dot(h_l, hinv)  # d,r

        # update F
        F_old = F
        F_up = 2 * gamma * F
        Phi = pairwise_distances(Z.T, H.T) ** 2  # N, r
        F_down = 4 * lamb3 * np.dot(Ls, F) + lamb2 * Phi + 2 * gamma * np.dot(np.dot(F, F.T), F)
        F = F * F_up / F_down

        temp = np.diag(np.sqrt(np.diag(1 / (np.dot(F.transpose(), F) + 1e-16))))
        F = np.dot(F, temp)

        # update W
        LambW = np.diag(F.sum(1))
        Delta1 = np.dot(np.dot(X, 2 * Ls + lamb2 * LambW), X.T)
        Delta2 = lamb2 * np.dot(np.dot(X, F), H.T)
        W = QPSM.QPSM(Delta1, Delta2, False)

        Z = np.dot(W.T, X)

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
    return Z, f, S
