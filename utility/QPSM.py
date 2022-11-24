# -*- coding: utf-8 -*-
# QUADRATIC PROBLEM ON THE STIEFEL MANIFOLD
# A generalized power iteration method for solving
# quadratic problem on the Stiefel manifold
# Feiping Nie, Rui Zhang, and Xuelong Li, Fellow IEEE

import numpy as np


def QPSM(A, B, verbose):
    """

    :param A: m,m
    :param B: m,k
    :param u:
    :param W: m,k
    :return:
    """
    m = A.shape[0]
    k = B.shape[1]
    W = np.zeros((m, k))
    W_old = np.zeros_like(W)
    u = np.max(np.real(np.linalg.eigvals(A)))
    A_ = u * np.eye(A.shape[0]) - A
    max_iter = 30
    for i in range(max_iter):
        M = 2 * np.dot(A_, W) + 2 * B
        W_old = W
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        W = np.dot(U, Vt)
        loss = np.linalg.norm(W - W_old)
        if verbose ==True:
            print(i, loss)
        if loss < 0.0001:
            break
    return W
