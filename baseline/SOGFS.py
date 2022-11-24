"""
Description: TKDE21 structure graph optimizaiton for unsupervised
"""
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from utility.subfunc import local_structure_learning, eig_lastk

eps = np.spacing(1)


def SOGFS(x, ls0, f0, local_reg, islocal, k, c, para):
    """
    TKDE21 structure graph optimization for unsupervised feature selection
    :param para:
    :param ls:
    :param f:
    :param local_reg:
    :param x: n d
    :param c: cluster number
    :return: w
    """
    gamma, p, m = para
    n_sample, n_feature = x.shape
    ls = ls0
    f = f0
    lambda_ = local_reg
    w = np.zeros((n_feature, m))

    iter_max = 30
    evals_all = []
    for iter_ in range(iter_max):
        # update W
        q = np.eye(n_feature)
        xlx = np.dot(np.dot(x.T, ls), x)
        lsq = xlx + gamma * q
        obj1 = []
        for iter_i in range(20):
            w, _ = eig_lastk(lsq, m)
            q_temp = (w * w).sum(1) + eps
            q = np.diag(np.power(q_temp, (p - 2) / 2) * p / 2)
            lsq = xlx + gamma * q
            obj1_temp = np.trace(np.dot(np.dot(w.T, lsq), w))
            obj1.append(obj1_temp)
            if iter_i > 1 and abs(obj1[iter_i] - obj1[iter_i - 1]) < 1e-3:
                # print("--inner conv", iter_i)
                break

        # update s
        x2 = np.dot(x, w)
        dist_x = pairwise_distances(x2) ** 2
        f_old = f
        dist_f = lambda_ * pairwise_distances(f) ** 2
        s = local_structure_learning(k, lambda_, dist_x, dist_f, islocal)

        # calculate ls
        ls = np.diag(s.sum(0)) - s
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
