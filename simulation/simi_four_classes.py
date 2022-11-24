# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import sklearn

from utility.class_corr_utils import label2matrix, cal_center_corr
from utility.subfunc import estimateReg, eig_lastk

from baseline.PCAN import PCAN
from baseline.SOGFS import SOGFS
from baseline.PCAN_orth import PCAN_orth
from models.ITPPC import ITPPC

np.random.seed(233)


def generate_multi_class(n):
    # generate the first class
    x1 = np.random.multivariate_normal([1, 1], np.diag([0.05, 0.05]), n).T  # (2,100) gaussian distribution
    x1 = np.vstack((x1, np.random.rand(n) * 2))  # [0,2] uniform distribution
    x1 = np.vstack((x1, np.random.rand(n) * 2 + 1))  # [1,3] uniform distribution
    y1 = np.ones(n)

    # generate the second class
    x2 = np.random.multivariate_normal([3, 1], np.diag([0.05, 0.05]), n).T  # (2,100) gaussian distribution
    x2 = np.vstack((x2, np.random.rand(n) * 2))  # [0,2]
    x2 = np.vstack((x2, np.random.rand(n) * 2 + 1))  # [1,3]
    y2 = np.ones(n) * 2

    # generate the third class
    x3 = np.random.multivariate_normal([2, 2], np.diag([0.05, 0.05]), n).T  # (2,100) gaussian distribution
    x3 = np.vstack((x3, np.random.rand(n) * 2))  # [0,2]
    x3 = np.vstack((x3, np.random.rand(n) * 2 + 1))  # [1,3]
    y3 = np.ones(n) * 3

    # generate the four class
    x4 = np.random.multivariate_normal([2, 4], np.diag([0.05, 0.05]), n).T  # (2,100) gaussian distribution
    x4 = np.vstack((x4, np.random.rand(n) * 2))  # [0,2]
    x4 = np.vstack((x4, np.random.rand(n) * 2 + 1))  # [1,3]
    y4 = np.ones(n) * 4

    x12 = np.hstack((x1, x2))
    y12 = np.hstack((y1, y2))

    x123 = np.hstack((x12, x3))
    y123 = np.hstack((y12, y3))
    x = np.hstack((x123, x4)).T  # (n, d)
    y = np.hstack((y123, y4)).T  # (n, )

    return x, y


def low_dim_plot(ax, X_dr, y, n_classes, mode=''):
    colors = ['red', 'black', 'orange', 'blue']
    for i in range(n_classes):
        plt.scatter(X_dr[y == i, 0], X_dr[y == i, 1], alpha=.7, c=colors[i])
    plt.title(mode)


def low_plot_row(x, x_pcan, x_sogfs, x_pcano, x_model, y, n_classes):
    ax = plt.figure(figsize=[22, 4])
    plt.subplot(151)
    low_dim_plot(ax, x[:, :2], y, n_classes, mode='Real distribution')

    plt.subplot(152)
    low_dim_plot(ax, x_pcan, y, n_classes, mode='PCAN')

    plt.subplot(153)
    low_dim_plot(ax, x_sogfs, y, n_classes, mode='SOGFS')

    plt.subplot(154)
    low_dim_plot(ax, x_pcano, y, n_classes, mode='PPC w/o all')

    plt.subplot(155)
    low_dim_plot(ax, x_model, y, n_classes, mode='PPC')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X, y = generate_multi_class(100)
    np.save('4class.npy', X)
    y = y - 1
    label_matrix, class_distribute, n_classes = label2matrix(y)
    X = sklearn.preprocessing.scale(X)

    Dist_x = pairwise_distances(X) ** 2
    Local_reg, S = estimateReg(Dist_x, 5)
    S = (S + S.T) / 2
    Ls = np.diag(S.sum(0)) - S

    F, _ = eig_lastk(Ls, n_classes)
    F = np.abs(F) + 0.001
    temp = np.diag(np.sqrt(np.diag(1 / (np.dot(F.transpose(), F) + 1e-16))))
    F = np.dot(F, temp)

    X_corr_euc = cal_center_corr(X[:, :2], label_matrix, class_distribute)
    Lc = np.diag(X_corr_euc.sum(0)) - X_corr_euc

    # PCAN
    W = PCAN(X, Ls, Local_reg, True, 5, 2, n_classes)
    X_pcan = np.dot(X, W)

    # SOGFS
    para = [1, 1, 2]
    W = SOGFS(X, Ls, F, Local_reg, True, 5, n_classes, para)
    X_sogfs = np.dot(X, W)

    # PCAN_orth
    W = PCAN_orth(X, Ls, Local_reg, True, 5, 2, n_classes)
    X_pcano = np.dot(X, W)

    # ITPPC
    para = (100, 10)
    X_itppc, F, S_model1 = ITPPC(X.T, Ls, Lc, F, Local_reg, True, 5, 2, n_classes, para)

    low_plot_row(X, X_pcan, X_sogfs, X_pcano, X_itppc.T, y, n_classes)
