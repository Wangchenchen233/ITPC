# -*- coding: utf-8 -*-
# @Time    : 2022/10/17 11:05
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : model1_main.py
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import math
import sklearn

from models import ITPPC, ITPPC_2, para_init

from utility.subfunc import local_structure_learning, estimateReg, eig_lastk

from utility.unsupervised_evaluation import dataset_pro, cluster_evaluation_cluster
import os
from tqdm import tqdm
import traceback

eps = np.spacing(1)

if __name__ == '__main__':

    Data_names = ['lung_discrete']
    Cluster_times = 20
    k = 5
    Dims = np.arange(10, 101, 10)

    for data_name in ['lung_discrete']:
        X, y, classes = dataset_pro(data_name, '')

        n_features = X.shape[1]
        S, Ls, C, Lc, F, Local_reg, n_classes = para_init.para_init(X, y, k, classes)

        paras = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        grid_search = [(lamb1, lamb2) for lamb1 in paras for lamb2 in paras]

        # result in diff para
        nmi_result = np.zeros((len(Dims), len(grid_search)))
        acc_result = np.zeros((len(Dims), len(grid_search)))
        std_result_all = np.zeros((len(Dims), 2 * len(grid_search)))
        ii = 0

        for d in tqdm(Dims):
            d = int(d)
            # print(d)
            kk = 0
            nmi_para_all = np.zeros(len(grid_search))
            acc_para_all = np.zeros(len(grid_search))
            std_all = np.zeros((len(grid_search), 2))
            for para in grid_search:  # tqdm(grid_search):

                F, S_l = ITPPC_2.ITPPC_2(X.T, Ls, F, Local_reg, k, d, n_classes, para)
                # Z, F, S_l = ITPPC.ITPPC(X.T, Ls, Lc, F, Local_reg, True, k, d, n_classes, para)

                # store K-means result with ave 20-times
                nmi_para_temp, acc_para_temp = cluster_evaluation_cluster(F, y, classes, Cluster_times)

                nmi_para_all[kk] = nmi_para_temp[0]
                acc_para_all[kk] = acc_para_temp[0]
                std_all[kk] = np.array([nmi_para_temp[1], acc_para_temp[1]])
                kk += 1
            nmi_result[ii, :] = nmi_para_all
            acc_result[ii, :] = acc_para_all
            std_result_all[ii] = std_all.flatten()  # 保存所有para的方差(nmi_std,acc_std)
            ii += 1
        nmi_result = acc_result
        print("nmi", np.max(nmi_result, 1))
        print("acc", np.max(acc_result, 1))
