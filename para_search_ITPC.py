import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
# from PreProClu.PPC import PPC
from models.ITPC import ITPC
from models.para_init import para_init
from utility.unsupervised_evaluation import dataset_pro, cluster_evaluation_cluster
import os
from tqdm import tqdm

eps = np.spacing(1)

if __name__ == '__main__':
    Data_names = ['lung_discrete']
    Cluster_times = 20

    for data_name in Data_names:
        Dims = np.arange(10, 101, 10)
        k = 5
        print(data_name, k)

        X, y, classes = dataset_pro(data_name, '')
        # import numpy as np

        unique_label = np.unique(y)
        classes = unique_label.shape[0]
        n_features = X.shape[1]
        S, Ls0, C, Lc, F0, Local_reg, n_classes = para_init(X, y, k, classes)
        F, S_l = ITPC(X.T, Ls0, Lc, F0, Local_reg, True, k, n_classes, (1, 1))

        paras = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        grid_search = [(lamb1, lamb2) for lamb1 in paras for lamb2 in paras]

        kk = 0
        nmi_para_all = np.zeros(len(grid_search))
        acc_para_all = np.zeros(len(grid_search))
        std_all = np.zeros((len(grid_search), 2))
        for para in tqdm(grid_search):
            F, S_l = ITPC(X.T, Ls0, Lc, F0, Local_reg, True, k, n_classes, para)
            # store K-means result with ave 20-times
            nmi_para_temp, acc_para_temp = cluster_evaluation_cluster(F, y, classes, Cluster_times)
            nmi_para_all[kk] = nmi_para_temp[0]
            acc_para_all[kk] = acc_para_temp[0]
            std_all[kk] = np.array([nmi_para_temp[1], acc_para_temp[1]])
            kk += 1
        print("nmi:", np.max(nmi_para_all))
        print("acc:", np.max(acc_para_all))
        resulr = nmi_para_all
