import numpy as np
# import sklearn.utils.linear_assignment_ as la
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import xlwt
import scipy.io as scio
import sklearn
import pandas as pd
import os
from tqdm import tqdm


def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    g = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            g[i, j] = np.count_nonzero(ss & tt)

    aa = linear_assignment(-g)
    aa = np.asarray(aa)
    aa = np.transpose(aa)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[aa[i][1]]] = label1[aa[i][0]]
    return new_l2.astype(int)


def evaluation(x_selected, n_clusters, y):
    """
    This function calculates ARI, ACC and NMI of clustering results

    Input
    -----
    X_selected: {numpy array}, shape (n_samples, n_selected_features}
            input data on the selected features
    n_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels

    Output
    ------
    nmi: {float}
        Normalized Mutual Information
    acc: {float}
        Accuracy

        k_means = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001, precompute_distances=True, verbose=0,
                     random_state=None, copy_x=True, n_jobs=1)
    """

    k_means = KMeans(n_clusters=n_clusters, tol=0.0001)
    k_means.fit(x_selected)
    y_predict = k_means.labels_ + 1

    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict)

    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)

    return nmi, acc


def evaluation_nmi_acc(y_predict, y):
    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict)

    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)
    return nmi, acc


def dataset_pro(data_name, methods):
    """
    数据处理和生成
    :param data_name: .m file, 'X': n,d; 'Y': n,1
    :return: 'X': n,d; 'Y': n,1
    """
    data_old = scio.loadmat(data_name + '.mat')
    label = data_old["Y"].astype('int')  # n 1
    unique_label = np.unique(label)
    classes = unique_label.shape[0]
    if methods == 'minmax':
        minmaxscaler = sklearn.preprocessing.MinMaxScaler()
        x = minmaxscaler.fit_transform(data_old["X"])
    else:
        x = sklearn.preprocessing.scale(data_old["X"])  # n d
    return x, label.reshape((label.shape[0],)), classes


def dataset_info(data_name, methods):
    """
    数据处理和生成
    :param data_name: .m file, 'X': n,d; 'Y': n,1
    :return: 'X': n,d; 'Y': n,1
    """
    data_old = scio.loadmat(data_name + '.mat')
    label = data_old["Y"].astype('int')  # n 1

    label_count = pd.DataFrame(data_old['Y'])
    label_info = label_count.value_counts()
    print(label_info)
    print(label_count)
    unique_label = np.unique(label)
    classes = unique_label.shape[0]

    return label.reshape((label.shape[0],)), classes, label_info


def cluster_evaluation(data, label, classes, idx, cluster_times, feature_nums):
    """
    different feature num with fixed cluster times
    :param classes:
    :param label:
    :param data: n d
    :param idx: feature weight idx
    :param cluster_times: 20
    :param feature_nums: [50, 100, 150, 200, 250, 300]
    """
    nmi_fs_cluster_times = []
    acc_fs_cluster_times = []

    for feature_num in feature_nums:

        x_selected = data[:, idx[:feature_num]]
        nmi_cluster_times = []
        acc_cluster_times = []
        for i in range(cluster_times):
            nmi, acc = evaluation(x_selected, classes, label)
            nmi_cluster_times.append(nmi)
            acc_cluster_times.append(acc)
        print('feature num:', feature_num)
        print(" NMI: {}+/-{}".format(np.mean(nmi_cluster_times, 0), np.std(nmi_cluster_times, 0)))
        print(" ACC: {}+/-{}".format(np.mean(acc_cluster_times, 0), np.std(acc_cluster_times, 0)))

        # if relationship:
        #     from utility.relationship import calculate_erd
        #     rate = calculate_erd(x_selected, label, corr)
        #     print("rate", rate)
        # else:
        #     rate = 0
        nmi_fs_cluster_times.append([np.mean(nmi_cluster_times, 0), np.std(nmi_cluster_times, 0)])
        acc_fs_cluster_times.append([np.mean(acc_cluster_times, 0), np.std(acc_cluster_times, 0)])
    # print('cluster evaluation done')
    return np.array(nmi_fs_cluster_times), np.array(acc_fs_cluster_times)


def cluster_evaluation_cluster(x, label, classes, cluster_times):
    """
    different feature num with fixed cluster times
    :param classes:
    :param label:
    :param data: n d
    :param idx: feature weight idx
    :param cluster_times: 20
    :param feature_nums: [50, 100, 150, 200, 250, 300]
    """

    nmi_cluster_times = []
    acc_cluster_times = []
    for i in range(cluster_times):
        nmi, acc = evaluation(x, classes, label)
        nmi_cluster_times.append(nmi)
        acc_cluster_times.append(acc)
        # print('dim num:', x.shape[1])
        # print(" NMI: {}".format(nmi))
        # print(" ACC: {}".format(acc))
    a = [np.mean(nmi_cluster_times, 0), np.std(nmi_cluster_times, 0)]
    b = [np.mean(acc_cluster_times, 0), np.std(acc_cluster_times, 0)]
    # print(a)
    # print(b)
    # print('cluster evaluation done')
    return np.array(a), np.array(b)


def cluster_evaluation_cluster_nmi_acc(y_predict, label, cluster_times):
    """
    different feature num with fixed cluster times
    :param classes:
    :param label:
    :param data: n d
    :param idx: feature weight idx
    :param cluster_times: 20
    :param feature_nums: [50, 100, 150, 200, 250, 300]
    """

    nmi_cluster_times = []
    acc_cluster_times = []
    for i in range(cluster_times):
        nmi, acc = evaluation_nmi_acc(y_predict, label)
        nmi_cluster_times.append(nmi)
        acc_cluster_times.append(acc)
        # print('dim num:', x.shape[1])
        # print(" NMI: {}".format(nmi))
        # print(" ACC: {}".format(acc))
    a = [np.mean(nmi_cluster_times, 0), np.std(nmi_cluster_times, 0)]
    b = [np.mean(acc_cluster_times, 0), np.std(acc_cluster_times, 0)]
    # print(a)
    # print(b)
    # print('cluster evaluation done')
    return np.array(a), np.array(b)


