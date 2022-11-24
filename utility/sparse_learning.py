import numpy as np
eps = np.spacing(1)


def feature_ranking(w):
    """
    This function ranks features according to the feature weights matrix W

    Input:
    -----
    W: {numpy array}, shape (n_features, n_classes)
        feature weights matrix

    Output:
    ------
    idx: {numpy array}, shape {n_features,}
        feature index ranked in descending order by feature importance
    """
    t = (w * w).sum(1)
    idx = np.argsort(t, 0)
    return idx[::-1]
