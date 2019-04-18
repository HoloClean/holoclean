import time

import numpy as np
from sklearn.cluster import KMeans
from utils import NULL_REPR


def quantize_km(env, df_raw, bin_number_dict):
    """
    Kmeans clustering using sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    Currently do 1D clustering
    :param df_raw: pandas.dataframe
    :param bin_number_dict: a dict existing requested number of bins for each numerical attrs,
    the numerical attrs not in this dict will not do quantization
    :return: pandas.dataframe after quantization
    """
    tic = time.time()
    df_quantized = df_raw.copy()

    for col in bin_number_dict.keys():
        df_col = df_quantized.loc[df_quantized[col] != NULL_REPR, [col]].reset_index(drop=True)
        # Matrix of possibly n-dimension values
        X_col = np.array(list(map(lambda v: v.split(env['numerical_sep']), df_col[col].values)),
                         dtype=np.float32)

        bin_number = bin_number_dict[col]
        n_clusters = min(bin_number, np.unique(X_col, axis=0).shape[0])

        km = KMeans(n_clusters=n_clusters)
        km.fit(X_col)

        label_pred = km.labels_
        centroids = km.cluster_centers_

        # Lookup cluster centroids and concatenate their values again with
        # env['numerical_sep'].
        def quantize_row(row):
            return env['numerical_sep'].join(map(str, centroids[label_pred[row.name]]))
        df_quantized.loc[df_quantized[col] != NULL_REPR, col] = df_col.apply(quantize_row, axis=1).values

    status = "DONE with quantization"
    toc = time.time()
    return status, toc - tic, df_quantized
