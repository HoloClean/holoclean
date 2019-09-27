import time

import numpy as np
from sklearn.cluster import KMeans
from utils import NULL_REPR


def quantize_km(env, df_raw, num_attr_groups_bins):
    """
    Kmeans clustering using sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    Currently do 1D clustering
    :param df_raw: pandas.dataframe
    :param num_attr_groups_bins: list[tuple] where each tuple consists of
    (# of bins, list[str]) where the list[str] is a group of attribues to be
    treated as numerical.
    Groups must be disjoint.

    :return: pandas.dataframe after quantization
    """
    tic = time.time()
    df_quantized = df_raw.copy()

    # Assert groups are disjoint
    num_attrs = [attr for _, group in num_attr_groups_bins for attr in group]
    assert len(set(num_attrs)) == len(num_attrs)

    for bins, attrs in num_attr_groups_bins:
        fil_notnull = (df_quantized[attrs] != NULL_REPR).all(axis=1)

        df_group = df_quantized.loc[fil_notnull, attrs].reset_index(drop=True)
        # Matrix of possibly n-dimension values
        X_attrs = df_group.values.astype(np.float)

        if bins >= np.unique(X_attrs, axis=0).shape[0]:
            # No need to quantize since more bins than unique values.
            continue

        km = KMeans(n_clusters=bins)
        km.fit(X_attrs)

        label_pred = km.labels_
        centroids = km.cluster_centers_

        # Lookup cluster centroids and replace their values.
        df_quantized.loc[fil_notnull, attrs] = np.array([centroids[label_pred[idx]]
            for idx in df_group.index]).astype(str)

    status = "DONE with quantization"
    toc = time.time()
    return status, toc - tic, df_quantized
