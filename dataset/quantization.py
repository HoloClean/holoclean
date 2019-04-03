import time

from sklearn.cluster import KMeans
from utils import NULL_REPR


# TODO(stoke):currently do only 1D data,
#  will quantize N-dimensional data
def quantize_km(df_raw, bin_number_dict, numerical_attrs):
    """
    Kmeans clustering using sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    Currently do 1D clustering
    :param df_raw: pandas.dataframe
    :param bin_number_dict: a dict existing requested number of bins for each numerical attrs,
    the numerical attrs not in this dict will not do quantization
    :param numerical_attrs: str list, containing all numerical attrs in df_raw
    :return: pandas.dataframe after quantization
    """
    tic = time.time()
    df_quantized = df_raw.copy()

    for col in numerical_attrs:
        if col not in bin_number_dict:
            continue
        df_col = df_quantized.loc[df_quantized[col] != NULL_REPR, [col]].astype(float)
        df_col = df_col.reset_index(drop=True)

        bin_number = bin_number_dict[col]
        n_clusters = min(bin_number, df_col[col].nunique())

        km = KMeans(n_clusters=n_clusters)
        km.fit(df_col)

        label_pred = km.labels_
        centroids = km.cluster_centers_

        def quantize_row(row):
            # row.name can get index, but index is larger than the length
            return str(centroids[label_pred[row.name], 0])
        df_quantized.loc[df_quantized[col] != NULL_REPR, col] = df_col.apply(quantize_row, axis=1).values

    status = "DONE with quantization"
    toc = time.time()
    return status, toc - tic, df_quantized
