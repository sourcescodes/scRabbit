from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import scanpy as sc
import numpy as np
import seaborn as sns
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd


def label_transfer(ref, query, rep='embedding', label='cell_type'):
    '''
    Label transfer

    parameters
    ----------
    ref:
        reference containing the projected representations and labels, Default use RNA data
    query:
        query data to transfer label
    rep:
        low-dimension representations, Default:embedding
    label:
        label name
    return:
        transfered label
    '''

    RNA_embedding = ref.obsm[rep]
    RNA_label = ref.obs[label]
    ATAC_embedding = query.obsm[rep]

    knn = KNeighborsClassifier().fit(RNA_embedding, RNA_label)
    ATAC_predict_label = knn.predict(ATAC_embedding)

    return ATAC_predict_label


def evaluation_scRabbit(adata, ref='RNA', query='ATAC', rep='embedding'):

    '''
    evaluation
    -----------
    return:
    '''


    RNA = adata[adata.obs['source'] == ref]
    ATAC = adata[adata.obs['source'] == query]
    ATAC_predict_label = label_transfer(RNA, ATAC, label='cell_type', rep=rep)
    print('F1:', f1_score(ATAC.obs['cell_type'], ATAC_predict_label, average='micro'))
    print('ARI:', adjusted_rand_score(ATAC.obs['cell_type'], ATAC_predict_label))
    print('NMI:', normalized_mutual_info_score(ATAC.obs['cell_type'], ATAC_predict_label))
    print('Batch Entropy:', batch_entropy_mixing_score(adata.obsm['X_umap'], adata.obs['source']))
    print('Silhouette:', silhouette(adata.obsm['X_umap'], adata.obs['cell_type']))


def batch_entropy_mixing_score(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    """
    Calculate batch entropy mixing score

    Algorithm
    ---------
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.

    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.

    Returns
    -------
    Batch entropy mixing score
    """

    #     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i] / P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i] / P[i]) / a
            entropy = entropy - adapt_p[i] * np.log(adapt_p[i] + 10 ** -8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
        P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
        [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)

    return Score / float(np.log2(N_batches))

def silhouette(
        X,
        cell_type,
        metric='euclidean',
        scale=True
):
    """
    Wrapper for sklearn silhouette function values range from [-1, 1] with
        1 being an ideal fit
        0 indicating overlapping clusters and
        -1 indicating misclassified cells
    By default, the score is scaled between 0 and 1. This is controlled `scale=True`

    :param group_key: key in adata.obs of cell labels
    :param embed: embedding key in adata.obsm, default: 'X_pca'
    :param scale: default True, scale between 0 (worst) and 1 (best)
    """
    asw = silhouette_score(
        X,
        cell_type,
        metric=metric
    )
    if scale:
        asw = (asw + 1) / 2
    return asw






