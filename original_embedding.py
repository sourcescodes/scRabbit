import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler


def batch_scale(adata, use_rep='X', chunk_size=20000):
    """
    Batch-specific scale data

    Parameters
    ----------
    adata
        AnnData
    use_rep
        use '.X' or '.obsm'
    chunk_size
        chunk large data into small chunks

    """
    for b in adata.obs['source'].unique():
        idx = np.where(adata.obs['source'] == b)[0]
        if use_rep == 'X':
            scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.X[idx[i * chunk_size:(i + 1) * chunk_size]] = scaler.transform(
                    adata.X[idx[i * chunk_size:(i + 1) * chunk_size]])
        else:
            scaler = MaxAbsScaler(copy=False).fit(adata.obsm[use_rep][idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.obsm[use_rep][idx[i * chunk_size:(i + 1) * chunk_size]] = scaler.transform(
                    adata.obsm[use_rep][idx[i * chunk_size:(i + 1) * chunk_size]])
    return adata


def get_original_embedding(RNA_file, ATAC_file, meta_file=None):
    """
        :param RNA_file: origin_RNA.h5ad file
        :param ATAC_file: origin_ATAC.h5ad file
        :param meta_file: cell type file(for each cell)
    """

    RNA = ad.read_h5ad(RNA_file)
    ATAC = ad.read_h5ad(ATAC_file)

    ATAC.obs['cell_type'] = ATAC.obs['cell_type'].astype('category')
    ATAC.obs['domain_id'] = 0
    ATAC.obs['domain_id'] = ATAC.obs['domain_id'].astype('category')
    ATAC.obs['source'] = 'ATAC'

    RNA.obs['cell_type'] = RNA.obs['cell_type'].astype('category')
    RNA.obs['domain_id'] = 1
    RNA.obs['domain_id'] = RNA.obs['domain_id'].astype('category')
    RNA.obs['source'] = 'RNA'

    shared = ATAC.concatenate(RNA, join='inner', batch_key='domain_id')
    print(shared)  # 22518 x 10900
    sc.pp.normalize_total(shared)
    sc.pp.log1p(shared)
    sc.pp.highly_variable_genes(shared, n_top_genes=2000, inplace=False, subset=True)
    sc.pp.pca(shared)
    sc.pp.neighbors(shared)
    sc.tl.umap(shared)
    colors = sc.pl.palettes.default_20
    sc.pl.umap(shared, color='source', palette=colors)
    sc.pl.umap(shared, color='cell_type', palette=colors)

