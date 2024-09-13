import os
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse, csr

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from anndata import AnnData
import scanpy as sc
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

from glob import glob


def concat_data(
        data_list,
        batch_categories=None,
        join='inner',
        batch_key='batch',
        index_unique=None,
        save=None
):
    """
        Concatenate multiple datasets along the observations axis with name ``batch_key``.

        Parameters
        ----------
        data_list
            A path list of AnnData matrices to concatenate with. Each matrix is referred to as a “batch”.
        batch_categories
            Categories for the batch annotation. By default, use increasing numbers.
        join
            Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
        batch_key
            Add the batch annotation to obs using this key. Default: 'batch'.
        index_unique
            Make the index unique by joining the existing index names with the batch category, using index_unique='-', for instance. Provide None to keep existing indices.
        save
            Path to save the new merged AnnData. Default: None.

        Returns
        -------
        New merged AnnData.
        """

    adata_list = []
    for root in data_list:
        if isinstance(root, AnnData):
            adata = root
        else:
            print("there is not an Anndata object!")
        adata_list.append(adata)
    if batch_categories is None:
        batch_categories = list(map(str, range(len(adata_list))))
    else:
        assert len(adata_list) == len(batch_categories)

    concat = AnnData.concatenate(*adata_list, join=join, batch_key=batch_key,
                                 batch_categories=batch_categories, index_unique=index_unique)
    print(concat)
    return concat


def preprocessing(
        adata: AnnData,
        min_features: int = 200,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features: int = 2000,
        backed=False,
        log=None
):
    """
        Preprocessing single-cell RNA-seq data

        Parameters
        ----------
        adata
            An AnnData matrice of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
        min_features
            Filtered out cells that are detected in less than n genes. Default: 200.
        min_cells
            Filtered out genes that are detected in less than n cells. Default: 3.
        target_sum
            After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization.
        n_top_features
            Number of highly-variable genes to keep. Default: 2000.
        log
            If evaluation, record each operation in the evaluation file. Default: None.

        Return
        -------
        The AnnData object after preprocessing.
        """
    if min_features is None: min_features = 200
    if min_cells is None: min_cells = 3

    if log: log.info("preprocessing.....")

    # Convert adata.X to a sparse matrix in CSR format.
    if type(adata.X) != csr.csr_matrix and (not backed) and (not adata.isbacked):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    # Filter gene columns to remove mitochondrial genes.
    adata = adata[:, [gene for gene in adata.var_names
                      if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    if log: log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_features)

    if log: log.info('Filtering features')
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if log: log.info('Normalizing total per cell')
    sc.pp.normalize_total(adata, target_sum=target_sum)

    if log: log.info('Log1p transforming')
    sc.pp.log1p(adata)

    # Store data and freeze the state of the AnnData object.
    adata.raw = adata

    if log: log.info('Finding variable features')
    if type(n_top_features) == int and n_top_features > 0:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key='batch', inplace=False, subset=True)
    else:
        print('n_top_features has wrong format')

    if log: log.info('Batch specific maxabs scaling')
    # adata = batch_scale(adata, chunk_size=chunk_size)
    adata.X = MaxAbsScaler().fit_transform(adata.X)
    if log: log.info('Processed dataset shape: {}'.format(adata.shape))

    return adata


def batch_scale(adata):
    for b in adata.obs['batch'].unique():
        idx = np.where(adata.obs['batch'] == b)[0]
        scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])  # Perform batch-specific scaling for each batch of cells.
        adata.X[idx] = scaler.transform(adata.X[idx])

    return adata


class BatchSampler(Sampler):
    '''
    Batch-specific Sampler
    sampled data of each batch is from the same dataset.
    '''

    def __init__(self, batch_size, batch_id, drop_last=False):
        """
            create a BatchSampler object

            Parameters
            ----------
            batch_size
                batch size for each sampling
            batch_id
                batch id of all samples : adata.obs['batch']
            drop_last
                drop the last samples that not up to one batch
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_id = batch_id

    def __iter__(self):
        batch = {}
        sampler = np.random.permutation(len(self.batch_id))
        for idx in sampler:
            c = self.batch_id[idx]
            if c not in batch:
                batch[c] = []
            batch[c].append(idx)

            if len(batch[c]) == self.batch_size:
                yield batch[c]
                batch[c] = []

        for c in batch.keys():
            if len(batch[c]) > 0 and not self.drop_last:
                yield batch[c]

    def __len__(self):
        if self.drop_last:
            return len(self.batch_id) // self.batch_size
        else:
            return len(self.batch_id + self.batch_size - 1) // self.batch_size


class SingleCellDataset(Dataset):
    '''
    Dataloader of single-cell data
    '''

    def __init__(self, adata, use_layer='X'):
        """
        create a SingleCellDataset object

        Parameters
        ----------
        adata
            AnnData object wrapping the single-cell data matrix
        """
        self.adata = adata
        self.shape = adata.shape
        self.use_layer = use_layer

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        if self.use_layer == 'X':
            if isinstance(self.adata.X[idx], np.ndarray):
                x = self.adata.X[idx].squeeze().astype(float)
            else:
                x = self.adata.X[idx].toarray().squeeze().astype(float)

        else:
            if self.use_layer in self.adata.layers:
                x = self.adata.layers[self.use_layer][idx]
            else:
                x = self.adata.obsm[self.use_layer][idx]
        domain_id = self.adata.obs['batch'].cat.codes.iloc[idx]

        return x, domain_id, idx


class SingleCellDataset_Vertical(Dataset):
    def __init__(self, adatas):
        self.adatas = adatas

    def __len__(self):
        return self.adatas[0].shape[0]

    def __getitem__(self, idx):
        x = self.adatas[0].X[idx].toarray().squeeze()

        for i in range(1, len(self.adatas)):
            x = np.concatenate((x, self.adatas[i].X[idx].toarray().squeeze()))

        return x, idx


def load_data(
        data_list,
        batch_categories=None,
        max_gene=None,
        join='inner',
        batch_key='batch',
        batch_name='batch',
        min_features=200,
        min_cells=3,
        target_sum=None,
        n_top_features=2000,
        backed=False,
        batch_size=64,
        fraction=None,
        n_obs=None,
        processed=False,
        log=None,
        use_layer='X',
        mode='c'
):
    """
    Load dataset with preprocessing

    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a 'batch'.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_key
        Add the batch annotation to obs using this key. Default: 'batch'.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    min_features
        Filtered out cells that are detected in less than min_features. Default: 600.
    min_cells
        Filtered out genes that are detected in less than min_cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    batch_size
        Number of samples per batch to load. Default: 64.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    log
        If evaluation, record each operation in the evaluation file. Default: None.

    Returns
    -------
    adata
        The AnnData object after combination and preprocessing.
    trainloader
        An iterable over the given dataset for training.
    testloader
        An iterable over the given dataset for testing
    """

    if mode == 'c':
        adata = concat_data(data_list, batch_categories, join=join, batch_key=batch_key)
        # adata = data_list[0].concatenate(data_list[1], join='inner', batch_key='domain_id')
        if log: log.info('Raw dataset shape:{}'.format(adata.shape))
        if batch_name != 'batch':
            if ',' in batch_name:
                names = batch_name.split(',')
                adata.obs['batch'] = adata.obs[names[0]].astype(str) + '_' + adata.obs[names[1]].astype(str)
            else:
                adata.obs['batch'] = adata.obs[batch_name]

        if 'batch' not in adata.obs:
            adata.obs['batch'] = 'batch'
        adata.obs['batch'] = adata.obs['batch'].astype('category')

        if log: log.info(
            'There are {} batches under batch_name: {}'.format(len(adata.obs['batch'].cat.categories), batch_name))

        if n_obs is not None or fraction is not None:
            sc.pp.subsample(adata, fraction=fraction, n_obs=n_obs)

        if not processed and use_layer == 'X':
            adata = preprocessing(
                adata,
                min_features=min_features,
                min_cells=min_cells,
                target_sum=target_sum,
                n_top_features=n_top_features,
                backed=backed,
                log=log,
            )
        else:
            if use_layer in adata.layers:
                adata.layers[use_layer] = MaxAbsScaler().fit_transform(adata.layers[use_layer])
            elif use_layer in adata.obsm:
                adata.obsm[use_layer] = MaxAbsScaler().fit_transform(adata.obsm[use_layer])
            else:
                raise ValueError("Not support use_layer: `{}` yet".format(use_layer))

        scdata = SingleCellDataset(adata, use_layer=use_layer)

        trainloader = DataLoader(
            scdata,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        batch_sampler = BatchSampler(batch_size, adata.obs['batch'], drop_last=False)
        testloader = DataLoader(scdata, batch_sampler=batch_sampler)

        return adata, trainloader, testloader
