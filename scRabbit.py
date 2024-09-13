from multiprocessing import freeze_support
import torch
import numpy as np
import os
import scanpy as sc
from anndata import AnnData
from typing import Union, List
from data import load_data
from logger import create_logger
from NET.VAE import VAE
from NET.utils import EarlyStopping
import pandas as pd
import torch.nn as nn
import evaluation


def scRabbit(
        data_list: Union[str, AnnData, List] = None,
        batch_categories: List = None,
        batch_name: str = 'batch',
        min_features: int = 200,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_feature: int = 2000,
        join: str = 'inner',
        batch_key: str = 'batch',
        processed: bool = False,
        fraction: float = None,
        n_obs: int = None,
        use_layer: str = 'X',
        backed: bool = False,
        batch_size: int = 64,
        lr: float = 2.2e-4,
        seed: int = 256,
        gpu: int = 0,
        outdir: str = None,
        verbose: bool = False,
        eval: bool = False,
        ignore_umap: bool = False,
        show: bool = True,
        ref_id=None,
        impute=False,
        lambda_recon=1,
        lambda_kl=0.5,
        lambda_domain=55,
        modalities='rna_atac'
):
    # init

    np.random.seed(seed)
    torch.manual_seed(seed)
    num_cell = []
    num_gene = []

    for i, adata in enumerate(data_list):
        num_cell.append(adata.X.shape[0])
        num_gene.append(adata.X.shape[1])

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(gpu)
        print('using gpu for training')
    else:
        device = 'cpu'

    if outdir:
        os.makedirs(os.path.join(outdir, 'checkpoint'), exist_ok=True)
        log = create_logger('scRabbit', fh=os.path.join(outdir, 'log.txt'), overwrite=True)
    else:
        log = create_logger('scRabbit')

    adata, trainloader, testloader = load_data(
        data_list=data_list,
        batch_categories=batch_categories,
        join=join,
        target_sum=target_sum,
        n_top_features=n_top_feature,
        batch_size=batch_size,
        min_features=min_features,
        min_cells=min_cells,
        fraction=fraction,
        n_obs=n_obs,
        processed=processed,
        use_layer=use_layer,
        backed=backed,
        batch_name=batch_name,
        batch_key=batch_key,
        log=log,
        max_gene=max(num_gene)
    )

    early_stopping = EarlyStopping(patience=10,
                                   checkpoint_file=os.path.join(outdir, 'checkpoint/model.pt') if outdir else None)
    input_dim = adata.shape[1] if (use_layer == 'X' or use_layer in adata.layers) else adata.obsm[use_layer].shape[1]
    n_domain = len(adata.obs['batch'].cat.categories)

    if ref_id is None:
        ref_id = n_domain - 1

    # model config
    encoder = [['fc', 1024, 1, 'relu'], ['fc', 64, '', '']]
    decoder = [['fc', input_dim, n_domain, 'sigmoid']]

    model = VAE(encoder, decoder, ref_id=ref_id, n_domain=n_domain)
    print('construct VAE model successfully')
    print('learning rate == {}'.format(lr))
    model.fit(
        adata=adata,
        dataloader=trainloader,
        lr=lr,
        max_iteration=30000,
        device=device,
        early_stopping=early_stopping,
        verbose=verbose,
        num_cell=num_cell,
        num_gene=num_gene,
        testloader=testloader,
        lambda_recon=lambda_recon,
        lambda_kl=lambda_kl,
        lambda_domain=lambda_domain,
        modalities=modalities
    )

    print('finish fit...')

    adata.obsm['embedding'] = model.encodeBatch(testloader, device=device, eval=eval)
    if impute:
        adata.obsm['impute'] = model.encodeBatch(testloader, out='impute', batch_id=impute, device=device, eval=eval)  ## TODO
    model.to('cpu')
    del model

    if not ignore_umap:
        log.info('Plot umap')
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='embedding')
        sc.tl.umap(adata, min_dist=0.1)
        # sc.tl.leiden(adata)
        adata.obsm['X_my_umap'] = adata.obsm['X_umap']

        # UMAP visualization

        # sc.set_figure_params(dpi=80, figsize=(3,3))
        cols = ['source', 'cell_type']
        color = [c for c in cols if c in adata.obs]
        if outdir:
            sc.settings.figdir = outdir
            save = '.png'
        else:
            save = None

        if len(color) > 0:
            sc.pl.umap(adata, color='source')
            sc.pl.umap(adata, color='cell_type')

        if outdir is not None:
            adata.write(os.path.join(outdir, 'scRabbit.h5ad'), compression='gzip')
    return adata


freeze_support()
