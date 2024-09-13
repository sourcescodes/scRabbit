import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.autonotebook import trange
from tqdm.contrib import tenumerate
from collections import defaultdict
from .layer import *
from .loss import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


class VAE(nn.Module):
    """
    Variational autoencoder framework.
    """

    def __init__(self, enc, dec, ref_id, n_domain=2):
        """
        Parameters
        ----------
        enc
            Encoder structure config.
        dec
            Decoder structure config.
        ref_id
            ID of the reference dataset.
        n_domain
            The number of different domains.
        """

        '''
            enc_cfg = [['fc', 1024, 1, 'relu'],['fc', 64, '', '']]
            dec_cfg = [['fc', 2000, 2, 'sigmoid']]
        '''
        super().__init__()
        x_dim = dec[-1][1]
        z_dim = enc[-1][1]
        self.encoder = Encoder(x_dim, enc)
        self.decoder = NN(z_dim, dec)
        self.discriminator = Discriminator(input_size=64)
        self.n_domain = n_domain
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.NNDR = scJoint_Loss(dim=64, p=0.8, use_gpu=True)
        self.discriminator_recon = Discriminator_recon(2000)
        self.ZINB_loss = ZINBLoss()
        self.ref_id = ref_id

    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters.
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def encodeBatch(
            self,
            dataloader,
            num_gene=None,
            device='cuda',
            out='latent',
            batch_id=None,
            return_idx=False,
            eval=False,
            mode='c'
    ):

        """
        Parameters
        ----------
        dataloader
            An iterable over the given dataset.
        num_gene
            The number of genes contained in different domains.
        device
            'cuda' or 'cpu' for training. Default: 'cuda'.
        out
            The inference layer for output. If 'latent', output latent feature z. If 'impute', output imputed gene expression matrix. Default: 'latent'.
        batch_id
            If None, use batch 0 decoder to infer for all samples. Else, use the corresponding decoder according to the sample batch id to infer for each sample.
        return_idx
            Whether return the dataloader sample index. Default: False.
        eval
            If True, set the model to evaluation mode. If False, set the model to train mode. Default: False.
        mode
            Choose from ['c', 's']
            If 'c', integrate data with common genes.
            If 's', integrate data with common genes and specific genes.
            Default: 'c'.

        Returns
        -------
        Inference layer 'latent' or 'impute'.
        """
        self.to(device)
        if eval:
            self.eval()
            print('eval mode')
        else:
            self.train()
        indices = np.zeros(dataloader.dataset.shape[0])
        if out == 'latent':
            if mode == 'c':
                output = np.zeros((dataloader.dataset.shape[0], self.z_dim))

                for x, y, idx in dataloader:
                    x = x.float().to(device)
                    z = self.encoder(x)[1]  # z, mu, var
                    output[idx] = z.detach().cpu().numpy()
                    indices[idx] = idx
            elif mode == 's':
                output = np.zeros((dataloader.dataset.shape[0], self.z_dim))
                for x, y, idx in dataloader:
                    x_c = x[:, 0:num_gene[self.n_domain]].float().to(device)
                    z = self.encoder(x_c, 0)[1]
                    output[idx] = z.detach().cpu().numpy()
        elif out == 'impute':  ## To Do
            output = np.zeros((dataloader.dataset.shape[0], self.x_dim))

            if batch_id in dataloader.dataset.adata.obs['batch'].cat.categories:
                batch_id = list(dataloader.dataset.adata.obs['batch'].cat.categories).index(batch_id)
            else:
                batch_id = 0

            for x, y, idx in dataloader:
                x = x.float().to(device)
                z = self.encoder(x)[1]
                output[idx] = self.decoder(z, torch.LongTensor([batch_id] * len(z))).detach().cpu().numpy()
                indices[idx] = idx

        return output

    def fit(
            self,
            adata,
            dataloader,
            testloader,
            num_cell,
            num_gene,
            lr=2.2e-4,
            max_iteration=30000,
            early_stopping=None,
            device='cuda',
            verbose=False,
            Prior=None,
            mode='c',
            lambda_recon=1,
            lambda_kl=0.5,
            lambda_domain=55,
            modalities='rna_atac'
    ):
        """
        train scRabbit

        Parameters
        ----------
        adata
            h5ad file prepared to integrate.
        dataloader
            An iterable over the given dataset for training.
        testloader
            An iterable over the given dataset for evaluation.
        num_cell
            The number of cells contained in different domains.
        num_gene
            The number of genes contained in different domains.
        lr
            Learning rate. Default: 2.2e-4.
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None.
        device
            'cuda' or 'cpu' for training. Default: 'cuda'.
        verbose
            Verbosity, True or False. Default: False.
        Prior
            Prior correspondence matrix.
        mode
            Choose from ['c', 's']
            If 'c', integrate data with common genes.
            If 's', integrate data with common genes and specific genes.
            Default: 'c'.
        lambda_recon
            Balanced parameter for reconstruction. Default: 1.0.
        lambda_kl
            Balanced parameter for KL divergence. Default: 0.5.
        lambda_domain
            Balanced parameter for domain classify. Default: 55.
        modalities
            the modality of integrated data.
            Choose from ['rna_atac', 'rna_spatial']
            If 'rna_atac', integrate scRNA-seq and scATAC-seq.
            If 'rna_spatial', integrate scRNA-seq and single-cell spatial transcriptomics.
            Default: 'rna_atac'

        """
        if mode == 'c':
            self.to(device)
            optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
            self.D = Discriminator_W(n_features=64, d_model=64, n_batches=2).to(device=device)
            optD = torch.optim.Adam(self.D.parameters(), lr=lr, weight_decay=5e-4)
            reg_crit = L1regularization().cuda()
            n_epoch = 310
            if modalities == 'rna_spatial':
                n_epoch = int(np.ceil(max_iteration / len(dataloader)))
                lambda_kl = 5.0
            with trange(n_epoch, total=n_epoch, desc='Epochs') as tq:
                i = 0
                for epoch in tq:

                    p = float(i + epoch * len(dataloader)) / n_epoch / len(dataloader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1  # alpha for DANN

                    tk0 = tenumerate(dataloader, total=len(dataloader), leave=False, desc='Iterations',
                                     disable=(not verbose))
                    epoch_loss = defaultdict(float)
                    for i, (x, y, idx) in tk0:
                        batch_RNA = []
                        batch_ATAC = []
                        x, y = x.float().to(device), y.long().to(device)
                        idx = idx.to(device)
                        batch_RNA_id_list = []
                        batch_ATAC_id_list = []
                        for id in range(len(y)):
                            if y[id] == 0:
                                batch_RNA.append(x[id])
                                batch_RNA_id_list.append(id)
                            else:
                                batch_ATAC.append(x[id])
                                batch_ATAC_id_list.append(id)

                        # forward

                        z, mu, var = self.encoder(x)
                        recon_x = self.decoder(z, y)
                        domain_prediction = self.discriminator(z, alpha)
                        domain_prediction = domain_prediction.float()
                        y = y.unsqueeze(1)
                        y = y.float()

                        recon_loss = F.binary_cross_entropy(recon_x, x) * 2000
                        kl_loss = kl_div(mu, var)
                        domain_loss = F.binary_cross_entropy(domain_prediction, y)

                        loss = {'recon_loss': lambda_recon * recon_loss, 'kl_loss': lambda_kl * kl_loss,
                                'domain_loss': lambda_domain * domain_loss}

                        optim.zero_grad()
                        sum(loss.values()).backward()
                        optim.step()

                        for k, v in loss.items():
                            epoch_loss[k] += loss[k].item()

                        info = ','.join(['{}={:.3f}'.format(k, v) for k, v in loss.items()])

                    epoch_loss = {k: v / (i + 1) for k, v in epoch_loss.items()}
                    epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k, v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info)

        elif mode == 's':  ## To Do
            self.to(device)
            optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)

            self.D = Discriminator_W(n_features=64, d_model=64, n_batches=2).to(device=device)
            optD = torch.optim.Adam(self.D.parameters(), lr=lr, weight_decay=5e-4)
            reg_crit = L1regularization().cuda()
            n_epoch = int(np.ceil(max_iteration / len(dataloader)))
            n_epoch = 200
            best_result = 0.
            with trange(n_epoch, total=n_epoch, desc='Epochs') as tq:

                i = 0

                for epoch in tq:

                    p = float(i + epoch * len(dataloader)) / n_epoch / len(dataloader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                    tk0 = tenumerate(dataloader, total=len(dataloader), leave=False, desc='Iterations',
                                     disable=(not verbose))
                    epoch_loss = defaultdict(float)

                    for i, (x, y, idx) in tk0:
                        x_c, y = x[:, 0:num_gene[self.n_domain]].float().to(device), y.long().to(device)
                        idx = idx.to(device)

                        loc_ref = torch.where(y == self.ref_id)[0]

                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])

                        loc_query = {}
                        idx_query = {}
                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)
                        tran_batch = {}
                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y == j)[0]
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])
                                tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:, idx_ref].to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref

                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)

                        z, mu, var = self.encoder(x_c)
                        recon_x_c = self.decoder(z, y)

                        recon_x1 = torch.stack(recon_x1)
                        recon_x2 = torch.stack(recon_x2)
                        domain_prediction = self.discriminator(z, alpha)
                        domain_prediction = domain_prediction.float()
                        y = y.unsqueeze(1)
                        y = y.float()

                        recon_loss += F.binary_cross_entropy(recon_x_c, x) * 2000  ## TO DO  1500 can
                        kl_loss = kl_div(mu, var)
                        domain_loss = F.binary_cross_entropy(domain_prediction, y)

                        loss = {'recon_loss': recon_loss, 'kl_loss': 0.5 * kl_loss, 'domain_loss': 30 * domain_loss}  #


                        optim.zero_grad()
                        sum(loss.values()).backward()
                        optim.step()

                        for k, v in loss.items():
                            epoch_loss[k] += loss[k].item()

                        info = ','.join(['{}={:.3f}'.format(k, v) for k, v in loss.items()])

                    epoch_loss = {k: v / (i + 1) for k, v in epoch_loss.items()}
                    epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k, v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info)
                    early_stopping(sum(epoch_loss.values()), self)
