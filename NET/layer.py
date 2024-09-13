import math
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function

activation = {
    'relu': nn.ReLU(),
    'rrelu': nn.RReLU(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'tanh': nn.Tanh(),
    '': None
}


class ReverseLayerF(Function):
    '''
        reverseLayer for domain adaptation
    '''

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DSBatchNorm(nn.Module):
    """
        Domain-specific Batch Normalization
    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """
        Parameters
        ----------
        num_features
            dimension of the features
        n_domain
            domain number
        """
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)  # 一个X * features的全零矩阵
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
        return out


class Block(nn.Module):
    """
    Basic block consist of:
        fc -> bn -> act -> dropout
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            norm='',
            act='',
            dropout=0
    ):
        """
        Parameters
        ----------
        input_dim
            dimension of input
        output_dim
            dimension of output
        norm
            batch normalization,
                * '' represent no batch normalization
                * 1 represent regular batch normalization
                * int>1 represent domain-specific batch normalization of n domain
        act
            activation function,
                * relu -> nn.ReLU
                * rrelu -> nn.RReLU
                * sigmoid -> nn.Sigmoid()
                * leaky_relu -> nn.LeakyReLU()
                * tanh -> nn.Tanh()
                * '' -> None
        dropout
            dropout rate
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if type(norm) == int:
            if norm == 1:
                self.norm = nn.BatchNorm1d(output_dim)
            else:
                self.norm = DSBatchNorm(output_dim, norm)
        else:
            self.norm = None

        self.act = activation[act]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, y=None):
        h = self.fc(x)
        if self.norm:
            if len(x) == 1:
                pass
            elif self.norm.__class__.__name__ == 'DSBatchNorm':
                h = self.norm(h, y)
            else:
                h = self.norm(h)
        if self.act:
            h = self.act(h)
        if self.dropout:
            h = self.dropout(h)
        return h


class NN(nn.Module):
    """
    Neural network consist of multi Blocks
    """

    def __init__(self, input_dim, cfg):
        """
        Parameters
        ----------
        input_dim
            input dimension
        cfg
            model structure configuration, 'fc' -> fully connected layer

        Example
        -------
        # >>> latent_dim = 64
        # >>> dec_cfg = [['fc', x_dim, n_domain, 'sigmoid']]
        # >>> decoder = NN(latent_dim, dec_cfg)
        """
        super().__init__()
        net = []
        for i, layer in enumerate(cfg):
            if i == 0:
                d_in = input_dim
            if layer[0] == 'fc':
                net.append(Block(d_in, *layer[1:]))
            d_in = layer[1]
        self.net = nn.ModuleList(net)

    def forward(self, x, y=None):
        for layer in self.net:
            x = layer(x, y)
        return x


class Encoder(nn.Module):
    """
    VAE Encoder
    """

    def __init__(self, input_dim, cfg):
        """
        Parameters
        ----------
        input_dim
            input dimension
        cfg
            encoder configuration, e.g. enc_cfg = [['fc', 1024, 1, 'relu'],['fc', 64, '', '']]
        """
        super().__init__()
        h_dim = cfg[-2][1]
        self.enc = NN(input_dim, cfg[:-1])
        self.mu_enc = NN(h_dim, cfg[-1:])
        self.var_enc = NN(h_dim, cfg[-1:])

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, y=None):
        """
        """
        q = self.enc(x, y)
        mu = self.mu_enc(q, y)
        var = torch.exp(self.var_enc(q, y))
        z = self.reparameterize(mu, var)
        return z, mu, var


class Discriminator(nn.Module):
    def __init__(self, input_size=64, dropout_rate=0.5):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(input_size, 32),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.Drop1 = nn.Dropout(p=dropout_rate)

    def forward(self, input, alpha=1):
        reverse_input = ReverseLayerF.apply(input, alpha)
        yhat = self.Drop1(self.discriminator(reverse_input))
        return torch.sigmoid(yhat)


class Discriminator_recon(nn.Module):   ## To Do
    def __init__(self, input_size=2000, dropout_rate=0.5):
        super(Discriminator_recon, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input):
        yhat = self.discriminator(input)
        return yhat


class Discriminator_W(nn.Module):   ## To Do
    def __init__(self, n_features, d_model=64, n_batches=2):
        """
            Parameters
            ----------
            n_features  : int, the number of features, e.g., genes.

            d_model: int, default 256. The number of dimensions of the latent space.

            n_batches: int, the number of batches, e.g., RNA and ATAC.

        """
        super(Discriminator_W, self).__init__()
        self.n_feature = n_features
        self.d_model = d_model
        self.n_batches = n_batches

        self.model = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, n_batches),
        )

    def forward(self, x):
        return self.model(x)


