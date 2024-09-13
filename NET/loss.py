import torch
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys


def kl_div(mu, var):
    return kl_divergence(Normal(mu, var.sqrt()),
                         Normal(torch.zeros_like(mu), torch.ones_like(var))).sum(dim=1).mean()


def cor(m):
    m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def NNDR_loss(embedding, identity_matrix, size):
    loss = torch.mean(torch.abs(torch.triu(cor(embedding), diagonal=1)))
    loss = loss + 1 / torch.mean(
        torch.abs(embedding - torch.mean(embedding, dim=0).view(1, size).repeat(embedding.size()[0], 1)))
    loss = loss + torch.mean(torch.abs(embedding))
    return loss


def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))

    return sim


class scJoint_Loss(nn.Module):  ## ToDo
    def __init__(self, dim=64, p=0.8, use_gpu=True):
        super(scJoint_Loss, self).__init__()
        if use_gpu:
            self.identity_matrix = torch.tensor(np.identity(dim)).float().cuda()
        else:
            self.identity_matrix = torch.tensor(np.identity(dim)).float()
        self.p = p
        self.dim = dim

    def forward(self, RNA_embeddings, ATAC_embeddings):

        # RNA
        RNA_embedding_cat = RNA_embeddings[0]
        RNA_NNDR_loss = NNDR_loss(RNA_embeddings[0], self.identity_matrix, self.dim)

        for i in range(1, len(RNA_embeddings)):
            RNA_embedding_cat = torch.cat([RNA_embedding_cat, RNA_embeddings[i]], 0)
            RNA_NNDR_loss += NNDR_loss(RNA_embeddings[i], self.identity_matrix, self.dim)

        RNA_NNDR_loss /= len(RNA_embeddings)

        # ATAC
        ATAC_NNDR_loss = NNDR_loss(ATAC_embeddings[0], self.identity_matrix, self.dim)

        for i in range(1, len(ATAC_embeddings)):
            ATAC_NNDR_loss += NNDR_loss(ATAC_embeddings[0], self.identity_matrix, self.dim)

        ATAC_NNDR_loss /= len(ATAC_embeddings)

        # cosine similarity loss

        top_k_sim = torch.topk(
            torch.max(cosine_sim(ATAC_embeddings[0], RNA_embedding_cat), dim=1)[0],
            int(ATAC_embeddings[0].shape[0] * self.p)
        )
        sim_loss = torch.mean(top_k_sim[0])

        for i in range(1, len(ATAC_embeddings)):
            top_k_sim = torch.topk(
                torch.max(cosine_sim(ATAC_embeddings[i], RNA_embedding_cat), dim=1)[0],
                int(ATAC_embeddings[0].shape[0] * self.p)
            )
            sim_loss += torch.mean(top_k_sim[0])

        sim_loss = sim_loss / len(ATAC_embeddings)

        loss = RNA_NNDR_loss + ATAC_NNDR_loss - sim_loss

        return loss


class ZINBLoss(nn.Module):  ## ToDo
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        # scale_factor = scale_factor[:, None]
        # mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss




