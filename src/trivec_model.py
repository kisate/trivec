import torch
import torch.nn as nn
from torch.nn import Module


class TriVec(Module):
    """
    TriVec model.
    Knowledge graph link prediction.

    Parameters
    ----------
    dim : int
        Embedding size.
    ent_total : int
        Num of different entities.
    rel_total : int
        Num of different relatives.


    Attributes
    ----------
    dim : int
        Embedding size.
    ent_total : int
        Num of different entities.
    rel_total : int
        Num of different relatives.
    ent_i : nn.Embedding
        i-th embedding for enteties ( i in [1:3] )
        embed_head = embed_tail.
    rel_i : nn.Embedding
        i-th embedding ( i in [1:3] )

    Notes
    -----
    Entities embedding: (e_1, e_2, e_3). No difference between embeddings for
    head node and tail node.
    Relative embedding: (rel_1, rel_2, rel_3).
    Score Function: negative softplus loss
    (e.i. loss(score) = softplus(-y*score), where y = 1 for positive triples
    and -1 for negative) with L3 regularization.

    More info here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7233093/
    """
    def __init__(self, ent_total: int, rel_total: int, dim=100):
        super(TriVec, self).__init__()
        self.dim = dim
        self.ent_total = ent_total
        self.rel_total = rel_total
        self.ent_1 = nn.Embedding(self.ent_total, self.dim)
        self.ent_2 = nn.Embedding(self.ent_total, self.dim)
        self.ent_3 = nn.Embedding(self.ent_total, self.dim)
        self.rel_1 = nn.Embedding(self.rel_total, self.dim)
        self.rel_2 = nn.Embedding(self.rel_total, self.dim)
        self.rel_3 = nn.Embedding(self.rel_total, self.dim)

        nn.init.xavier_uniform_(self.ent_1.weight.data)
        nn.init.xavier_uniform_(self.ent_2.weight.data)
        nn.init.xavier_uniform_(self.ent_3.weight.data)
        nn.init.xavier_uniform_(self.rel_1.weight.data)
        nn.init.xavier_uniform_(self.rel_2.weight.data)
        nn.init.xavier_uniform_(self.rel_3.weight.data)

    def _get_emb(self, h_idx, t_idx, r_idx):
        h_1 = self.ent_1(h_idx)
        h_2 = self.ent_2(h_idx)
        h_3 = self.ent_3(h_idx)
        t_1 = self.ent_1(t_idx)
        t_2 = self.ent_2(t_idx)
        t_3 = self.ent_3(t_idx)
        r_1 = self.rel_1(r_idx)
        r_2 = self.rel_2(r_idx)
        r_3 = self.rel_3(r_idx)

        return h_1, h_2, h_3, t_1, t_2, t_3, r_1, r_2, r_3

    def get_score(self, h_idx, t_idx, r_idx):
        h_1, h_2, h_3, t_1, t_2, t_3, r_1, r_2, r_3 = self._get_emb(h_idx,
                                                                    t_idx,
                                                                    r_idx)

        return (h_1 * r_1 * t_3).sum(dim=1) + (h_2 * r_2 * t_2).sum(dim=1) + (
                h_3 * r_3 * t_1).sum(dim=1)

    @staticmethod
    def _get_indexes_from_data(data):
        h_idx = data[:, 0]
        t_idx = data[:, 2]
        r_idx = data[:, 1]
        return h_idx, t_idx, r_idx

    def forward(self, data):
        h_idx, t_idx, r_idx = self._get_indexes_from_data(data)
        score = self.get_score(h_idx, t_idx, r_idx)
        return score

    def regularization(self, data):
        h_idx, t_idx, r_idx = self._get_indexes_from_data(data)
        h_1, h_2, h_3, t_1, t_2, t_3, r_1, r_2, r_3 = self._get_emb(h_idx,
                                                                    t_idx,
                                                                    r_idx)
        regul = (torch.mean(torch.abs(h_1) ** 3, dim=1) +
                 torch.mean(torch.abs(h_2) ** 3, dim=1) +
                 torch.mean(torch.abs(h_3) ** 3, dim=1) +
                 torch.mean(torch.abs(t_1) ** 3, dim=1) +
                 torch.mean(torch.abs(t_2) ** 3, dim=1) +
                 torch.mean(torch.abs(t_3) ** 3, dim=1) +
                 torch.mean(torch.abs(r_1) ** 3, dim=1) +
                 torch.mean(torch.abs(r_2) ** 3, dim=1) +
                 torch.mean(torch.abs(r_3) ** 3, dim=1))/9
        return regul

    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.numpy()
